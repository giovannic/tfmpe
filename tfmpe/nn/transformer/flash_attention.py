"""Flash Attention v4 via CuTeDSL + JAX FFI.

Wraps the CuTeDSL flash attention kernels from flash_attn.cute for
Blackwell GPUs (SM100/SM120) and exposes them as a JAX-compatible
attention function matching the Flax attention_fn contract.

Integration path: CuTeDSL kernel → cute.compile --enable-tvm-ffi →
jax_tvm_ffi.register_ffi_target → jax.ffi.ffi_call → jax.custom_vjp.
"""

import math
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import jax
import jax.numpy as jnp
import jax_tvm_ffi
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl.cutlass import Arch
from jaxtyping import Array
from flax.typing import Dtype, PrecisionLike
from flax.nnx.module import Module

from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.flash_bwd_sm120 import FlashAttentionBackwardSm120
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned

# ---------------------------------------------------------------------------
# Monkey-patch flash_attn.cute.utils.atomic_add_fp32 to use the current
# nvvm.atomicrmw signature (positional op/ptr/a, no `res=` kwarg). Without
# this the bwd kernel fails with `atomicrmw() got an unexpected keyword
# argument 'res'` when it walks the dQ accumulator atomics.
# ---------------------------------------------------------------------------

from flash_attn.cute import utils as _fa_utils
from cutlass._mlir.dialects import nvvm as _nvvm
from flash_attn.cute.utils import dsl_user_op as _dsl_user_op


@_dsl_user_op
def _atomic_add_fp32_patched(a, gmem_ptr, *, loc=None, ip=None) -> None:
    _nvvm.atomicrmw(
        op=_nvvm.AtomicOpKind.FADD,
        ptr=gmem_ptr.llvm_ptr,
        a=Float32(a).ir_value(),
    )


_fa_utils.atomic_add_fp32 = _atomic_add_fp32_patched
# The bwd module pulled a reference at import time; patch that too.
from flash_attn.cute import flash_bwd as _fa_bwd  # noqa: E402

if hasattr(_fa_bwd, "utils"):
    _fa_bwd.utils.atomic_add_fp32 = _atomic_add_fp32_patched

# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

_JAX_TO_CUTLASS = {
    jnp.dtype(jnp.float16): cutlass.Float16,
    jnp.dtype(jnp.bfloat16): cutlass.BFloat16,
    jnp.dtype(jnp.float32): cutlass.Float32,
}

_CUTLASS_TO_JAX = {v: k for k, v in _JAX_TO_CUTLASS.items()}

# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------


def _get_arch() -> int:
    """Return SM arch as int (e.g. 100, 120)."""
    # Touch a JAX GPU device so PJRT retains a primary CUDA context on the
    # current thread; otherwise cuCtxGetDevice() returns NOT_INITIALIZED.
    _gpu()
    err, device = cuda.cuCtxGetDevice()
    assert err == cuda.CUresult.CUDA_SUCCESS, (
        f"cuCtxGetDevice failed: {err}"
    )
    err, major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device,
    )
    err, minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device,
    )
    return major * 10 + minor


# ---------------------------------------------------------------------------
# CuTeDSL wrapper classes — reorder args for JAX FFI compatibility
#
# JAX FFI arg_spec ["args", "rets"] expands as:
#   fun(*all_inputs, *all_outputs)
# But the kernel signature interleaves inputs and outputs. The wrappers
# accept FFI-ordered args and delegate to the kernel in its native order.
# ---------------------------------------------------------------------------


class _FwdWrapper:
    """Non-varlen forward wrapper.  Bakes softmax_scale at compile time."""

    def __init__(self, kernel, softmax_scale: float):
        self.kernel = kernel
        self.softmax_scale = softmax_scale

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mQ, mK, mV, mO, mLSE, self.softmax_scale, stream=stream,
        )


class _FwdVarlenWrapper:
    """Varlen forward wrapper with cu_seqlens."""

    def __init__(self, kernel, softmax_scale: float):
        self.kernel = kernel
        self.softmax_scale = softmax_scale

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mQ, mK, mV, mO, mLSE, self.softmax_scale,
            mCuSeqlensQ, mCuSeqlensK, stream=stream,
        )


class _BwdPreWrapper:
    """Non-varlen backward preprocess wrapper.

    Parameter order follows FFI convention (inputs first, outputs after),
    not the kernel's native interleaved order. The kernel call inside
    reorders them: mdPsum/mLSE both have shape (B, H, S) and dtype float32,
    so swapping them doesn't trip compilation — it just produces silent
    NaN gradients by reading LSE from the uninitialised mdPsum output slot.
    """

    def __init__(self, kernel):
        self.kernel = kernel

    @cute.jit
    def __call__(
        self,
        # --- inputs (args) ---
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        # --- outputs (rets) ---
        mdPsum: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdQaccum: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        # FlashAttentionBackwardPreprocess does not call assume_tensor_aligned
        # internally (unlike flash_fwd/flash_bwd main kernels), so the universal
        # copy atom for O/dO sees only element-level alignment and IR
        # verification rejects the 128-bit copy. Assert it here.
        mO, mdO, mdPsum, mLSE, mLSElog2, mdQaccum = [
            assume_tensor_aligned(t)  # type: ignore
            for t in (mO, mdO, mdPsum, mLSE, mLSElog2, mdQaccum)
        ]
        self.kernel(
            mO, mdO, mdPsum, mLSE, mLSElog2, mdQaccum,
            None, None, None,  # cu_seqlens_q, seqused_q, dlse
            stream=stream,
        )


class _BwdPreVarlenWrapper:
    """Varlen backward preprocess wrapper.

    Passes ``mCuSeqlensQ`` so the kernel takes the rank-2 layout branch
    (transpose=[1,0]) for ``mPdPsum`` / ``mLSE`` / ``mLSElog2`` instead of
    the rank-3 branch. Same FFI argument ordering convention as the
    non-varlen wrapper.
    """

    def __init__(self, kernel):
        self.kernel = kernel

    @cute.jit
    def __call__(
        self,
        # --- inputs (args) ---
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mLSE: cute.Tensor,
        # --- outputs (rets) ---
        mdPsum: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdQaccum: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        mO, mdO, mdPsum, mLSE, mLSElog2, mdQaccum = [
            assume_tensor_aligned(t)  # type: ignore
            for t in (mO, mdO, mdPsum, mLSE, mLSElog2, mdQaccum)
        ]
        self.kernel(
            mO, mdO, mdPsum, mLSE, mLSElog2, mdQaccum,
            mCuSeqlensQ, None, None,  # seqused_q, dlse
            stream=stream,
        )


class _BwdPostWrapper:
    """Non-varlen backward postprocess: dQaccum -> dQ.

    The main backward kernel writes dQ into a flat float32 accumulator with
    an MMA-derived layout; a naïve reshape does not recover (batch, seq,
    heads, dim). This kernel reads the accumulator with the matching tiled
    layout and writes dQ in the expected layout, applying softmax_scale and
    casting to the input dtype.
    """

    def __init__(self, kernel, softmax_scale: float):
        self.kernel = kernel
        self.softmax_scale = softmax_scale

    @cute.jit
    def __call__(
        self,
        # --- inputs (args) ---
        mdQaccum: cute.Tensor,
        # --- outputs (rets) ---
        mdQ: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mdQaccum, mdQ, self.softmax_scale,
            None, None,  # cu_seqlens_q, seqused_q
            stream=stream,
        )


class _BwdPostVarlenWrapper:
    """Varlen backward postprocess: dQaccum -> dQ with cu_seqlens."""

    def __init__(self, kernel, softmax_scale: float):
        self.kernel = kernel
        self.softmax_scale = softmax_scale

    @cute.jit
    def __call__(
        self,
        # --- inputs (args) ---
        mdQaccum: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        # --- outputs (rets) ---
        mdQ: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mdQaccum, mdQ, self.softmax_scale,
            mCuSeqlensQ, None,  # seqused_q
            stream=stream,
        )


class _BwdWrapper:
    """Non-varlen backward wrapper.  Bakes softmax_scale.

    ``supports_semaphore`` is False on SM120 (Sm80 kernel path), where the
    class asserts ``mdQ_semaphore is None``. The FFI still takes an unused
    semaphore tensor for a stable ABI; it is simply dropped.

    ``mdQaccum`` is atomically accumulated into by the kernel; it appears
    as both an FFI input and output (aliased via ``input_output_aliases``)
    so JAX observes the mutation after the call returns. The last
    ``mdQaccum_out`` parameter is the aliased output buffer — the kernel
    writes through ``mdQaccum`` (same memory), so this one is unused.
    """

    def __init__(
        self, kernel, softmax_scale: float, supports_semaphore: bool = True,
    ):
        self.kernel = kernel
        self.softmax_scale = softmax_scale
        self.supports_semaphore = supports_semaphore

    @cute.jit
    def __call__(
        self,
        # --- inputs (args) ---
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQ_sem: cute.Tensor,
        mdQaccum: cute.Tensor,
        # --- outputs (rets) ---
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQaccum_out: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        if cutlass.const_expr(self.supports_semaphore):
            self.kernel(
                mQ, mK, mV, mdO, mLSElog2, mdPsum,
                mdQaccum, mdK, mdV,
                self.softmax_scale,
                mdQ_semaphore=mdQ_sem,
                stream=stream,
            )
        else:
            self.kernel(
                mQ, mK, mV, mdO, mLSElog2, mdPsum,
                mdQaccum, mdK, mdV,
                self.softmax_scale,
                stream=stream,
            )


class _BwdVarlenWrapper:
    """Varlen backward wrapper.

    Forwards ``mCuSeqlensQ``/``mCuSeqlensK`` to the Sm80 bwd kernel so it
    takes the rank-3 varlen branch (Q/dO/dQaccum shaped as
    ``(total_q, num_heads, head_dim)`` etc.) instead of the rank-4
    non-varlen branch that would trigger a shape/coord mismatch.
    """

    def __init__(
        self, kernel, softmax_scale: float, supports_semaphore: bool = True,
    ):
        self.kernel = kernel
        self.softmax_scale = softmax_scale
        self.supports_semaphore = supports_semaphore

    @cute.jit
    def __call__(
        self,
        # --- inputs (args) ---
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQ_sem: cute.Tensor,
        mdQaccum: cute.Tensor,
        # --- outputs (rets) ---
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQaccum_out: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        if cutlass.const_expr(self.supports_semaphore):
            self.kernel(
                mQ, mK, mV, mdO, mLSElog2, mdPsum,
                mdQaccum, mdK, mdV,
                self.softmax_scale,
                mCuSeqlensQ=mCuSeqlensQ, mCuSeqlensK=mCuSeqlensK,
                mdQ_semaphore=mdQ_sem,
                stream=stream,
            )
        else:
            self.kernel(
                mQ, mK, mV, mdO, mLSElog2, mdPsum,
                mdQaccum, mdK, mdV,
                self.softmax_scale,
                mCuSeqlensQ=mCuSeqlensQ, mCuSeqlensK=mCuSeqlensK,
                stream=stream,
            )


# ---------------------------------------------------------------------------
# Example tensor helpers (for cute.compile)
# ---------------------------------------------------------------------------

_GPU = None


def _gpu():
    global _GPU
    if _GPU is None:
        _GPU = jax.devices("gpu")[0]
    return _GPU


def _example_tensor(shape, jax_dtype):
    """Create a CuTe tensor from a JAX zero-array for compilation."""
    arr = jnp.zeros(shape, dtype=jax_dtype, device=_gpu())
    t = from_dlpack(arr, assumed_align=16)
    return t.mark_layout_dynamic(leading_dim=len(shape) - 1)


def _example_tensor_int32(shape):
    arr = jnp.zeros(shape, dtype=jnp.int32, device=_gpu())
    t = from_dlpack(arr, assumed_align=4)
    return t.mark_layout_dynamic(leading_dim=0)


# ---------------------------------------------------------------------------
# Forward config helpers
# ---------------------------------------------------------------------------


def _fwd_tile_config_sm100(head_dim, head_dim_v):
    """Tile sizes for SM100 forward (simplified from interface.py)."""
    return 128, 128, 2  # tile_m, tile_n, q_stage


def _fwd_tile_config_sm120(head_dim, head_dim_v):
    """Tile sizes for SM120 forward."""
    if head_dim <= 64:
        return 128, 128, 1
    return 128, 64, 1


# ---------------------------------------------------------------------------
# Kernel compilation and registration
# ---------------------------------------------------------------------------

# Cache: compile_key → (fwd_fn_ref, bwd_fn_ref, attn_fn)
_REGISTERED: dict[tuple, tuple[Any, Any, Any]] = {}


def register_flash_attn_ops(
    head_dim: int,
    num_heads: int,
    dtype=jnp.bfloat16,
    head_dim_v: int | None = None,
    num_heads_kv: int | None = None,
    varlen: bool = False,
) -> None:
    """Compile and register flash attention kernels for a given config.

    Must be called before the first use of ``flash_attention_v4`` with
    this (head_dim, num_heads, dtype) combination.
    """
    head_dim_v = head_dim_v or head_dim
    num_heads_kv = num_heads_kv or num_heads
    jax_dtype = jnp.dtype(dtype)
    arch = _get_arch()
    qhead_per_kvhead = num_heads // num_heads_kv

    cache_key = (jax_dtype.name, head_dim, head_dim_v, qhead_per_kvhead, arch, varlen)
    if cache_key in _REGISTERED:
        return

    cutlass_dtype = _JAX_TO_CUTLASS[jax_dtype]
    softmax_scale = 1.0 / math.sqrt(head_dim)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # ---- Select arch-specific kernel class and tile config ----
    if arch // 10 in (10, 11):
        tile_m, tile_n, q_stage = _fwd_tile_config_sm100(head_dim, head_dim_v)
        FwdCls = FlashAttentionForwardSm100
        BwdCls = FlashAttentionBackwardSm100
        fwd_kwargs = dict(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=False,
            is_local=False,
            is_split_kv=False,
            pack_gqa=False,
            m_block_size=tile_m,
            n_block_size=tile_n,
            q_stage=q_stage,
            is_persistent=not varlen,
            is_varlen_q=varlen,
        )
        # SM100 bwd uses tile_m/tile_n and supports semaphores.
        m_block_bwd = 128
        n_block_bwd = 128
        bwd_kwargs = dict(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            is_causal=False,
            is_local=False,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=m_block_bwd,
            tile_n=n_block_bwd,
        )
        bwd_supports_semaphore = True
    elif arch // 10 == 12:
        tile_m, tile_n, q_stage = _fwd_tile_config_sm120(head_dim, head_dim_v)
        FwdCls = FlashAttentionForwardSm120
        BwdCls = FlashAttentionBackwardSm120
        fwd_kwargs = dict(
            dtype=cutlass_dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=False,
            is_local=False,
            pack_gqa=False,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=1,
            num_threads=128,
            Q_in_regs=False,
        )
        # SM120 reuses the Sm80 bwd class. It takes m_block_size/n_block_size
        # (not tile_m/tile_n), has no is_local, asserts mdQ_semaphore is None,
        # and uses a smaller tile with 128 threads.
        m_block_bwd = 64
        n_block_bwd = 64
        num_stages_Q_bwd = 2 if head_dim <= 64 else 1
        num_stages_dO_bwd = 2 if head_dim <= 64 else 1
        bwd_kwargs = dict(
            dtype=cutlass_dtype,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            m_block_size=m_block_bwd,
            n_block_size=n_block_bwd,
            num_stages_Q=num_stages_Q_bwd,
            num_stages_dO=num_stages_dO_bwd,
            num_threads=128,
            pack_gqa=False,
            is_causal=False,
            SdP_swapAB=False,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=4,
            AtomLayoutNdKV=4,
            AtomLayoutMdQ=4,
            V_in_regs=False,
        )
        bwd_supports_semaphore = False
    else:
        raise ValueError(
            f"flashattention_v4 requires Blackwell (SM100+), got SM{arch}"
        )

    # ---- Forward ----
    fa_fwd = FwdCls(**fwd_kwargs)
    if arch // 10 == 12:
        # FlashAttentionForwardSm120 inherits Sm80's __init__, which sets
        # self.arch = Arch.sm_120a and overrides the intended class attribute
        # `arch = 80`. Later, __call__ computes
        # `use_tma_O = self.arch >= Arch.sm_90` → True, compiling the TMA
        # epilogue branch even though the Sm80 kernel path never builds a
        # tma_atom_O. Forcing self.arch back to sm_80 selects the CpAsync
        # epilogue that this code path actually supports.
        fa_fwd.arch = Arch.sm_80
    M_ex = 32  # example seqlen for compilation (dynamic at runtime)
    if varlen:
        wrapper_fwd = _FwdVarlenWrapper(fa_fwd, softmax_scale)
        ex_q = _example_tensor((M_ex, num_heads, head_dim), jax_dtype)
        ex_k = _example_tensor((M_ex, num_heads_kv, head_dim), jax_dtype)
        ex_v = _example_tensor((M_ex, num_heads_kv, head_dim_v), jax_dtype)
        ex_cu = _example_tensor_int32((3,))  # batch+1
        ex_o = _example_tensor((M_ex, num_heads, head_dim_v), jax_dtype)
        ex_lse = _example_tensor((num_heads, M_ex), jnp.float32)
        fwd_compiled = cute.compile(
            wrapper_fwd,
            ex_q, ex_k, ex_v, ex_cu, ex_cu, ex_o, ex_lse,
            stream, options="--enable-tvm-ffi",
        )
    else:
        wrapper_fwd = _FwdWrapper(fa_fwd, softmax_scale)
        ex_q = _example_tensor((M_ex, M_ex, num_heads, head_dim), jax_dtype)
        ex_k = _example_tensor((M_ex, M_ex, num_heads_kv, head_dim), jax_dtype)
        ex_v = _example_tensor((M_ex, M_ex, num_heads_kv, head_dim_v), jax_dtype)
        ex_o = _example_tensor((M_ex, M_ex, num_heads, head_dim_v), jax_dtype)
        ex_lse = _example_tensor((M_ex, num_heads, M_ex), jnp.float32)
        fwd_compiled = cute.compile(
            wrapper_fwd,
            ex_q, ex_k, ex_v, ex_o, ex_lse,
            stream, options="--enable-tvm-ffi",
        )

    fwd_name = f"cute.fa4_fwd_{cache_key}"
    jax_tvm_ffi.register_ffi_target(
        fwd_name, fwd_compiled,
        arg_spec=["args", "rets"],
        platform="gpu",
    )

    # ---- Backward preprocess ----
    bwd_pre = FlashAttentionBackwardPreprocess(
        cutlass_dtype, head_dim, head_dim_v, m_block_bwd,
    )

    head_dim_rounded = math.ceil(head_dim / 32) * 32
    if not varlen:
        bwd_pre_wrapper = _BwdPreWrapper(bwd_pre)
        ex_out = _example_tensor((M_ex, M_ex, num_heads, head_dim_v), jax_dtype)
        ex_dout = _example_tensor((M_ex, M_ex, num_heads, head_dim_v), jax_dtype)
        ex_dpsum = _example_tensor((M_ex, num_heads, M_ex), jnp.float32)
        ex_lse_pre = _example_tensor((M_ex, num_heads, M_ex), jnp.float32)
        ex_lse_log2 = _example_tensor((M_ex, num_heads, M_ex), jnp.float32)
        seqlen_rounded = math.ceil(M_ex / m_block_bwd) * m_block_bwd
        ex_dq_accum = _example_tensor(
            (M_ex, num_heads, seqlen_rounded * head_dim_rounded), jnp.float32,
        )
        bwd_pre_compiled = cute.compile(
            bwd_pre_wrapper,
            ex_out, ex_dout, ex_lse_pre,          # inputs
            ex_dpsum, ex_lse_log2, ex_dq_accum,   # outputs
            stream, options="--enable-tvm-ffi",
        )
    else:
        bwd_pre_wrapper = _BwdPreVarlenWrapper(bwd_pre)
        ex_out = _example_tensor((M_ex, num_heads, head_dim_v), jax_dtype)
        ex_dout = _example_tensor((M_ex, num_heads, head_dim_v), jax_dtype)
        ex_cu_pre = _example_tensor_int32((3,))  # batch+1
        ex_dpsum = _example_tensor((num_heads, M_ex), jnp.float32)
        ex_lse_pre = _example_tensor((num_heads, M_ex), jnp.float32)
        ex_lse_log2 = _example_tensor((num_heads, M_ex), jnp.float32)
        seqlen_rounded = math.ceil(M_ex / m_block_bwd) * m_block_bwd
        ex_dq_accum = _example_tensor(
            (num_heads, seqlen_rounded * head_dim_rounded), jnp.float32,
        )
        bwd_pre_compiled = cute.compile(
            bwd_pre_wrapper,
            ex_out, ex_dout, ex_cu_pre, ex_lse_pre,   # inputs
            ex_dpsum, ex_lse_log2, ex_dq_accum,       # outputs
            stream, options="--enable-tvm-ffi",
        )
    bwd_pre_name = f"cute.fa4_bwd_pre_{cache_key}"
    jax_tvm_ffi.register_ffi_target(
        bwd_pre_name, bwd_pre_compiled,
        arg_spec=["args", "rets"],
        platform="gpu",
    )

    # ---- Backward main ----
    fa_bwd = BwdCls(**bwd_kwargs)

    # Example tensors for backward compile
    if not varlen:
        bwd_wrapper = _BwdWrapper(
            fa_bwd, softmax_scale, supports_semaphore=bwd_supports_semaphore,
        )
        ex_bq = _example_tensor((M_ex, M_ex, num_heads, head_dim), jax_dtype)
        ex_bk = _example_tensor((M_ex, M_ex, num_heads_kv, head_dim), jax_dtype)
        ex_bv = _example_tensor((M_ex, M_ex, num_heads_kv, head_dim_v), jax_dtype)
        ex_bdout = _example_tensor((M_ex, M_ex, num_heads, head_dim_v), jax_dtype)
        ex_blse = _example_tensor((M_ex, num_heads, M_ex), jnp.float32)
        ex_bdpsum = _example_tensor((M_ex, num_heads, M_ex), jnp.float32)
        n_mblocks = math.ceil(M_ex / m_block_bwd)
        ex_dq_sem = _example_tensor_int32((M_ex * num_heads * n_mblocks,))
        ex_dq_acc = _example_tensor(
            (M_ex, num_heads, seqlen_rounded * head_dim_rounded), jnp.float32,
        )
        ex_dk = _example_tensor((M_ex, M_ex, num_heads_kv, head_dim), jax_dtype)
        ex_dv = _example_tensor((M_ex, M_ex, num_heads_kv, head_dim_v), jax_dtype)

        bwd_compiled = cute.compile(
            bwd_wrapper,
            ex_bq, ex_bk, ex_bv, ex_bdout, ex_blse, ex_bdpsum,
            ex_dq_sem, ex_dq_acc,
            ex_dk, ex_dv, ex_dq_acc,  # ex_dq_acc also appears as aliased output
            stream, options="--enable-tvm-ffi",
        )
    else:
        bwd_wrapper = _BwdVarlenWrapper(
            fa_bwd, softmax_scale, supports_semaphore=bwd_supports_semaphore,
        )
        ex_bq = _example_tensor((M_ex, num_heads, head_dim), jax_dtype)
        ex_bk = _example_tensor((M_ex, num_heads_kv, head_dim), jax_dtype)
        ex_bv = _example_tensor((M_ex, num_heads_kv, head_dim_v), jax_dtype)
        ex_bdout = _example_tensor((M_ex, num_heads, head_dim_v), jax_dtype)
        ex_cu_q_bwd = _example_tensor_int32((3,))
        ex_cu_k_bwd = _example_tensor_int32((3,))
        ex_blse = _example_tensor((num_heads, M_ex), jnp.float32)
        ex_bdpsum = _example_tensor((num_heads, M_ex), jnp.float32)
        n_mblocks = math.ceil(M_ex / m_block_bwd)
        ex_dq_sem = _example_tensor_int32((num_heads * n_mblocks,))
        ex_dq_acc = _example_tensor(
            (num_heads, seqlen_rounded * head_dim_rounded), jnp.float32,
        )
        ex_dk = _example_tensor((M_ex, num_heads_kv, head_dim), jax_dtype)
        ex_dv = _example_tensor((M_ex, num_heads_kv, head_dim_v), jax_dtype)

        bwd_compiled = cute.compile(
            bwd_wrapper,
            ex_bq, ex_bk, ex_bv, ex_bdout,
            ex_cu_q_bwd, ex_cu_k_bwd,
            ex_blse, ex_bdpsum,
            ex_dq_sem, ex_dq_acc,
            ex_dk, ex_dv, ex_dq_acc,  # ex_dq_acc also appears as aliased output
            stream, options="--enable-tvm-ffi",
        )
    bwd_name = f"cute.fa4_bwd_{cache_key}"
    jax_tvm_ffi.register_ffi_target(
        bwd_name, bwd_compiled,
        arg_spec=["args", "rets"],
        platform="gpu",
    )

    # ---- Backward postprocess: dQaccum (flat MMA layout) -> dQ ----
    # Must match the bwd main's num_threads/AtomLayoutMdQ/dQ_swapAB so the
    # flat dQaccum layout read here matches the one written by the bwd kernel.
    bwd_post = FlashAttentionBackwardPostprocess(
        cutlass_dtype,
        head_dim=head_dim,
        arch=arch,
        tile_m=m_block_bwd,
        num_threads=128,
        AtomLayoutMdQ=4,
        dQ_swapAB=False,
    )
    if not varlen:
        bwd_post_wrapper = _BwdPostWrapper(bwd_post, softmax_scale)
        ex_post_dq = _example_tensor(
            (M_ex, M_ex, num_heads, head_dim), jax_dtype,
        )
        bwd_post_compiled = cute.compile(
            bwd_post_wrapper,
            ex_dq_acc,     # input: dq_accum
            ex_post_dq,    # output: dq in q-layout
            stream, options="--enable-tvm-ffi",
        )
    else:
        bwd_post_wrapper = _BwdPostVarlenWrapper(bwd_post, softmax_scale)
        ex_post_cu = _example_tensor_int32((3,))
        ex_post_dq = _example_tensor((M_ex, num_heads, head_dim), jax_dtype)
        bwd_post_compiled = cute.compile(
            bwd_post_wrapper,
            ex_dq_acc, ex_post_cu,   # inputs
            ex_post_dq,              # output
            stream, options="--enable-tvm-ffi",
        )
    bwd_post_name = f"cute.fa4_bwd_post_{cache_key}"
    jax_tvm_ffi.register_ffi_target(
        bwd_post_name, bwd_post_compiled,
        arg_spec=["args", "rets"],
        platform="gpu",
    )

    # ---- Build custom_vjp wrapper ----
    _build_attn_fn(
        cache_key, fwd_name, bwd_pre_name, bwd_name, bwd_post_name,
        head_dim, head_dim_v, num_heads, num_heads_kv,
        m_block_bwd, jax_dtype, varlen,
    )


def _build_attn_fn(
    cache_key, fwd_name, bwd_pre_name, bwd_name, bwd_post_name,
    head_dim, head_dim_v, num_heads, num_heads_kv,
    m_block_bwd, jax_dtype, varlen,
):
    """Create the custom_vjp-wrapped attention function and cache it."""
    head_dim_rounded = math.ceil(head_dim / 32) * 32

    if not varlen:
        # ---- Non-varlen path ----

        @jax.custom_vjp
        def attn_fn(q, k, v):
            o, _lse = jax.ffi.ffi_call(
                fwd_name,
                (
                    jax.ShapeDtypeStruct(
                        q.shape[:-1] + (head_dim_v,), q.dtype,
                    ),
                    jax.ShapeDtypeStruct(
                        (q.shape[0], num_heads, q.shape[1]), jnp.float32,
                    ),
                ),
                vmap_method="broadcast_all",
            )(q, k, v)
            return o

        def attn_fwd(q, k, v):
            o, lse = jax.ffi.ffi_call(
                fwd_name,
                (
                    jax.ShapeDtypeStruct(
                        q.shape[:-1] + (head_dim_v,), q.dtype,
                    ),
                    jax.ShapeDtypeStruct(
                        (q.shape[0], num_heads, q.shape[1]), jnp.float32,
                    ),
                ),
                vmap_method="broadcast_all",
            )(q, k, v)
            return o, (q, k, v, o, lse)

        def attn_bwd(res, g):
            q, k, v, o, lse = res
            batch = q.shape[0]
            seqlen_q = q.shape[1]
            seqlen_q_rounded = (
                math.ceil(seqlen_q / m_block_bwd) * m_block_bwd
            )
            n_mblocks = math.ceil(seqlen_q / m_block_bwd)

            # Preprocess: dpsum, lse_log2, zeroed dq_accum. dpsum/lse_log2 use
            # the rounded seqlen (kernel pads to full tiles); dq_accum is flat
            # (seqlen_q_rounded * head_dim_rounded) as expected by the Sm80
            # bwd kernel.
            dpsum, lse_log2, dq_accum = jax.ffi.ffi_call(
                bwd_pre_name,
                (
                    jax.ShapeDtypeStruct(
                        (batch, num_heads, seqlen_q_rounded), jnp.float32,
                    ),
                    jax.ShapeDtypeStruct(
                        (batch, num_heads, seqlen_q_rounded), jnp.float32,
                    ),
                    jax.ShapeDtypeStruct(
                        (batch, num_heads, seqlen_q_rounded * head_dim_rounded),
                        jnp.float32,
                    ),
                ),
                vmap_method="broadcast_all",
            )(o, g, lse)

            # Semaphore for dQ atomic accumulation (ignored on SM120 path).
            dq_sem = jnp.zeros(
                (batch * num_heads * n_mblocks,), dtype=jnp.int32,
            )

            # Main backward. dq_accum is atomically accumulated in place by
            # the kernel; declare it as an output aliased to input 7 so JAX
            # observes the mutation.
            dk, dv, dq_accum = jax.ffi.ffi_call(
                bwd_name,
                (
                    jax.ShapeDtypeStruct(k.shape, k.dtype),
                    jax.ShapeDtypeStruct(v.shape, v.dtype),
                    jax.ShapeDtypeStruct(dq_accum.shape, dq_accum.dtype),
                ),
                input_output_aliases={7: 2},
                vmap_method="broadcast_all",
            )(q, k, v, g, lse_log2, dpsum, dq_sem, dq_accum)

            # Postprocess kernel reads dQaccum with the matching MMA tile
            # layout, applies softmax_scale, and writes dQ in q-layout cast
            # to q.dtype. Manual reshape/transpose can't recover this layout.
            (dq,) = jax.ffi.ffi_call(
                bwd_post_name,
                (jax.ShapeDtypeStruct(q.shape, q.dtype),),
                vmap_method="broadcast_all",
            )(dq_accum)
            return dq, dk, dv

        attn_fn.defvjp(attn_fwd, attn_bwd)

    else:
        # ---- Varlen path ----

        @jax.custom_vjp
        def attn_fn(q, k, v, cu_seqlens_q, cu_seqlens_k):
            o, _lse = jax.ffi.ffi_call(
                fwd_name,
                (
                    jax.ShapeDtypeStruct(
                        q.shape[:-1] + (head_dim_v,), q.dtype,
                    ),
                    jax.ShapeDtypeStruct(
                        (num_heads, q.shape[0]), jnp.float32,
                    ),
                ),
                vmap_method="broadcast_all",
            )(q, k, v, cu_seqlens_q, cu_seqlens_k)
            return o

        def attn_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k):
            o, lse = jax.ffi.ffi_call(
                fwd_name,
                (
                    jax.ShapeDtypeStruct(
                        q.shape[:-1] + (head_dim_v,), q.dtype,
                    ),
                    jax.ShapeDtypeStruct(
                        (num_heads, q.shape[0]), jnp.float32,
                    ),
                ),
                vmap_method="broadcast_all",
            )(q, k, v, cu_seqlens_q, cu_seqlens_k)
            return o, (q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k)

        def attn_bwd(res, g):
            q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k = res
            total_q = q.shape[0]
            seqlen_q_rounded = (
                math.ceil(total_q / m_block_bwd) * m_block_bwd
            )
            n_mblocks = math.ceil(total_q / m_block_bwd)

            # Preprocess
            dpsum, lse_log2, dq_accum = jax.ffi.ffi_call(
                bwd_pre_name,
                (
                    jax.ShapeDtypeStruct(
                        (num_heads, total_q), jnp.float32,
                    ),
                    jax.ShapeDtypeStruct(
                        (num_heads, total_q), jnp.float32,
                    ),
                    jax.ShapeDtypeStruct(
                        (num_heads, seqlen_q_rounded * head_dim_rounded),
                        jnp.float32,
                    ),
                ),
                vmap_method="broadcast_all",
            )(o, g, cu_seqlens_q, lse)

            dq_sem = jnp.zeros(
                (num_heads * n_mblocks,), dtype=jnp.int32,
            )

            dk, dv, dq_accum = jax.ffi.ffi_call(
                bwd_name,
                (
                    jax.ShapeDtypeStruct(k.shape, k.dtype),
                    jax.ShapeDtypeStruct(v.shape, v.dtype),
                    jax.ShapeDtypeStruct(dq_accum.shape, dq_accum.dtype),
                ),
                input_output_aliases={9: 2},
                vmap_method="broadcast_all",
            )(
                q, k, v, g,
                cu_seqlens_q, cu_seqlens_k,
                lse_log2, dpsum,
                dq_sem, dq_accum,
            )

            (dq,) = jax.ffi.ffi_call(
                bwd_post_name,
                (jax.ShapeDtypeStruct(q.shape, q.dtype),),
                vmap_method="broadcast_all",
            )(dq_accum, cu_seqlens_q)
            return dq, dk, dv, None, None  # None for cu_seqlens grads

        attn_fn.defvjp(attn_fwd, attn_bwd)

    _REGISTERED[cache_key] = (None, None, attn_fn)


# ---------------------------------------------------------------------------
# Mask → varlen packing utilities
# ---------------------------------------------------------------------------


def _mask_to_seqlens(mask: Array) -> tuple[Array, Array]:
    """Convert a padding mask to sequence lengths.

    Parameters
    ----------
    mask : Array
        Boolean mask, shape (..., num_heads, q_len, kv_len).
        True = attend, False = masked.

    Returns
    -------
    seqlens_q, seqlens_k : Array
        Per-batch sequence lengths, each shape (batch,).
    """
    # Any head that has True → token is valid.
    # Reduce over heads and the opposite seq dim.
    q_valid = jnp.any(mask, axis=(-3, -1))   # (..., q_len)
    k_valid = jnp.any(mask, axis=(-3, -2))   # (..., kv_len)
    seqlens_q = jnp.sum(q_valid.astype(jnp.int32), axis=-1)
    seqlens_k = jnp.sum(k_valid.astype(jnp.int32), axis=-1)
    return seqlens_q, seqlens_k


def _valid_mask(seqlens: Array, max_len: int) -> Array:
    """Boolean mask of shape ``(batch, max_len)``: True where token valid."""
    return jnp.arange(max_len)[None, :] < seqlens[:, None]


def _pack_order(valid: Array) -> Array:
    """Permutation that places valid tokens first in original order.

    Stable-sort on the negated validity flag sends True entries to the
    front while preserving per-batch order. Invalid tokens end up at the
    tail and are ignored by the kernel (cu_seqlens bounds the work).
    """
    return jnp.argsort(jnp.logical_not(valid).reshape(-1).astype(jnp.int32),
                       stable=True)


def _pack_varlen(
    query: Array, key: Array, value: Array,
    seqlens_q: Array, seqlens_k: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Compact padded tensors into contiguous varlen format.

    Parameters
    ----------
    query : Array, shape (batch, seqlen_q, heads, dim)
    key : Array, shape (batch, seqlen_k, heads_kv, dim)
    value : Array, shape (batch, seqlen_k, heads_kv, dim_v)
    seqlens_q, seqlens_k : Array, shape (batch,)

    Returns
    -------
    q_packed, k_packed, v_packed : tensors shaped (batch*seqlen, ...) with
        valid tokens compacted to the front and padding tokens at the tail.
    cu_seqlens_q, cu_seqlens_k : cumulative seq lengths, shape (batch+1,).
    order_q : permutation used for q packing (shape batch*seqlen_q).
        The inverse permutation is used by ``_unpack_varlen`` to scatter
        the varlen output back to padded layout.
    """
    cu_seqlens_q = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(seqlens_q)],
    )
    cu_seqlens_k = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(seqlens_k)],
    )

    batch, sq = query.shape[:2]
    sk = key.shape[1]

    valid_q = _valid_mask(seqlens_q, sq)
    valid_k = _valid_mask(seqlens_k, sk)
    order_q = _pack_order(valid_q)
    order_k = _pack_order(valid_k)

    q_packed = query.reshape(batch * sq, *query.shape[2:])[order_q]
    k_packed = key.reshape(batch * sk, *key.shape[2:])[order_k]
    v_packed = value.reshape(batch * sk, *value.shape[2:])[order_k]

    return (
        q_packed, k_packed, v_packed,
        cu_seqlens_q, cu_seqlens_k,
        order_q,
    )


def _unpack_varlen(
    out_packed: Array,
    batch: int, seqlen_q: int, num_heads: int, head_dim_v: int,
    order_q: Array, valid_q: Array,
) -> Array:
    """Scatter packed output back to padded shape; zero padding slots."""
    inv_order = jnp.argsort(order_q)
    out_flat = out_packed[inv_order]
    out = out_flat.reshape(batch, seqlen_q, num_heads, head_dim_v)
    return jnp.where(valid_q[:, :, None, None], out, jnp.zeros_like(out))


# ---------------------------------------------------------------------------
# Public API — matches Flax attention_fn signature
# ---------------------------------------------------------------------------


def flash_attention_v4(
    query: Array,
    key: Array,
    value: Array,
    mask: Array | None = None,
    dropout_rng: Array | None = None,
    dropout_rate: float = 0.0,
    broadcast_dropout: bool = True,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
) -> Array:
    """CuTeDSL Flash Attention v4 for Blackwell GPUs.

    Drop-in replacement for the Flax attention_fn interface.

    Parameters
    ----------
    query : Array
        Shape [..., q_length, num_heads, qk_depth_per_head]
    key : Array
        Shape [..., kv_length, num_heads, qk_depth_per_head]
    value : Array
        Shape [..., kv_length, num_heads, v_depth_per_head]
    mask : Array | None
        Padding mask. If provided, converted to varlen cu_seqlens.

    Returns
    -------
    Array
        Shape [..., q_length, num_heads, v_depth_per_head]
    """
    *batch_dims, q_len, num_heads, head_dim = query.shape
    head_dim_v = value.shape[-1]
    num_heads_kv = key.shape[-2]
    batch_size = math.prod(batch_dims) if batch_dims else 1

    qhead_per_kvhead = num_heads // num_heads_kv
    jax_dtype = jnp.dtype(query.dtype)
    arch = _get_arch()

    if mask is not None:
        # --- Varlen path ---
        cache_key = (jax_dtype.name, head_dim, head_dim_v, qhead_per_kvhead, arch, True)
        if cache_key not in _REGISTERED:
            register_flash_attn_ops(
                head_dim, num_heads, jax_dtype, head_dim_v, num_heads_kv,
                varlen=True,
            )
        _, _, attn_fn = _REGISTERED[cache_key]

        seqlens_q, seqlens_k = _mask_to_seqlens(mask)
        seqlens_q = seqlens_q.reshape(batch_size)
        seqlens_k = seqlens_k.reshape(batch_size)

        q_flat = query.reshape(batch_size, q_len, num_heads, head_dim)
        kv_len = key.shape[-3]
        k_flat = key.reshape(batch_size, kv_len, num_heads_kv, head_dim)
        v_flat = value.reshape(batch_size, kv_len, num_heads_kv, head_dim_v)

        q_packed, k_packed, v_packed, cu_q, cu_k, order_q = _pack_varlen(
            q_flat, k_flat, v_flat, seqlens_q, seqlens_k,
        )
        out_packed = attn_fn(q_packed, k_packed, v_packed, cu_q, cu_k)
        valid_q = _valid_mask(seqlens_q, q_len)
        out = _unpack_varlen(
            out_packed, batch_size, q_len, num_heads, head_dim_v,
            order_q, valid_q,
        )
    else:
        # --- Non-varlen path ---
        cache_key = (jax_dtype.name, head_dim, head_dim_v, qhead_per_kvhead, arch, False)
        if cache_key not in _REGISTERED:
            register_flash_attn_ops(
                head_dim, num_heads, jax_dtype, head_dim_v, num_heads_kv,
                varlen=False,
            )
        _, _, attn_fn = _REGISTERED[cache_key]

        q_flat = query.reshape(batch_size, q_len, num_heads, head_dim)
        kv_len = key.shape[-3]
        k_flat = key.reshape(batch_size, kv_len, num_heads_kv, head_dim)
        v_flat = value.reshape(batch_size, kv_len, num_heads_kv, head_dim_v)

        out = attn_fn(q_flat, k_flat, v_flat)

    result = out.reshape(*batch_dims, q_len, num_heads, head_dim_v)
    if dtype is not None:
        result = result.astype(dtype)
    return result

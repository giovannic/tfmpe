# Implementation Plan: Tokenized Flow Matching Estimator (TFMPE)

## Task Overview

Implementation follows a three-phase approach:
1. **ODE Infrastructure** (Tasks 1-3): Build and test ODE solving helpers with doubling flow test
2. **TFMPE Core & Integration** (Tasks 4-6): Implement TFMPE class with sampling/log_prob, integration tests
3. **Training & Benchmarks** (Tasks 7-10): Implement fit_fast, fit_memory_efficient, E2E test, benchmarks

## Steering Document Compliance

All tasks follow structure.md conventions:
- Source in `tfmpe/estimators/` with utilities in `ode.py`, `tfmpe.py`, `training.py`
- Tests mirror structure in `test/estimators/`
- Type annotations via jaxtyping (Array, PRNGKeyArray, Scalar)
- Line length ≤ 80 characters, helper functions for complex logic
- Benchmarks in `test/benchmark/`

## Atomic Task Requirements

**Each task meets these criteria:**
- **File Scope**: Touches 1-3 related files maximum
- **Time Boxing**: Completable in 15-30 minutes by experienced developer
- **Single Purpose**: One testable outcome per task
- **Specific Files**: Exact file paths specified
- **Agent-Friendly**: Clear input/output with minimal context switching

---

## Tasks

### Phase 1: ODE Infrastructure

- [x] 1. Create ODE solver module structure and Diffrax solver wrapper
  - Files: `tfmpe/estimators/ode.py`, `tfmpe/estimators/__init__.py`
  - Implement `solve_forward_ode()` using Diffrax Dopri5 solver
    - Parameters: vf_fn (callable), x0 (Array), solver, time_span, rtol, atol
    - Returns: final state Array shape (*x0.shape)
    - Use diffrax.diffeqsolve with vector_field argument
  - Implement helper `_prepare_diffrax_vf()` to wrap vector field for Diffrax API
    - Handles time parameter ordering (Diffrax expects t as first arg)
  - Add exports to `tfmpe/estimators/__init__.py`
  - Purpose: Foundation for ODE-based sampling
  - _Leverage: sfmpe_legacy/sfmpe/structured_cnf.py for algorithm structure (adapted for Diffrax)_
  - _Requirements: 1.1_

- [x] 2. Implement backward ODE and augmented ODE solvers
  - Files: `tfmpe/estimators/ode.py`
  - Implement `solve_backward_ode()` for log probability computation
    - Reverse time direction via negated vf_fn
    - Parameters: vf_fn (callable), x1 (Array), solver, time_span, rtol, atol
    - Returns: final state Array
  - Implement `solve_augmented_ode()` with trace-based log determinant
    - Augmented state: [x, log_det_jacobian]
    - Trace computation via vjp method (vector-Jacobian products)
    - Parameters: vf_fn (callable), x0 (Array), solver, time_span, rtol, atol
    - Returns: Tuple[Array, Scalar] = (final_x, final_log_det_jacobian)
  - Implement helper `_augmented_ode_fn()` that handles augmented state dynamics
  - Purpose: Log probability computation via FFJORD algorithm
  - _Leverage: FFJORD trace estimation from sfmpe_legacy/sfmpe/structured_cnf.py_
  - _Requirements: 1.1, 1.2_

- [x] 3. Implement batched ODE solver with vmap
  - Files: `tfmpe/estimators/ode.py`
  - Implement `batch_solve_forward_ode()` for efficient batch sampling
    - Signature: vf_fn, x0_batch (Array shape (batch, *state_shape)), solver, time_span, rtol, atol
    - Returns: Array shape (batch, *state_shape)
    - Use jax.vmap over first axis to parallelize across samples
    - Vectorized vf_fn application per sample
  - Implement `batch_solve_augmented_ode()` for batched log prob
    - Signature: vf_fn, x0_batch, solver, time_span, rtol, atol
    - Returns: Tuple[Array shape (batch, *state_shape), Array shape (batch,)]
  - Purpose: Enable efficient batch operations for sampling/log prob
  - _Requirements: 1.1, 1.2_

- [x] 4. Create doubling continuous flow unit test
  - Files: `test/estimators/test_ode.py`
  - Test `solve_forward_ode()` with doubling flow vector field
    - Vector field: f(θ) = log(2) · θ (exponential scaling)
    - Initial: θ₀ ~ N(0, 1)
    - Final: θ₁ should follow N(0, 4) analytically
    - Verify mean ≈ 0, std ≈ 2 with rtol=0.1
    - Sample 1000 trajectories, check statistics
  - Test `solve_backward_ode()` reverses forward trajectory
    - Verify backward(forward(x)) ≈ x within tolerance
  - Purpose: Verify correctness of forward/backward ODE solving
  - _Leverage: test patterns from sfmpe_legacy/test/test_cnf_density.py_
  - _Requirements: 1.1, 1.2, 4.1, 4.2, 4.3, 4.4_

- [x] 5. Create unit tests for augmented ODE and batch operations
  - Files: `test/estimators/test_ode.py` (continue from 4)
  - Test `solve_augmented_ode()` trace computation
    - Compare trace estimate vs analytical jacobian for simple linear vf
    - Linear vf: f(x) = A·x (matrix A) should have log_det ≈ log|det(A)|
    - Test with 2x2 and 5x5 matrices
  - Test batch operations: `batch_solve_forward_ode()`, `batch_solve_augmented_ode()`
    - Verify vmap gives same results as loop over samples
    - Verify batch shapes correct: (batch_size, *state_shape)
  - Purpose: Ensure batch operations and trace computation work correctly
  - _Requirements: 1.1, 1.2, 1.3, 4.1, 4.3_

---

### Phase 2: TFMPE Core & Integration

- [x] 6. Create TFMPE core class with initialization
  - Files: `tfmpe/estimators/tfmpe.py`
  - Implement `TFMPE` class with initialization
    - Parameters:
      - `vf_network`: Callable or nnx.Module for vector field (Token → Array → Array)
      - `base_dist_fn`: Callable(rng, shape) → Array for base distribution sampling
      - `base_log_prob_fn`: Callable(x) → scalar for log probability of base
      - `solver`: Diffrax solver instance (default: Heun())
      - `ode_kwargs`: dict with rtol, atol defaults (default: rtol=1e-5, atol=1e-5)
    - Store as instance attributes
  - Implement type annotations using jaxtyping (Array, PRNGKeyArray, Scalar)
  - Add docstrings with numpy-style format
  - Purpose: Foundation for TFMPE interface
  - _Leverage: SFMPE class structure from sfmpe_legacy/sfmpe/sfmpe.py_
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 7. Implement TFMPE sampling method
  - Files: `tfmpe/estimators/tfmpe.py`
  - Implement `sample_posterior()` method
    - Signature: context (Token), params (Token), rng (PRNGKeyArray) → Token
    - Fills params.data with base_dist_fn samples
    - Calls solve_forward_ode() with vector field and params
    - Returns sampled Token with structure metadata preserved
  - Purpose: Generate posterior samples via ODE solving
  - _Leverage: sample() method from sfmpe_legacy/sfmpe/structured_cnf.py_
  - _Requirements: 2.1, 3.1, 3.2_

- [x] 8. Implement TFMPE log probability method
  - Files: `tfmpe/estimators/tfmpe.py`
  - Implement `log_prob_posterior_samples()` method
    - Signature: theta (Token), context (Token) → Array (scalar)
    - Computes base log prob via base_log_prob_fn(theta.data)
    - Calls solve_augmented_ode() to get trace-based log determinant
    - Returns log_prob_base + log_det_jacobian
  - Optional `n_epsilon` parameter for FFJORD trace samples (default: 10)
  - Purpose: Compute log probabilities for FFJORD algorithm
  - _Leverage: log_prob() method from sfmpe_legacy/sfmpe/structured_cnf.py_
  - _Requirements: 2.1, 2.2, 2.3, 5.2, 5.3_

- [x] 9. Create TFMPE integration tests
  - Files: `test/estimators/test_tfmpe.py`
  - Test `sample_posterior()` with synthetic data
    - Create simple context Token
    - Call sample_posterior() for 100 samples
    - Verify output Token shape and structure
    - Verify samples are finite, no NaNs/Infs
  - Test `log_prob_posterior_samples()`
    - Compute log_prob for known distribution
    - Compare to analytical log_prob within stochastic tolerance (rtol=0.1, atol=0.5)
    - Test with simple Gaussian posterior
  - Test Token format preservation
    - Verify sampled Token has same labels, masks as context
    - Verify slices metadata preserved
  - Test consistency: log_prob(samples from posterior)
    - Verify log probabilities are finite and reasonable
  - Purpose: Verify TFMPE sampling and log_prob work correctly with Token data
  - _Leverage: Integration test patterns from sfmpe_legacy/test/_
  - _Requirements: 3.1, 3.2, 5.1, 5.2, 5.3, 5.4_

---

### Phase 3: Training & Benchmarks

- [ ] 10. Implement CFM loss function
  - Files: `tfmpe/estimators/training.py`
  - Implement `cfm_loss()` function
    - Signature: tfmpe (TFMPE), theta (Array), context (Array), time (Scalar), rng (PRNGKeyArray) → Scalar
    - Parameters:
      - theta: flattened parameters Array (batch, d_flat)
      - context: flattened observations Array (batch, d_obs)
      - time: uniform sample from [0, 1] Shape ()
      - rng: PRNG key for noise sampling
    - Algorithm:
      1. Compute σ_t = 1 - (1 - σ_min) × t where σ_min = 1e-4
      2. Sample ε ~ N(0, I) with same shape as theta
      3. Interpolate: θ_t = θ × σ_t + ε × sqrt(1 - σ_t²)
      4. Compute target velocity: u_t = (θ - (1 - σ_min) × θ_t) / (1 - σ_t)
      5. Predict velocity: v = vf_network(θ_t, context, t)
      6. Compute MSE: mean((v - u_t)²)
    - Handle masking: apply padding masks if provided
    - Return scalar loss
  - Purpose: Core training loss for flow matching
  - _Leverage: _cfm_loss from sfmpe_legacy/sfmpe/sfmpe.py_
  - _Requirements: 6.5_

- [x] 11. Implement fit_fast() speed-optimized training loop
  - Files: `tfmpe/estimators/training.py`
  - Implement `fit_fast()` function
    - Signature: tfmpe (TFMPE), train_tokens (Token), val_tokens (Token), opt (nnx.Optimizer), n_iter (int), rng (PRNGKeyArray) → Tuple[TFMPE, Array]
    - Implementation:
      1. Extract flattened arrays from Tokens
      2. Reshape to (n_batches, batch_size, d_flat)
      3. Create loss_fn that vmaps cfm_loss over batch samples
      4. Implement gradient_step() that applies optimizer.update()
      5. Use nnx.scan to scan over iterations
      6. Compute weighted loss: loss × (batch_size / total_size)
      7. Compute validation loss on full val set
      8. Return (trained_tfmpe, losses) shape (n_iter, 2)
  - **Critical**: Implementation must be fully jittable with jax.jit()
    - No Python control flow in scanned loop
    - All randomness via PRNG keys (key_split per iteration)
    - Pre-compute batch indices
    - Pass nnx.Optimizer as argument to avoid unhashable optax callables in JIT
    - Mark n_iter as static_argnames when JITing
  - Purpose: Fast training via JIT compilation
  - _Leverage: fit_model_no_branch from sfmpe_legacy/sfmpe/train.py_
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 12. Implement fit_memory_efficient() training loop
  - Files: `tfmpe/estimators/training.py`
  - Implement `fit_memory_efficient()` function
    - Signature: tfmpe (TFMPE), token_gen (TokenGenerator), val_tokens (Token), optimizer (optax.GradientTransformation), n_iter (int), rng (PRNGKeyArray), early_stopping_patience (int) → Tuple[TFMPE, Array]
    - Implementation:
      1. Python loop over iterations
      2. For each iteration, get batch from token_gen
      3. Extract flattened arrays, vmap cfm_loss over batch
      4. Compute gradient, apply optimizer.update()
      5. Accumulate weighted loss
      6. Every K iterations: compute validation loss
      7. Early stopping: stop if val loss doesn't improve for patience iterations
      8. Return (trained_tfmpe, losses) - losses shape (actual_iters, 2)
    - Support variable batch sizes from generator
    - Collect losses in list, convert to array at end
  - Purpose: Memory-efficient training on large datasets
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 13. Create end-to-end training test
  - Files: `test/estimators/test_e2e_training.py`
  - Generate hierarchical Gaussian linear model data
    - Parameters: σ ~ HalfNormal(1), μᵢ ~ Normal(0, 1), yᵢⱼ ~ Normal(μᵢ, σ)
    - Generate N_groups=10 groups, N_obs=20 observations per group
    - Flatten to Token format with labels for structure
  - Create TFMPE with simple transformer vector field
  - Test fit_fast() training
    - Train for 50 iterations with batch_size=5
    - Verify training loss decreases
    - Verify final model produces valid samples and log_probs
    - Check execution time < 60 seconds
  - Test fit_memory_efficient() training
    - Create TokenGenerator for streaming batches
    - Train for 50 iterations with batch_size=5
    - Verify training loss decreases
    - Verify early stopping works (patience=5)
  - Purpose: Verify complete training pipeline on realistic problem
  - _Leverage: E2E test patterns from sfmpe_legacy/test/_
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 14. Create performance benchmarks
  - Files: `test/benchmark/test_estimator_benchmark.py`
  - Implement speed benchmark (standard dataset)
    - Hierarchical Gaussian data: N_groups=50, N_obs=10
    - Measure fit_fast() training iteration time (20 iterations)
    - Benchmark name: "tfmpe_fit_fast_speed"
    - Mark with @pytest.mark.benchmark(group="speed")
  - Implement scale benchmark (large dataset)
    - Hierarchical Gaussian data: N_groups=500, N_obs=20
    - Measure fit_fast() training iteration time (10 iterations)
    - Benchmark name: "tfmpe_fit_fast_scale"
    - Mark with @pytest.mark.benchmark(group="scale")
  - Implement sampling benchmark
    - Measure batch_solve_forward_ode() for 100, 1000, 10000 samples
    - Benchmark different batch sizes
  - Implement log_prob benchmark
    - Measure batch_solve_augmented_ode() for 100, 1000, 10000 samples
  - Purpose: Track performance regressions and verify scalability
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

---

## Implementation Notes

### Jittability for fit_fast()
- Use nnx.scan for iteration loop (enables JAX JIT)
- Pre-compute epoch/batch indices arrays
- Split PRNG keys deterministically: `jax.random.fold_in(key, iter)`
- Reshape training data outside jit (or pre-reshape as constant)
- Return scalar loss values only

### Token Integration
- Token: container with `data`, `labels`, `masks`, `slices`, `independence`
- Use `Token.data` for flattened arrays in ODE solvers
- Use `Token.slices` to unflatten samples back to structured format
- Preserve Token metadata (labels, masks) in sampled output

### Diffrax API
- Solver: `diffrax.Heun()` (default) or configurable
- `diffrax.diffeqsolve(vf, solver, t0, t1, dt0, y0, args=...)`
- TimeKeeper: `diffrax.SaveAt(ts=times)` for trajectory saving
- For final state only: use `t0, t1` without SaveAt

### Testing Patterns
- Use pytest for all tests
- Mark slow tests with `@pytest.mark.slow`
- Use `numpy.testing.assert_allclose()` for numerical comparisons
- Use custom PRNG keys per test: `jax.random.PRNGKey(seed)`

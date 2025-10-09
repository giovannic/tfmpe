# Implementation Plan

## Task Overview

This implementation follows a test-driven development approach, creating tests before implementations for each component. The transformer is built bottom-up: configuration � embedding � encoder/decoder blocks � full transformer � benchmarks. Each task is atomic, touching 1-3 related files and completable in 15-30 minutes.

## Steering Document Compliance

- **structure.md**: Code in `tfmpe/nn/transformer/`, tests in `test/test_nn/test_transformer/`, benchmarks in `test/benchmark/`
- **tech.md**: Uses JAX/Flax, `jaxtyping.Array`, pytest, pytest-benchmark, pytest-memray, numpy docstrings
- **Import order**: Built-in, external (jax, flax), internal (tfmpe), types, constants

## Atomic Task Requirements

**Each task meets:**
- **File Scope**: 1-3 related files maximum
- **Time Boxing**: 15-30 minutes per task
- **Single Purpose**: One testable outcome
- **Specific Files**: Exact file paths specified
- **Agent-Friendly**: Clear inputs/outputs, minimal context switching

## Tasks

### 1. Configuration Module

- [ ] 1.1 Create test for TransformerConfig dataclass in test/test_nn/test_transformer/test_config.py
  - File: test/test_nn/test_transformer/test_config.py (create)
  - Test default values instantiation
  - Test custom configuration values
  - Test that latent_dim divisible by n_heads (validation in __post_init__)
  - Test ValueError when latent_dim not divisible by n_heads
  - Purpose: Verify configuration dataclass behavior before implementation
  - _Requirements: 4.1-4.10_

- [ ] 1.2 Implement TransformerConfig dataclass in tfmpe/nn/transformer/config.py
  - File: tfmpe/nn/transformer/config.py (create)
  - Create @dataclass with fields: latent_dim, n_encoder, n_decoder, n_heads, n_ff, label_dim, index_out_dim, dropout, activation
  - Add default values matching design.md
  - Add __post_init__ to validate latent_dim % n_heads == 0
  - Include numpy-style docstring
  - Purpose: Configuration container for transformer architecture
  - _Leverage: tfmpe/preprocessing/tokens.py (dataclass pattern)_
  - _Requirements: 4.1-4.10_

### 2. Embedding Module

- [ ] 2.1 Create test for GaussianFourierEmbedding in test/test_nn/test_transformer/test_embedding.py
  - File: test/test_nn/test_transformer/test_embedding.py (create)
  - Test output shape: input (..., in_dim) � output (..., out_dim)
  - Test with different input shapes using @pytest.mark.parametrize
  - Test that output contains both sin and cos components
  - Purpose: Verify Gaussian Fourier embedding shape and behavior
  - _Requirements: 3.1_

- [ ] 2.2 Implement GaussianFourierEmbedding in tfmpe/nn/transformer/embedding.py
  - File: tfmpe/nn/transformer/embedding.py (create)
  - Implement nnx.Module with random Gaussian matrix for frequency basis
  - __call__ computes sin/cos of 2� * input @ basis
  - Concatenate cos and sin components
  - Include numpy-style docstring with shape specifications
  - Purpose: Fourier feature encoding for continuous indices
  - _Leverage: sfmpe_legacy/sfmpe/nn/transformer/embedding.py (reference)_
  - _Requirements: 3.1_

- [ ] 2.3 Add tests for Embedding layer to test_embedding.py
  - File: test/test_nn/test_transformer/test_embedding.py (modify)
  - Test output shape without indices: input (batch, n_tokens, value_dim) � (batch, n_tokens, latent_dim)
  - Test output shape with indices
  - Test output shape with functional_inputs
  - Test time broadcasting from scalar to (batch, 1)
  - Test with sample dimensions: (samples, batch, n_tokens, value_dim)
  - Purpose: Verify embedding layer handles all input variations
  - _Requirements: 3.1, 1.4_

- [ ] 2.4 Implement Embedding layer in embedding.py
  - File: tfmpe/nn/transformer/embedding.py (modify)
  - __init__ creates nnx.Embed for labels, GaussianFourierEmbedding for indices, nnx.Linear for projection
  - __call__ concatenates: values + label_embeds + (optional index_embeds) + (optional functional_inputs) + time
  - Apply linear projection to latent_dim
  - Broadcast time and labels to match batch dimensions
  - Include numpy-style docstring
  - Purpose: Embed token data into latent space
  - _Leverage: sfmpe_legacy/sfmpe/nn/transformer/embedding.py (reference)_
  - _Requirements: 3.1, 1.4_

### 3. Encoder/Decoder Blocks

- [x] 3.1 Create test for MLP in test/test_nn/test_transformer/test_encoder.py
  - File: test/test_nn/test_transformer/test_encoder.py (create)
  - Test output shape preservation: (batch, n_tokens, latent_dim) � (batch, n_tokens, latent_dim)
  - Test with different n_ff values using @pytest.mark.parametrize
  - Test dropout in train vs eval mode
  - Purpose: Verify MLP feedforward network before encoder/decoder
  - _Requirements: 3.2_

- [x] 3.2 Implement MLP module in tfmpe/nn/transformer/encoder.py
  - File: tfmpe/nn/transformer/encoder.py (create)
  - Create FFLayer with nnx.Linear, nnx.Dropout, activation
  - Use nnx.vmap and nnx.scan to create n_ff layers
  - __call__ applies layers sequentially using nnx.scan
  - Include numpy-style docstring
  - Purpose: Multi-layer feedforward with dropout
  - _Leverage: sfmpe_legacy/sfmpe/nn/transformer/encoder.py (reference)_
  - _Requirements: 3.2_

- [x] 3.3 Add test for EncoderBlock to test_encoder.py
  - File: test/test_nn/test_transformer/test_encoder.py (modify)
  - Test output shape preserved
  - Test self-attention with mask (verify mask zeros out connections)
  - Purpose: Verify encoder block before full transformer
  - _Requirements: 2.2, 3.2_

- [x] 3.4 Implement EncoderBlock in encoder.py
  - File: tfmpe/nn/transformer/encoder.py (modify)
  - __init__ creates nnx.MultiHeadAttention, nnx.LayerNorm, MLP
  - __call__ applies: self_attention + residual + norm + MLP + residual + norm
  - Use config for n_heads, latent_dim, dropout
  - Include numpy-style docstring
  - Purpose: Self-attention transformer block
  - _Leverage: sfmpe_legacy/sfmpe/nn/transformer/encoder.py (reference)_
  - _Requirements: 2.2, 3.2_

- [x] 3.5 Add test for DecoderBlock to test_encoder.py
  - File: test/test_nn/test_transformer/test_encoder.py (modify)
  - Test output shape: query (batch, n_q, latent_dim) → (batch, n_q, latent_dim)
  - Test cross-attention with mask
  - Purpose: Verify decoder block cross-attention
  - _Requirements: 2.4, 3.2_

- [x] 3.6 Implement DecoderBlock in encoder.py
  - File: tfmpe/nn/transformer/encoder.py (modify)
  - __init__ creates nnx.MultiHeadAttention, nnx.LayerNorm, MLP
  - __call__ applies: cross_attention(query=x, key=context, value=context) + residual + norm + MLP + residual + norm
  - Include numpy-style docstring
  - Purpose: Cross-attention transformer block
  - _Leverage: sfmpe_legacy/sfmpe/nn/transformer/encoder.py (reference)_
  - _Requirements: 2.4, 3.2_

### 4. Main Transformer

- [x] 4.1 Create test for Transformer in test/test_nn/test_transformer/test_transformer.py
  - File: test/test_nn/test_transformer/test_transformer.py (create)
  - Create fixture for sample Tokens objects (context and parameters)
  - Test __init__ accepts config dataclass
  - Test forward pass output shape: (batch, n_param_tokens, value_dim)
  - Test with sample dimensions: (samples, batch, ...)
  - Test encode method output shape
  - Test decode method output shape
  - Use @pytest.mark.parametrize for varying n_tokens, latent_dim
  - Purpose: Verify transformer before implementation
  - _Leverage: test/test_preprocessing/test_tokens_basic.py (Tokens fixtures)_
  - _Requirements: 1.1-1.5, 2.1-2.5, 3.4_

- [x] 4.2 Implement Transformer.__init__ in tfmpe/nn/transformer/transformer.py
  - File: tfmpe/nn/transformer/transformer.py (create)
  - __init__ creates: Embedding, n_encoder EncoderBlocks via nnx.vmap, n_decoder DecoderBlocks via nnx.vmap, output Linear
  - Accept config: TransformerConfig, value_dim, n_labels, index_dim, rngs
  - Include numpy-style class and __init__ docstrings
  - Purpose: Initialize transformer components
  - _Leverage: sfmpe_legacy/sfmpe/nn/transformer/transformer.py (reference)_
  - _Requirements: 1.1-1.5, 4.11-4.12_

- [x] 4.3 Implement Transformer.encode method in transformer.py
  - File: tfmpe/nn/transformer/transformer.py (modify)
  - Extract data, labels from Tokens object
  - Embed tokens with time
  - Apply encoder blocks sequentially using nnx.scan with self_attention_mask
  - Return encoded array
  - Include numpy-style docstring with shapes
  - Purpose: Encode context or parameter tokens
  - _Requirements: 2.1-2.2_

- [x] 4.4 Implement Transformer.decode method in transformer.py
  - File: tfmpe/nn/transformer/transformer.py (modify)
  - Embed parameter tokens with time
  - Encode parameters with encoder blocks
  - Decode with decoder blocks using cross-attention to encoded_context
  - Apply output linear layer
  - Return vector field array
  - Include numpy-style docstring with shapes
  - Purpose: Decode parameters conditioned on context
  - _Requirements: 2.3-2.5_

- [x] 4.5 Implement Transformer.__call__ method in transformer.py
  - File: tfmpe/nn/transformer/transformer.py (modify)
  - Validate context_tokens.sample_shape == param_tokens.sample_shape
  - Encode context tokens
  - Decode parameter tokens with encoded context
  - Return vector field output
  - Include numpy-style docstring
  - Purpose: End-to-end forward pass through transformer
  - _Requirements: 1.1-1.5, 2.5_

- [x] 4.6 Update tfmpe/nn/transformer/__init__.py to export classes
  - File: tfmpe/nn/transformer/__init__.py (modify)
  - Import and expose: TransformerConfig, Transformer
  - Update __all__ list
  - Purpose: Make transformer accessible from package
  - _Requirements: Usability_

### 5. Integration Testing

- [ ] 5.1 Create integration test in test/test_nn/test_transformer/test_integration.py
  - File: test/test_nn/test_transformer/test_integration.py (create)
  - Create PyTree � Tokens � split to context/param views
  - Initialize Transformer with config
  - Forward pass through transformer
  - Verify output shape matches param_tokens.data shape
  - Compute gradient w.r.t. transformer parameters using jax.grad
  - Verify gradient dict has expected keys
  - Purpose: Test complete TFMPE pipeline integration
  - _Leverage: test/test_preprocessing/test_tokens_basic.py (Tokens creation)_
  - _Requirements: 1.1-1.5, Integration Testing_

### 6. Speed Benchmarks

- [ ] 6.1 Create forward pass speed benchmark in test/benchmark/test_transformer_speed.py
  - File: test/benchmark/test_transformer_speed.py (create)
  - Use pytest-benchmark fixture
  - Create benchmark function accepting n_tokens, config parameters
  - Compile transformer forward pass with jax.jit
  - Benchmark compiled function (exclude compilation time)
  - Use @pytest.mark.speed for standard configs (n_tokens=100, latent_dim=128)
  - Assert execution time < threshold (e.g., 0.1s)
  - Purpose: Measure and enforce forward pass speed
  - _Requirements: 5.1-5.11_

- [ ] 6.2 Add backward pass speed benchmark to test_transformer_speed.py
  - File: test/benchmark/test_transformer_speed.py (modify)
  - Create jax.grad function for transformer
  - Compile gradient computation with jax.jit
  - Benchmark compiled gradient function
  - Use @pytest.mark.speed marker
  - Assert gradient computation time < threshold (e.g., 0.2s)
  - Purpose: Measure and enforce backward pass speed
  - _Requirements: 5.1-5.11_

- [ ] 6.3 Add large-scale speed benchmarks to test_transformer_speed.py
  - File: test/benchmark/test_transformer_speed.py (modify)
  - Create benchmarks with large configs (n_tokens=1000, latent_dim=512, n_encoder=8)
  - Use @pytest.mark.scale marker
  - Test both forward and backward passes
  - Higher time thresholds for large configs
  - Purpose: Ensure transformer scales to large problems
  - _Requirements: 5.10-5.11_

### 7. Memory Benchmarks

- [ ] 7.1 Create forward pass memory benchmark in test/benchmark/test_transformer_memory.py
  - File: test/benchmark/test_transformer_memory.py (create)
  - Use @pytest.mark.limit_memory decorator from pytest-memray
  - Create function executing transformer forward pass
  - Set memory limit (e.g., 100 MB for standard config)
  - Use @pytest.mark.speed for standard configs
  - Purpose: Measure and enforce forward pass memory usage
  - _Requirements: 6.1-6.10_

- [ ] 7.2 Add backward pass memory benchmark to test_transformer_memory.py
  - File: test/benchmark/test_transformer_memory.py (modify)
  - Use pytest-memray to profile gradient computation
  - Set memory limit for backward pass (typically higher than forward)
  - Use @pytest.mark.speed marker
  - Purpose: Measure and enforce backward pass memory
  - _Requirements: 6.1-6.10_

- [ ] 7.3 Add large-scale memory benchmarks to test_transformer_memory.py
  - File: test/benchmark/test_transformer_memory.py (modify)
  - Create benchmarks with large configs (n_tokens=1000, latent_dim=512)
  - Use @pytest.mark.scale marker
  - Higher memory limits for large configs (e.g., 500 MB)
  - Purpose: Ensure memory efficiency on large problems
  - _Requirements: 6.9-6.10_

### 8. Final Verification

- [ ] 8.1 Run pyright type checking on transformer module
  - Command: `pyright tfmpe/nn/transformer/`
  - Verify no type errors
  - Fix any type annotation issues
  - Purpose: Ensure type safety and maintainability
  - _Requirements: Maintainability (pyright)_

- [ ] 8.2 Run all transformer tests
  - Command: `python -m pytest test/test_nn/test_transformer/ -v`
  - Verify all unit and integration tests pass
  - Fix any failures
  - Purpose: Ensure all components work correctly
  - _Requirements: All_

- [ ] 8.3 Run speed benchmarks (excluding scale)
  - Command: `python -m pytest test/benchmark/ -m "speed" -v`
  - Verify benchmarks pass time thresholds
  - Purpose: Ensure performance meets requirements
  - _Requirements: 5.10, 6.9_

- [ ] 8.4 Run memory benchmarks (excluding scale)
  - Command: `python -m pytest test/benchmark/ -m "speed" -v --memray`
  - Verify benchmarks pass memory thresholds
  - Purpose: Ensure memory efficiency
  - _Requirements: 6.9_

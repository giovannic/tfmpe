# Requirements Document

## Introduction

The Transformer feature implements a neural network architecture for Tokenised Flow Matching for Posterior Estimation (TFMPE). This transformer processes structured parameter data using an encoder-decoder architecture to generate vector fields for flow matching in simulation-based inference.

Unlike the legacy implementation which accepts separate arrays for context and parameters, this implementation leverages the new `Tokens` unified data interface to handle structured parameter data, masking, and functional inputs coherently.

## Alignment with Package Vision

This feature directly supports the package's core goals:

- **TFMPE Implementation**: Provides the neural network backbone for the tokenised estimator
- **Data-preprocessing Integration**: Seamlessly works with the `Tokens` interface, eliminating manual parameter flattening and mask construction
- **Scalability**: Addresses memory limitations through benchmarking and optimization, critical for large-scale hierarchical models
- **Speed**: Includes execution time benchmarks to ensure fast training and inference

## Requirements

### Requirement 1: Tokens-based Interface

**User Story:** As a scientist using TFMPE, I want the transformer to accept `Tokens` objects directly, so that I don't need to manually manage parameter flattening, labels, masks, and functional inputs.

#### Acceptance Criteria

1. WHEN the transformer is called THEN it SHALL accept a `Tokens` object for context data
2. WHEN the transformer is called THEN it SHALL accept a `Tokens` object for parameter data
3. WHEN processing Tokens objects THEN the transformer SHALL extract data, labels, masks, and functional inputs from the unified interface
4. WHEN Tokens objects contain functional inputs THEN the transformer SHALL incorporate them into token embeddings
5. WHEN Tokens objects specify independence structure THEN the transformer SHALL use the self-attention and cross-attention masks derived from that structure

### Requirement 2: Encoder-Decoder Architecture

**User Story:** As a scientist training a TFMPE model, I want an encoder-decoder transformer architecture that processes context separately from parameters, so that the model can effectively learn conditional distributions.

#### Acceptance Criteria

1. WHEN context tokens are provided THEN the encoder SHALL process them through multiple encoder blocks
2. WHEN encoder blocks process tokens THEN they SHALL apply self-attention using the context self-attention mask
3. WHEN parameter tokens are provided THEN the decoder SHALL first encode them through encoder blocks using parameter self-attention mask
4. WHEN decoder processes tokens THEN it SHALL apply cross-attention between encoded parameters and encoded context using cross-attention mask
5. WHEN the decoder completes THEN it SHALL output a vector field with shape matching the input parameter tokens

### Requirement 3: Modular Components with Shape Testing

**User Story:** As a developer maintaining the transformer, I want each component (embedding, encoder block, decoder block) tested for correct output shapes, so that I can catch shape mismatches early.

#### Acceptance Criteria

1. WHEN the embedding layer is called THEN it SHALL output shape `(*sample_shape, n_tokens, latent_dim)`
2. WHEN an encoder block processes tokens THEN it SHALL preserve the input shape
3. WHEN a decoder block processes tokens THEN it SHALL preserve the query token shape
4. WHEN the final linear layer is called THEN it SHALL output shape `(*sample_shape, n_theta_tokens, value_dim)`
5. WHEN any component is tested THEN the test SHALL verify the output shape matches the expected shape

### Requirement 4: Dataclass-based Configuration

**User Story:** As a scientist experimenting with model architectures, I want to configure the transformer using a dataclass with sensible defaults, so that I don't need to pass numerous parameters individually.

#### Acceptance Criteria

1. WHEN creating a configuration THEN it SHALL be a Python dataclass
2. WHEN initializing the configuration THEN it SHALL include `n_encoder` (number of encoder layers)
3. WHEN initializing the configuration THEN it SHALL include `n_decoder` (number of decoder layers)
4. WHEN initializing the configuration THEN it SHALL include `n_heads` (number of attention heads)
5. WHEN initializing the configuration THEN it SHALL include `n_ff` (number of feedforward layers)
6. WHEN initializing the configuration THEN it SHALL include `latent_dim` (hidden dimension size)
7. WHEN initializing the configuration THEN it SHALL include `dropout` (dropout rate)
8. WHEN initializing the configuration THEN it SHALL include `activation` (activation function)
9. WHEN initializing the configuration THEN it SHALL include `label_dim` (label embedding dimension)
10. WHEN initializing the configuration THEN it SHALL include `index_out_dim` (index embedding output dimension)
11. WHEN initializing the transformer THEN it SHALL accept the configuration dataclass as a single parameter
12. WHEN initializing the transformer THEN it SHALL also accept `value_dim`, `n_labels`, `index_dim` as separate parameters derived from data

### Requirement 5: Execution Time Benchmarking

**User Story:** As a scientist working with large models, I want to benchmark transformer execution time for forward and backward passes, so that I can ensure training remains fast enough for my use case.

#### Acceptance Criteria

1. WHEN a speed benchmark is run THEN it SHALL use pytest-benchmark for statistical testing
2. WHEN a speed benchmark measures forward pass THEN it SHALL time the compiled forward function
3. WHEN a speed benchmark measures backward pass THEN it SHALL time `jax.grad` computation
4. WHEN a benchmark is parameterized THEN it SHALL accept `n_tokens` (number of tokens)
5. WHEN a benchmark is parameterized THEN it SHALL accept architecture parameters via configuration dataclass
6. WHEN a benchmark is parameterized THEN it SHALL accept a `time_threshold` in seconds
7. WHEN execution time exceeds threshold THEN the benchmark SHALL fail
8. WHEN the benchmark runs THEN it SHALL compile the model first (excluding compilation from timing)
9. WHEN reporting results THEN pytest-benchmark SHALL provide statistical summaries (mean, stddev, etc.)
10. WHEN the benchmark is marked THEN it SHALL use `@pytest.mark.speed` for standard configurations
11. WHEN the benchmark tests large configurations THEN it SHALL use `@pytest.mark.scale` marker

### Requirement 6: Memory Usage Benchmarking

**User Story:** As a scientist working with memory-constrained hardware, I want to benchmark transformer memory usage for forward and backward passes, so that I can ensure the model fits within available GPU memory.

#### Acceptance Criteria

1. WHEN a memory benchmark is run THEN it SHALL use pytest-memray for memory profiling
2. WHEN a memory benchmark measures forward pass THEN it SHALL profile peak memory during forward execution
3. WHEN a memory benchmark measures backward pass THEN it SHALL profile peak memory during `jax.grad` computation
4. WHEN a benchmark is parameterized THEN it SHALL accept `n_tokens` (number of tokens)
5. WHEN a benchmark is parameterized THEN it SHALL accept architecture parameters via configuration dataclass
6. WHEN a benchmark is parameterized THEN it SHALL accept a `memory_threshold` in MB
7. WHEN peak memory usage exceeds threshold THEN the benchmark SHALL fail
8. WHEN reporting results THEN it SHALL output peak memory allocation in MB
9. WHEN the benchmark is marked THEN it SHALL use `@pytest.mark.speed` for standard configurations
10. WHEN the benchmark tests large configurations THEN it SHALL use `@pytest.mark.scale` marker

## Non-Functional Requirements

### Performance

- Forward pass execution SHALL complete within time thresholds specified in benchmarks
- Backward pass (gradient computation) SHALL complete within time thresholds specified in benchmarks
- Memory usage SHALL stay within memory thresholds specified in benchmarks
- All component tests SHALL run in under 5 seconds (excluding compilation)
- The transformer SHALL be JIT-compilable with JAX for production use

### Usability

- The transformer interface SHALL be simpler than the legacy implementation by using `Tokens` objects
- Configuration SHALL use a dataclass with sensible defaults to reduce boilerplate
- Error messages SHALL clearly indicate shape mismatches when they occur
- Documentation SHALL include examples of creating `Tokens` objects for transformer input

### Maintainability

- Each component (embedding, encoder, decoder) SHALL be in a separate module
- All components SHALL use JAX/Flax type annotations (`Array` from jaxtyping)
- All modules SHALL follow numpy-style docstrings
- Tests SHALL be organized by component (embedding, encoder, decoder, transformer)
- The codebase SHALL pass `pyright` type checking with no errors
- Benchmark code SHALL be in `test/benchmark/` directory
- The project SHALL add `pytest-memray` as a development dependency

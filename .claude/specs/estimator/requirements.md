# Requirements Document: Tokenized Flow Matching Estimator (TFMPE)

## Introduction

The TFMPE (Tokenized Flow Matching for Posterior Estimation) is a unified estimator class that combines sampling and log probability computation for flow matching-based posterior estimation. This consolidates the functionality previously split between `StructuredCNF` and `SFMPE` in the legacy codebase, integrating seamlessly with the new Token-based preprocessing system.

## Alignment with Package Vision

This feature directly supports the package's core goals:
- **TFMPE**: The primary estimator for structured parameter inference problems
- **Sample Efficiency**: Leverages flow matching with continuous normalizing flows for efficient posterior estimation
- **Speed & Scalability**: Combined ODE solving infrastructure and tokenized training loops enable fast, memory-efficient training on hierarchical models

## Requirements

### Requirement 1: ODE Solving Infrastructure

**User Story:** As a researcher, I want helper functions for setting up and solving ODEs for flow matching, so that I can efficiently compute sampling trajectories and log probabilities without worrying about low-level ODE mechanics.

#### Acceptance Criteria

1. WHEN solving forward ODE (t=0 to t=1) for sampling THEN SHALL efficiently compute trajectory using `jax.experimental.ode.odeint` with configurable tolerances
2. WHEN solving backward ODE (t=1 to t=0) for log probability THEN SHALL use augmented state with trace estimation for FFJORD computation
3. WHEN computing trajectories THEN SHALL handle both single samples and batches via vmap
4. WHEN solving ODEs THEN SHALL accept vector field function, initial state, time points, and ODE solver parameters

### Requirement 2: TFMPE Core Class for Sampling and Log Probability

**User Story:** As a researcher, I want a unified TFMPE class that handles both sampling from the posterior and computing log probabilities, so that I have a single interface for inference without worrying about whether I'm using CNF or SFMPE semantics.

#### Acceptance Criteria

1. WHEN calling `sample_posterior()` THEN SHALL return posterior samples by solving forward ODE from base distribution
2. WHEN calling `log_prob_posterior_samples()` THEN SHALL compute log probabilities for provided samples using backward ODE with trace estimation
3. WHEN initializing TFMPE THEN SHALL accept trained vector field network and base distribution specification
4. WHEN sampling THEN SHALL support batch sampling with configurable number of samples
5. WHEN initializing TFMPE THEN SHALL accept a vector field network compatible with Token inputs (structured labels, indices, masks)

### Requirement 3: Integration with Token-Based Preprocessing

**User Story:** As a researcher, I want TFMPE to work seamlessly with the Token-based preprocessing system, so that I can handle structured parameters without manual flattening/unflattening logic.

#### Acceptance Criteria

1. WHEN TFMPE receives data THEN SHALL work with Token objects containing flattened arrays, labels, masks, and independence specifications
2. WHEN sampling THEN SHALL return samples in Token format with appropriate structure metadata
3. WHEN computing log probabilities THEN SHALL accept Token-formatted posterior samples

### Requirement 4: ODE Helper Unit Tests

**User Story:** As a developer, I want thorough unit tests for ODE solving helpers, so that I can verify correctness before building higher-level TFMPE functionality.

#### Acceptance Criteria

1. WHEN running doubling continuous flow test (f(θ) = log(2)·θ) THEN SHALL verify that N(0,1) transforms to N(0,4) within tolerance
2. WHEN testing forward ODE THEN SHALL verify trajectory computation with analytical solutions
3. WHEN testing backward ODE THEN SHALL verify trace-based log probability computation with stochastic tolerance
4. WHEN testing batch operations THEN SHALL verify vmap correctness across sample batches

### Requirement 5: TFMPE Integration and Functional Tests

**User Story:** As a developer, I want integration tests for TFMPE's sampling and log probability methods, so that I can verify end-to-end functionality before implementing training loops.

#### Acceptance Criteria

1. WHEN training TFMPE on synthetic hierarchical data (θ ~ N(0,1), y ~ N(θ, σ²)) THEN SHALL recover base distribution via backward sampling
2. WHEN testing TFMPE THEN SHALL verify log probability computations are accurate within stochastic tolerance (rtol=0.1, atol=0.5)
3. WHEN sampling from posterior THEN SHALL verify consistency between forward ODE sampling and base distribution
4. WHEN training TFMPE THEN SHALL handle structured inputs with labels, masks, and independence specifications

### Requirement 6: Speed-Optimized Training Loop

**User Story:** As a researcher, I want a fast training loop for TFMPE, so that I can quickly iterate on model designs and configurations.

#### Acceptance Criteria

1. WHEN training with `fit_fast()` THEN SHALL use nnx.scan-based training without Python loop overhead
2. WHEN training THEN SHALL support batched loss computation with weighted aggregation (loss × batch_size / total_size)
3. WHEN training THEN SHALL accept validation data and compute validation loss
4. WHEN training THEN SHALL return loss history shape (n_iter, 2) with [train_loss, val_loss]
5. WHEN training THEN SHALL implement Continuous Flow Matching (CFM) loss with linear interpolation between base and data distributions
6. WHEN compiling `fit_fast()` THEN SHALL be fully jittable with `jax.jit()` for performance

### Requirement 7: Memory-Efficient Training Loop with TokenGenerator

**User Story:** As a researcher, I want a memory-efficient training loop for TFMPE on large datasets, so that I can scale to hierarchical problems without memory constraints.

#### Acceptance Criteria

1. WHEN training with `fit_memory_efficient()` THEN SHALL accept TokenGenerator for on-the-fly batch generation
2. WHEN training THEN SHALL avoid materializing entire dataset in memory
3. WHEN training THEN SHALL yield Token batches from generator during iteration
4. WHEN training THEN SHALL support early stopping with validation monitoring
5. WHEN training THEN SHALL implement same CFM loss and loss tracking as speed-optimized loop

### Requirement 8: End-to-End Training Test

**User Story:** As a developer, I want an E2E test that trains TFMPE on a realistic hierarchical model, so that I can verify the complete training pipeline works correctly.

#### Acceptance Criteria

1. WHEN running E2E test THEN SHALL train TFMPE on hierarchical Gaussian linear model with:
   - σ ~ HalfNormal(1) [global parameter]
   - μᵢ ~ Normal(0, 1) for each i [local parameters]
   - yᵢⱼ ~ Normal(μᵢ, σ) for each observation j [observations]
2. WHEN training THEN SHALL verify convergence through loss decrease
3. WHEN testing THEN SHALL verify that trained estimator produces valid samples and log probabilities
4. WHEN testing THEN SHALL complete within reasonable time (< 60 seconds)

### Requirement 9: Performance Benchmarks

**User Story:** As a developer, I want benchmarks for TFMPE training and inference, so that I can track performance regressions and verify scalability.

#### Acceptance Criteria

1. WHEN running speed benchmark THEN SHALL measure training loop iteration time on standard dataset
2. WHEN running scale benchmark THEN SHALL measure training loop iteration time on large hierarchical dataset
3. WHEN running benchmarks THEN SHALL measure sampling speed for different batch sizes
4. WHEN benchmarking THEN SHALL measure log probability computation speed

## Non-Functional Requirements

### Performance
- ODE solving must complete in < 1 second per sample for typical problems
- Speed-optimized training loop must be 3-5x faster than Python loop equivalent
- Log probability computation must use stochastic FFJORD with acceptable variance (rtol=0.1, atol=0.5)
- Memory-efficient training must support datasets with > 100k samples without OOM

### Maintainability
- Code must follow project structure conventions (tfmpe/estimators/ for core, test/ for tests)
- Type annotations required via jaxtyping (Array, PRNGKeyArray, etc.)
- Line length ≤ 80 characters per project guidelines
- Helper functions extracted for long/complex implementations

### Compatibility
- Must integrate with existing Token preprocessing system (tfmpe/preprocessing/)
- Must work with transformer-based vector field networks in tfmpe/nn/transformer
- Must follow JAX/Flax patterns established in legacy code

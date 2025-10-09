# Implementation Plan: Bottom-Up Training Algorithm

## Task Overview

Implementation follows a test-driven approach: first write the E2E test to establish expected behavior, then implement the algorithm incrementally to make the test pass. The approach prioritizes completing a working end-to-end test before implementation details, enabling rapid iteration and validation.

## Steering Document Compliance

- **Functions over Classes** (tech.md): `fit_bottom_up()` is a standalone function, not a class
- **Type Annotations** (tech.md): Full jaxtyping with `Array`, `PRNGKeyArray`, `Callable` types
- **Code Quality** (tech.md): <80 char lines, reuse existing components, clear logic
- **Project Structure** (structure.md): Single file addition to `tfmpe/estimators/training.py`; tests in existing `test_e2e_training.py`

## Atomic Task Requirements

Each task is scoped to 1-3 related files, completable in 15-30 minutes, with a single testable outcome.

## Tasks

### Task 1: Create E2E Test for Bottom-Up Training (TEST FIRST)

- [x] 1. Write `TestE2ETrainingBottomUp` test class in test_e2e_training.py
  - File: `test/test_estimators/test_e2e_training.py`
  - Create new test class `TestE2ETrainingBottomUp` after existing `TestE2ETrainingFast` class
  - Add test method `test_fit_bottom_up_trains_successfully()` that:
    - Uses `create_hierarchical_gaussian_data()` to generate 2 rounds of data (n_samples=10 per round)
    - Calls `fit_bottom_up(tfmpe, y_obs, simulator_fn, prior_fn, n_rounds=2, n_samples_per_round=10, opt, n_iter=5, rng)`
    - Asserts returned TFMPE is valid (not None, has expected attributes)
    - Asserts losses is list of 2 tuples (one per round)
    - Asserts all losses are finite: `jnp.all(jnp.isfinite(losses[i][j]))` for each round and train/val pair
  - Add test method `test_fit_bottom_up_loss_decreases()` that:
    - Verifies no catastrophic divergence by checking train loss in later iterations < 10x first iteration per round
  - Add minimal test helper `create_simulator_fn()` that wraps existing model simulator
  - Add minimal test helper `create_prior_fn()` that returns sampled parameters from prior
  - _Requirements: 1.1_
  - _Leverage: test/test_estimators/test_e2e_training.py (existing test patterns), tfmpe/preprocessing/tokens.py_

### Task 2: Implement Core `fit_bottom_up()` Function Skeleton

- [x] 2. Implement `fit_bottom_up()` with corrected signature
  - File: `tfmpe/estimators/training.py`
  - **Final Signature**:
    ```python
    def fit_bottom_up(
        tfmpe: TFMPE,
        y_obs: Dict[str, Array],
        simulator_fn: Callable[[PRNGKeyArray, Dict, int], Dict],
        prior_fn: Callable[[PRNGKeyArray, int, int], Dict],
        local_fn: Callable[[PRNGKeyArray, Dict, int], Dict],
        global_names: List[str],
        n_groups: int,
        n_rounds: int,
        n_samples_per_round: int,
        n_val_samples: int,
        opt: nnx.Optimizer,
        n_iter_per_round: int,
        rng: PRNGKeyArray,
        independence: Independence,
    ) -> Tuple[TFMPE, List[Tuple[Array, Array, Array, Array]]]:
    ```
  - Key aspects:
    - `global_names`: List of parameter names that are global (non-local)
    - `n_val_samples`: Configurable validation sample count
    - Return type: List of 4-tuples (train_loss_local, val_loss_local, train_loss_global, val_loss_global)
    - Raises NotImplementedError if n_rounds > 1
  - Imports: combine_tokens from tfmpe.preprocessing.combine
  - Purpose: Establish correct function interface matching legacy algorithm

### Task 3: Implement Round 0 (Two-fit with n=1 local, then n=n_groups)

- [x] 3. Implement Round 0 in `fit_bottom_up()` function
  - File: `tfmpe/estimators/training.py`
  - **Step 3a - First fit: Train p(y|theta) with n=1 local**:
    - Sample theta from prior with n=1
    - Simulate y observations with n=1
    - Create tokens: params=theta, context=y
    - Call fit_fast() with n_val_samples validation samples
    - Store: train_loss_local, val_loss_local
  - **Step 3b - Prepare data for second fit: Generate theta_full, sample z**:
    - Extract globals: theta_global = {k: v for k, v in theta if k in global_names}
    - Generate locals with n=n_groups: theta_local_expanded = local_fn(rng, theta_global, n_groups)
    - Combine: theta_full = {**theta_global, **theta_local_expanded}
    - Expand y to n_groups via broadcasting
    - Sample z via tfmpe.sample_posterior with context=theta_full, params=y_template
  - **Step 3c - Second fit: Train p(theta,z|y) with n=n_groups**:
    - Create param tokens from theta_full
    - Create context tokens from sampled z
    - Do NOT use combine_tokens() - use theta_full and z_full directly
    - Call fit_fast() with n_val_samples validation samples
    - Store: train_loss_global, val_loss_global
  - Return 4-tuple: (train_loss_local, val_loss_local, train_loss_global, val_loss_global)
  - Purpose: Full Round 0 implementation with two-fit pattern matching legacy algorithm

### Task 4: Implement Rounds 1+ (Currently NotImplementedError)

- [ ] 4. Round 1+ implementation deferred
  - File: `tfmpe/estimators/training.py`
  - **Current Status**: Function raises NotImplementedError if n_rounds > 1
  - **Future Work**: Implement iterative refinement similar to Round 0 but with:
    - Posterior sampling from previous round instead of prior sampling
    - Same two-fit pattern (local n=1, then global n=n_groups)
  - Purpose: Support multi-round training (deferred for MVP)

### Task 5: Finalize and Debug Integration

- [ ] 5. Complete `fit_bottom_up()` implementation and fix shape issues
  - File: `tfmpe/estimators/training.py`
  - **Current implementation status**:
    - Function signature: ✓ Correct
    - Round 0 logic: ✓ Implemented
    - Validation: ✓ NotImplementedError for n_rounds > 1
    - Return value: ✓ 4-tuple losses
  - **Known issues to fix**:
    - Shape mismatch in transformer embedding (combining tokens from n=1 and n=n_groups)
    - Solution: Remove combine_tokens() calls in second fit, use theta_full and z_full directly as created
  - **Testing**: Run E2E test and fix shape mismatches
  - Purpose: Complete MVP implementation (Round 0 only) with passing E2E test
  - _Requirements: 4.1, 3.1_

### Task 6: Run E2E Test and Debug Integration Issues

- [ ] 6. Execute E2E test and fix integration issues
  - Files: `test/test_estimators/test_e2e_training.py`, `tfmpe/estimators/training.py`
  - Run: `python -m pytest test/test_estimators/test_e2e_training.py::TestE2ETrainingBottomUp::test_fit_bottom_up_trains_successfully -xvs`
  - Expected failures/issues:
    - Import errors: Add missing imports in training.py (e.g., Labeller, Tokens)
    - Shape mismatches: Debug simulator_fn output shapes vs. expected token shapes
    - PRNG key management: Fix key splitting if needed
    - Token reconstruction: Verify decode/encode consistency
  - Debug by:
    - Adding print statements for token shapes and data structure
    - Checking y_simulated structure matches y_obs keys
    - Validating theta_samples dict keys match labeller expectations
  - Stop when test runs without import/attribute errors (test assertions may still fail)
  - Purpose: Identify and fix integration issues before final validation
  - _Requirements: 1.1_
  - _Leverage: pytest runner, existing test patterns_

### Task 7: Validate Loss Computation and Convergence

- [ ] 7. Fix loss computation and verify convergence behavior
  - Files: `tfmpe/estimators/training.py`, `test/test_estimators/test_e2e_training.py`
  - Run full E2E test: `python -m pytest test/test_estimators/test_e2e_training.py::TestE2ETrainingBottomUp -xvs`
  - Debug assertions:
    - If losses is wrong type/shape: Fix all_losses construction in loop
    - If losses are NaN: Check token creation (all keys present, shapes consistent)
    - If losses diverge catastrophically: Reduce learning rate in test setup or n_iter
    - If test passes but seems wrong: Verify loss values make sense (should decrease over iterations within each round)
  - Verify test_fit_bottom_up_loss_decreases passes (no 10x divergence per round)
  - Add assertion for losses list length: `len(all_losses) == n_rounds`
  - Purpose: Ensure training is numerically stable and converging
  - _Requirements: 1.2, 1.3_
  - _Leverage: cfm_loss() validation patterns_

### Task 8: Add Documentation and Comments

- [ ] 8. Add inline documentation and docstring enhancements
  - File: `tfmpe/estimators/training.py`
  - Add inline comments explaining:
    - Token creation for each round (why context vs params)
    - PRNG key splitting strategy
    - Difference between local likelihood and global posterior objectives
    - Non-jittable nature and why (Python loop over rounds)
  - Enhance docstring with:
    - Note about non-jittable behavior
    - Example usage section showing typical call pattern
    - Description of returned loss structure
  - Add section in docstring about when to use fit_bottom_up vs fit_fast (multi-round refinement scenario)
  - Purpose: Make implementation maintainable and usable
  - _Requirements: 4.1_
  - _Leverage: fit_fast() docstring as reference_

---

## Testing Approach

1. **Test-First**: Task 1 writes E2E test before implementation
2. **Iterative Debugging**: Tasks 3-5 implement incrementally; Task 6 runs test and fixes issues
3. **Validation**: Task 7 verifies loss behavior and convergence
4. **Integration**: Tasks verify end-to-end flow with real TFMPE and hierarchical Gaussian data

## Expected Implementation Complexity

- **Straightforward Sequential Logic**: No complex algorithms; reuse existing fit_fast and TFMPE methods
- **Token API Mastery Required**: Understanding Tokens.from_pytree(), decode_keys(), and Labeller is critical
- **PRNG Management**: Standard JAX pattern; split key per round and fit_fast call
- **Minimal New Code**: ~100-150 lines of fit_bottom_up implementation (rest is existing component reuse)

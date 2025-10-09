# Implementation Plan: Remove TokenView and Enhance Token Updates

## Task Overview

This implementation removes the `TokenView` class and enhances the `Tokens` class with slicing and data update capabilities. Tasks proceed in dependency order:
1. Add `select_tokens()` and `with_data()` methods to `Tokens`
2. Update ODE solver signatures to accept context and params
3. Update transformer to remove `TokenView` type hints
4. Consolidate tests without redundancy
5. Remove `TokenView` module and clean up imports

## Steering Document Compliance

- **tech.md**: All type annotations use `Tokens` directly (no `Union` types); no new class hierarchy
- **structure.md**: Single cohesive token abstraction in `tfmpe/preprocessing/tokens.py`; cleanup of `token_view.py`

## Atomic Task Requirements

Each task:
- Touches 1-3 related files maximum
- Completable in 15-30 minutes
- Has one testable outcome
- Specifies exact file paths
- Minimizes context switching

## Tasks

- [x] 1. Modify `select_tokens()` method to return Tokens instead of TokenView
  - **Files**: `tfmpe/preprocessing/tokens.py`
  - **Implementation**:
    - Locate existing `select_tokens()` method (currently returns `TokenView`)
    - Modify to compute and return `Tokens` object instead:
      - Compute token indices for selected keys using slice metadata
      - Extract data slice: `self.data[..., indices, :]`
      - Extract labels slice: `self.labels[..., indices]`
      - Extract padding_mask slice if present
      - Extract functional_inputs slice if present
      - Extract self_attention_mask submatrix using `jnp.ix_(indices, indices)`
      - Re-index slice metadata: compute new offsets starting from 0
      - Return new `Tokens` object with sliced data and re-indexed metadata
    - Keep existing validation (selected keys must exist; raise `KeyError` if not)
  - **Tests**: Will be validated by `test_select_tokens_*` suite
  - _Leverage: `SliceInfo` namedtuple, existing logic currently in `TokenView._token_indices`, `TokenView.slices`, etc._
  - _Requirements: 1.1_

- [ ] 2. Add `with_data()` method to Tokens class
  - **Files**: `tfmpe/preprocessing/tokens.py`
  - **Implementation**:
    - Add method that takes `new_data: Array` parameter
    - Validate `new_data.shape == self.data.shape`; raise `ValueError` with helpful message if not
    - Return new `Tokens` with:
      - `data=new_data`
      - All other fields unchanged (labels, masks, slices, metadata, independence)
    - Keep implementation simple (no complex logic)
  - **Tests**: Will be validated by `test_with_data_*` suite
  - _Leverage: None (straightforward replacement)_
  - _Requirements: 3.1_

- [x] 3. Update `solve_forward_ode()` signature in ODE module
  - **Files**: `tfmpe/estimators/ode.py`
  - **Implementation**:
    - Update function signature to accept `context: Tokens, params: Tokens`
    - Update docstring to document new parameters
    - Update inner `ode_func()` to receive params state and call `vf_fn(context, y_params, t)`
    - Extract returned params object and integrate with diffrax
    - Ensure context is passed unchanged through ODE integration
  - **Tests**: Will be validated by ODE solver tests
  - _Leverage: Existing diffrax integration pattern_
  - _Requirements: 2.1_

- [x] 4. Update `solve_backward_ode()` signature in ODE module
  - **Files**: `tfmpe/estimators/ode.py`
  - **Implementation**:
    - Update function signature to accept `context: Tokens, params: Tokens`
    - Update docstring to document new parameters
    - Update inner `ode_func()` to call `vf_fn(context, y_params, 1.0 - t)` with negation
    - Ensure context remains fixed during backward integration
  - **Tests**: Will be validated by ODE solver tests
  - _Leverage: Existing backward ODE pattern_
  - _Requirements: 2.1_

- [x] 5. Update `solve_augmented_ode()` signature in ODE module
  - **Files**: `tfmpe/estimators/ode.py`
  - **Implementation**:
    - Update function signature to accept `context: Tokens, params: Tokens`
    - Update docstring to document new parameters
    - Update `augmented_ode_func()` to call `vf_fn(context, x, actual_time)`
    - Preserve trace estimation logic (unchanged)
    - Ensure context passed at each step
  - **Tests**: Will be validated by augmented ODE tests
  - _Leverage: Existing FFJORD implementation_
  - _Requirements: 2.1_

- [ ] 6. Update `batch_solve_forward_ode()` signature in ODE module
  - **Files**: `tfmpe/estimators/ode.py`
  - **Implementation**:
    - Update function signature to accept `context: Tokens, params_batch: Tokens`
    - Update docstring
    - Use `jax.vmap()` over `solve_forward_ode()` for batch dimension
    - Handle vmap correctly with context (not vmapped) and params batch (vmapped)
  - **Tests**: Will be validated by batch ODE tests
  - _Leverage: Existing vmap pattern_
  - _Requirements: 2.1_

- [ ] 7. Update `batch_solve_augmented_ode()` signature in ODE module
  - **Files**: `tfmpe/estimators/ode.py`
  - **Implementation**:
    - Update function signature to accept `context: Tokens, params_batch: Tokens`
    - Update docstring
    - Use `jax.vmap()` over `solve_augmented_ode()` correctly
    - Handle RNG splitting for batch dimension
  - **Tests**: Will be validated by batch augmented ODE tests
  - _Leverage: Existing vmap pattern_
  - _Requirements: 2.1_

- [x] 8. Update Embedding class type hints
  - **Files**: `tfmpe/nn/transformer/embedding.py`
  - **Implementation**:
    - Remove import: `from ...preprocessing.token_view import TokenView`
    - Update `__call__()` signature: change `tokens: Union[Tokens, TokenView]` to `tokens: Tokens`
    - Update docstring to reference `Tokens` only
    - No logic changes (properties work identically)
  - **Tests**: Existing embedding tests should still pass
  - _Leverage: No changes needed to logic_
  - _Requirements: 4.1_

- [x] 9. Update Transformer class type hints and methods
  - **Files**: `tfmpe/nn/transformer/transformer.py`
  - **Implementation**:
    - Remove import: `from ...preprocessing.token_view import TokenView`
    - Update `encode()` signature: change `tokens: Union[Tokens, TokenView]` to `tokens: Tokens`
    - Update `decode()` signature: change `tokens: Union[Tokens, TokenView]` to `tokens: Tokens`
    - Update `forward()` signature: change `context: TokenView, params: TokenView` to `context: Tokens, params: Tokens`
    - Update all docstrings to reference `Tokens` only
    - No logic changes (method calls work identically)
  - **Tests**: Existing transformer tests should still pass
  - _Leverage: No changes needed to logic_
  - _Requirements: 4.1_

- [x] 10. Repurpose TokenView tests to test Tokens.select_tokens()
  - **Files**: `test/test_preprocessing/test_tokens_dynamic.py`
  - **Implementation**:
    - Modify existing `test_select_tokens_*` functions to test `Tokens.select_tokens()` instead of `TokenView`
    - Change assertions from `isinstance(view, TokenView)` to `isinstance(sliced, Tokens)`
    - Keep all validation logic identical (data, labels, masks, slices)
    - Delete duplicate test functions (keep only one version per behavior)
    - Add new test: `test_select_tokens_slices_reindexed()` to verify offset re-indexing
    - Examples of tests to consolidate:
      - `test_select_tokens_subset_data`
      - `test_select_tokens_subset_labels`
      - `test_select_tokens_self_attention_mask`
      - `test_tokenview_decode_consistent_with_tokens` → rename to `test_select_tokens_decode`
  - **Tests**: All modified tests should pass
  - _Leverage: Existing test logic from `TokenView` tests_
  - _Requirements: 5.1_

- [ ] 11. Add tests for Tokens.with_data()
  - **Files**: `test/test_preprocessing/test_tokens_dynamic.py`
  - **Implementation**:
    - Add `test_with_data_full_replacement()`: verify data is replaced, shape validated
    - Add `test_with_data_preserves_metadata()`: verify labels, masks, slices unchanged
    - Add `test_with_data_preserves_independence()`: verify independence spec unchanged
    - Add `test_with_data_shape_validation()`: verify shape mismatch raises ValueError
    - Add `test_with_data_sample_dims()`: verify works with sample dimensions
    - Each test follows existing test patterns (fixtures, assertions)
  - **Tests**: All new tests should pass
  - _Leverage: Existing test fixtures and patterns_
  - _Requirements: 3.1_

- [ ] 12. Run type checking and fix type errors
  - **Files**: `tfmpe/` (all modified files)
  - **Implementation**:
    - Run `pyright` on entire codebase
    - Fix any type errors from signature changes
    - Verify no `Union[Tokens, TokenView]` references remain
    - Ensure all function parameters correctly typed
  - **Tests**: `pyright` should pass with no errors
  - _Leverage: No changes to code logic, only type annotations_
  - _Requirements: 4.1_

- [ ] 13. Run test suite to verify no regressions
  - **Files**: `test/` (entire test directory)
  - **Implementation**:
    - Run `python -m pytest test/ -m "not slow"` to verify fast tests pass
    - Run `python -m pytest test/test_preprocessing/` specifically to verify token tests
    - Fix any test failures arising from changes
    - Verify no redundant tests exist (consolidation successful)
  - **Tests**: All tests should pass
  - _Leverage: Existing pytest setup_
  - _Requirements: 5.1_

- [x] 14. Delete TokenView module and clean up imports
  - **Files**: `tfmpe/preprocessing/token_view.py`, `tfmpe/preprocessing/__init__.py`
  - **Implementation**:
    - Delete file: `tfmpe/preprocessing/token_view.py`
    - Remove from `__init__.py`: any exports of `TokenView`
    - Verify no remaining imports of `token_view` module in codebase (should have been removed in tasks 8-9)
    - Use grep/search to confirm no orphaned references
  - **Tests**: Type checker and imports should pass
  - _Leverage: No changes needed_
  - _Requirements: 1.1_

- [x] 15. Verify complete integration
  - **Files**: `tfmpe/` (all files)
  - **Implementation**:
    - Run full test suite: `python -m pytest test/ -m "not slow"`
    - Run `pyright` to verify all type checking passes
    - Verify no import errors or orphaned references to `TokenView`
    - Document any behavioral changes in docstrings
  - **Tests**: All tests pass, no type errors, clean imports
  - _Leverage: Existing CI/testing infrastructure_
  - _Requirements: 4.1, 5.1_


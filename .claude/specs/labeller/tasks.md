# Implementation Plan: Labeller Refactoring

## Task Overview

The implementation follows a phased approach to minimize breaking changes and allow validation at each step:
1. Create and test `Labeller` type
2. Integrate `Labeller` into `Tokens` and remove `label_map`
3. Add `key_order` property to `Tokens` and remove field
4. Remove `select_tokens()` method
5. Refactor `combine_tokens()` for flexible key handling
6. Update all integration tests

## Steering Document Compliance

All tasks follow `structure.md` conventions:
- Test files mirror source file structure
- Named imports at top, following standard order
- Type annotations using jaxtyping
- Functions over classes approach
- JAX-based array operations

## Atomic Task Requirements

Each task is designed to:
- Touch 1-3 related files maximum
- Complete in 15-30 minutes
- Have single testable outcome
- Include specific file paths
- Be clear for agent execution

## Tasks

- [x] 1. Create Labeller dataclass in tfmpe/preprocessing/utils.py
  - File: `tfmpe/preprocessing/utils.py`
  - Add `Labeller` dataclass with `label_map: Dict[str, int]` field
  - Add docstring explaining purpose (stores global label mappings)
  - Purpose: Define the Labeller type to store key-to-label mappings
  - _Leverage: existing `SliceInfo` and `Independence` in same file_
  - _Requirements: 1.1_

- [x] 2. Implement Labeller.label() method in tfmpe/preprocessing/utils.py
  - File: `tfmpe/preprocessing/utils.py` (continue from task 1)
  - Implement `.label(slices: Dict[str, SliceInfo]) -> Array` method
  - Iterate through slices in offset order, create per-key label arrays using label_map, concatenate
  - Return 1D array of shape (n_total_tokens,) with label indices
  - Add comprehensive docstring with parameter descriptions
  - Purpose: Enable label array generation from Labeller
  - _Leverage: label generation logic from `Tokens.from_pytree()` lines 129-155 in tokens.py_
  - _Requirements: 1.2, 1.3_

- [x] 3. Create unit tests for Labeller in test/test_preprocessing/test_labeller.py
  - File: `test/test_preprocessing/test_labeller.py` (new file)
  - Test Labeller initialization with label_map
  - Test `.label()` with single and multiple keys
  - Test `.label()` with different event_shapes
  - Test label indices are correct per key and array shape
  - Test label consistency across multiple calls
  - Test error handling (empty label_map, missing keys in slices)
  - Purpose: Comprehensive Labeller validation
  - _Leverage: fixtures from conftest.py, test patterns from test_tokens_dynamic.py_
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 4. Add Labeller to preprocessing module exports in tfmpe/preprocessing/__init__.py
  - File: `tfmpe/preprocessing/__init__.py`
  - Add `Labeller` to imports from `utils`
  - Add `Labeller` to `__all__` list
  - Purpose: Make Labeller publicly available through preprocessing module
  - _Leverage: existing module structure_
  - _Requirements: 1.1_

- [x] 5. Remove label_map field from Tokens dataclass in tfmpe/preprocessing/tokens.py
  - File: `tfmpe/preprocessing/tokens.py`
  - Remove `label_map: Dict[str, int]` field definition (line 62-63)
  - Remove `label_map` construction from `from_pytree()` (line 129)
  - Remove `label_map` from return statement in `from_pytree()` (line 179)
  - Remove `label_map` from `with_values()` method (line 442)
  - Remove `label_map` from PyTree aux_data in `tree_unflatten()` (line 467)
  - Update docstrings to reflect field removal
  - Purpose: Decouple label mapping from Tokens class
  - _Leverage: existing Tokens structure_
  - _Requirements: 3.1_

- [x] 6. Add key_order property to Tokens class in tfmpe/preprocessing/tokens.py
  - File: `tfmpe/preprocessing/tokens.py`
  - Remove `key_order: List[str]` field definition (line 49-50)
  - Remove `key_order` construction from `from_pytree()` (line 126)
  - Remove `key_order` from return statement in `from_pytree()` (line 179)
  - Remove `key_order` from `with_values()` method (line 443)
  - Remove `key_order` from PyTree aux_data in `tree_unflatten()` (line 468)
  - Add `@property` `key_order()` method that derives order from slices by sorting keys by slice offset
  - Update docstrings to document new property
  - Purpose: Eliminate redundant key_order storage while maintaining API compatibility
  - _Leverage: existing slices structure_
  - _Requirements: 3.1_

- [x] 7. Remove select_tokens() method from Tokens class in tfmpe/preprocessing/tokens.py
  - File: `tfmpe/preprocessing/tokens.py`
  - Remove entire `select_tokens()` method (lines 239-365)
  - Update Tokens docstring to remove mention of select_tokens
  - Purpose: Simplify Tokens API by removing slicing method
  - _Leverage: none (pure removal)_
  - _Requirements: 3.2_

- [x] 8. Update combine_tokens() to support different key_order in tfmpe/preprocessing/combine.py
  - File: `tfmpe/preprocessing/combine.py`
  - Remove key_order equality check (lines 47-51)
  - Compute union of all keys from both tokens: `union_keys = list(dict.fromkeys(tokens1.key_order + tokens2.key_order))`
  - Refactor max_event_shapes dict building to iterate over union_keys instead of tokens1.key_order
  - For each key in union_keys:
    - Get event_shape from whichever token has the key (or max if both have it)
    - Use union_keys order when building new slices offsets
  - Update _build_combined_labels() to work with union_keys
  - Update _build_combined_padding_mask() to work with union_keys
  - Update docstring to reflect new behavior (union of keys)
  - Purpose: Enable combining tokens with different keys
  - _Leverage: existing padding and masking logic, Tokens.key_order property_
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 9. Create integration tests for combine_tokens() with different keys in test/test_preprocessing/test_combine.py
  - File: `test/test_preprocessing/test_combine.py` (add to existing if exists)
  - Test combining tokens with identical key_order (existing behavior should still work)
  - Test combining tokens with subset relationship (one token has keys that other doesn't)
  - Test combining tokens with completely different keys
  - Verify combined slices are correct, labels are correct, padding mask is correct
  - Purpose: Validate flexible combine_tokens() behavior
  - _Leverage: fixtures from conftest.py, token creation patterns_
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 10. Update test_tokens_dynamic.py to remove select_tokens() tests in test/test_preprocessing/test_tokens_dynamic.py
  - File: `test/test_preprocessing/test_tokens_dynamic.py`
  - Remove or refactor tests that specifically test `select_tokens()` method
  - Tests to refactor (approximately 9 tests):
    - test_select_tokens_* (all variants)
  - For each test, either:
    - Delete if testing select_tokens-specific behavior
    - Refactor to test direct Tokens creation with selected keys if testing general slicing behavior
  - Update or remove assertions on `label_map` (keep assertions on key_order via property)
  - Purpose: Clean up tests for removed select_tokens method
  - _Leverage: existing token creation utilities in conftest.py_
  - _Requirements: 3.2, 5.1_

- [x] 11. Update test_transformer.py fixtures to remove select_tokens() in test/test_nn/test_transformer/test_transformer.py
  - File: `test/test_nn/test_transformer/test_transformer.py`
  - Update `context_tokens` fixture (line 49) to create Tokens directly instead of using select_tokens
  - Update `param_tokens` fixture (line 55) to create Tokens directly instead of using select_tokens
  - Ensure resulting Tokens have same structure as before (same keys, slices, labels)
  - Purpose: Adapt integration tests to use new Tokens creation approach
  - _Leverage: Tokens.from_pytree() and fixtures_
  - _Requirements: 5.2_

- [x] 12. Verify pyright type checking passes in tfmpe/
  - Command: `pyright tfmpe/preprocessing/`
  - Ensure no new type errors introduced by:
    - Labeller type annotations
    - Tokens modifications (property)
    - combine_tokens() changes
  - Fix any type issues found
  - Purpose: Ensure code quality standards met
  - _Leverage: existing pyright configuration_
  - _Requirements: Non-functional (Maintainability)_

- [x] 13. Run all test suites to verify integration in test/
  - Command: `python -m pytest test/` (all tests)
  - Verify all tests pass including:
    - test_preprocessing/ (new Labeller tests, updated Tokens tests)
    - test_nn/ (transformer tests with updated fixtures)
    - Any other integration tests
  - Fix any failing tests
  - Purpose: End-to-end validation of entire refactoring
  - _Leverage: existing test infrastructure_
  - _Requirements: 5.1, 5.2, 5.3, Non-functional (all)_

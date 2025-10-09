# Requirements Document: Labeller Refactoring

## Introduction

Currently, the `Tokens` class uses `select_tokens()` to create sliced token sets with selected keys. This method was originally designed to tie together different token instances with shared independence and label data. However, with the introduction of the `Independence` type, the purpose of `Tokens.select_tokens()` has become conceptually redundant—it's silly to create a global `Tokens` instance just to call `select_tokens()`.

This refactoring introduces a new `Labeller` type that decouples global labelling information from token data. The `Labeller` stores only the label mapping (key → integer index) and provides a `.label()` method that generates label arrays from token slice information. This allows individual token instances to be created independently while maintaining consistent labeling through a shared `Labeller` instance. Additionally, `Tokens.key_order` will become a computed property derived from slices, eliminating redundant storage.

## Alignment with Package Vision

This refactoring supports the package goals of **maintainability** and **efficient data preprocessing**:
- **Cleaner API**: Removes conceptual confusion by separating labelling concerns from token slicing
- **Better separation of concerns**: `Labeller` handles labels, `Tokens` focuses on data representation
- **More flexible token usage**: Tokens can be created and manipulated independently without requiring a global `Tokens` instance for labeling context
- **DRY principle**: Eliminates duplicate key storage by keeping only `label_map` in `Labeller` and deriving `key_order` from `slices`
- **Simplified `Tokens` class**: Reduces class complexity by removing redundant fields while maintaining backward compatibility via properties

## Requirements

### Requirement 1: Create Labeller Type

**User Story:** As a developer, I want a `Labeller` type that stores global label mappings, so that any token instance can have consistent labels with others without needing a global `Tokens` instance.

#### Acceptance Criteria

1. WHEN a `Labeller` is instantiated with a label_map THEN it SHALL accept `Dict[str, int]` mapping key names to integer indices
2. WHEN a `Labeller` has a `.label()` method called with slices THEN it SHALL return a 1D label array with integer labels for all keys in slice offset order
3. WHEN different `Labeller` instances are created with the same label_map THEN they SHALL produce identical label arrays
4. WHEN a `Labeller` is used with multiple `Tokens` instances THEN all instances SHALL share consistent label values across keys

### Requirement 2: Unit Test Labeller

**User Story:** As a developer, I want comprehensive unit tests for `Labeller`, so that I can verify its correctness and catch regressions.

#### Acceptance Criteria

1. WHEN testing `Labeller` initialization THEN unit tests SHALL verify label_map storage
2. WHEN testing `.label()` method THEN unit tests SHALL verify label arrays have correct indices and shape (n_total_tokens,)
3. WHEN testing `.label()` with different slice configurations THEN unit tests SHALL verify correct label generation for various event_shapes and key types
4. WHEN testing `Labeller` with edge cases THEN unit tests SHALL cover single-key, multi-key scenarios and error conditions

### Requirement 3: Remove select_tokens Method

**User Story:** As a developer, I want `select_tokens()` removed from the `Tokens` class, so that token slicing logic is simplified and the API is cleaner.

#### Acceptance Criteria

1. WHEN `select_tokens()` is removed from `Tokens` THEN existing code paths that use it SHALL be updated to use alternative approaches
2. WHEN existing integration tests use `select_tokens()` THEN they SHALL be refactored to directly create `Tokens` instances with selected keys or use alternative filtering mechanisms
3. WHEN removing `select_tokens()` THEN the `Tokens.label_map` field SHALL be removed and `Tokens.key_order` SHALL become a computed property

### Requirement 4: Relax combine_tokens Key Order Constraint

**User Story:** As a developer, I want `combine_tokens()` to work with tokens that have different key_order, so that token combination is more flexible.

#### Acceptance Criteria

1. WHEN combining two `Tokens` instances with different key_order THEN `combine_tokens()` SHALL succeed instead of raising ValueError
2. WHEN combining tokens with different keys THEN the result SHALL contain the union of all keys from both tokens
3. WHEN a key appears in only one input token THEN the combined result SHALL use that token's slice info for that key
4. WHEN combining tokens with overlapping keys THEN the result SHALL use max event_shape per key (existing behavior)

### Requirement 5: Update Integration Tests

**User Story:** As a developer, I want the refactored code integrated into all existing test suites, so that the new `Labeller` is validated end-to-end.

#### Acceptance Criteria

1. WHEN running existing integration tests in `test_tokens_dynamic.py` THEN all tests SHALL pass with refactored code
2. WHEN running transformer integration tests in `test_transformer.py` THEN all tests SHALL pass with the new `Labeller` used for label generation
3. WHEN a test previously used `select_tokens()` THEN the test SHALL be updated to construct tokens directly with selected keys or use alternative mechanisms

## Non-Functional Requirements

### Performance
- Labeller initialization and `.label()` method calls SHALL complete in negligible time (no performance regression)
- Label array generation SHALL use the same efficient JAX operations as current implementation
- Computing `key_order` property from slices SHALL be efficient
- `combine_tokens()` performance SHALL not degrade when handling different key orders

### Usability
- `Labeller` API SHALL be intuitive and follow existing naming conventions (consistent with `Tokens` and `Independence`)
- `Labeller` SHALL be documented with docstrings explaining purpose and usage
- Type annotations SHALL be comprehensive and include jaxtyping types
- `Tokens.key_order` property SHALL maintain same interface as before (no code changes needed)

### Maintainability
- Removal of `label_map` and `key_order` fields from `Tokens` SHALL reduce class complexity
- Code SHALL pass `pyright` type checking with no new issues
- All tests SHALL pass including integration tests
- `combine_tokens()` refactoring SHALL maintain readability and documentation

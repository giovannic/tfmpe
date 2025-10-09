# Requirements Document: Remove TokenView and Enhance Token Updates

## Introduction

This refactoring addresses fundamental design flaws in the token abstraction layer:
1. **Incorrect type contracts** - Vector field functions assume `Array` input when they should accept `Tokens`
2. **Cumbersome abstraction** - `TokenView`'s lazy slicing philosophy conflicts with continuous flow requirements that need frequent token updates
3. **Type barriers** - Alternate types (`TokenView` vs `Token`) force defensive type-checking instead of duck typing

The solution removes `TokenView` entirely and adds direct data vector update methods to `Tokens`.

## Alignment with Package Vision

TFMPE's success depends on efficient continuous flow matching with frequent parameter updates. The current design creates friction in the update path. By simplifying to a single cohesive abstraction (`Tokens` only), we improve:
- **Maintainability**: Eliminate defensive type checks; rely on structural typing
- **Efficiency**: Streamline the update path for continuous flows
- **Clarity**: Single consistent interface reduces cognitive overhead

## Requirements

### Requirement 1: Remove TokenView Class and Preserve select_tokens Functionality

**User Story:** As a developer, I want to eliminate the `TokenView` class while preserving its slicing functionality, so that I have a single consistent token abstraction without type-checking friction.

#### Acceptance Criteria

1. WHEN `select_tokens()` is called THEN it SHALL return a new `Tokens` object (not `TokenView`) containing only selected keys
2. IF data is sliced from parent's flat array THEN the returned `Tokens` SHALL have re-indexed slice metadata (offsets starting at 0)
3. WHEN accessing properties on sliced `Tokens` THEN they SHALL work identically to original implementation (data, labels, masks, slices)
4. IF the `TokenView` class is removed THEN imports and module references SHALL be cleaned up entirely

### Requirement 2: Update Vector Field Function Signatures

**User Story:** As a user, I want vector field functions to accept `Tokens` directly, so that type contracts match actual usage patterns.

#### Acceptance Criteria

1. WHEN a vector field function is defined THEN it SHALL accept `Tokens` as first parameter (not `Array`)
2. IF transformer outputs vector field THEN it SHALL return output shaped/typed for `Tokens` consumption
3. WHEN ODE solvers call vector fields THEN they SHALL pass `Tokens` directly
4. IF documentation specifies function signatures THEN it SHALL reflect `Tokens` parameter

### Requirement 3: Add Direct Data Vector Update Method to Tokens

**User Story:** As a developer using continuous flows, I want a direct method to update the entire token data vector, so that I can efficiently replace parameter tokens during training/sampling.

#### Acceptance Criteria

1. WHEN updating a token data vector THEN there SHALL be a method that accepts the new flat array directly (not through keys)
2. WHEN using the update method THEN data consistency SHALL be maintained (labels, masks, slices unchanged)
3. IF shape validation fails THEN validation errors SHALL provide clear messaging
4. WHEN vector field function returns updated `Tokens` THEN it replaces the entire state seamlessly

### Requirement 4: Update All Call Sites

**User Story:** As a maintainer, I want all code calling `TokenView` or using old vector field signatures to be updated, so that the system is internally consistent.

#### Acceptance Criteria

1. WHEN code previously called `select_tokens()` THEN it SHALL work identically with new `Tokens`-returning version
2. IF code previously extracted data from `TokenView` THEN it SHALL work with returned `Tokens` directly
3. WHEN tests verify token operations THEN redundant tests SHALL be eliminated to avoid duplication
4. IF type errors arise during refactoring THEN they SHALL be resolved before completion

### Requirement 5: Consolidate Tests Without Redundancy

**User Story:** As a test maintainer, I want to repurpose `TokenView` tests into `Tokens` tests without creating redundant coverage, so that test suite remains lean and maintainable.

#### Acceptance Criteria

1. WHEN `TokenView` tests are repurposed THEN shared functionality tests SHALL be merged with `Tokens` tests
2. IF both `TokenView` and `Tokens` tested identical operations THEN only one test instance SHALL remain
3. WHEN testing slicing functionality THEN tests SHALL verify behavior on `Tokens` directly (not separate class)
4. IF test coverage remains complete THEN no functionality from original tests SHALL be lost

## Non-Functional Requirements

### Performance
- No degradation in test execution time
- No increase in memory usage from token operations

### Maintainability
- All type annotations remain valid (`pyright` passes)
- Code readability improved by removing defensive type checks
- Update path simplified and more direct

### Usability
- Duck typing works naturally (no `isinstance` checks needed)
- Clear method names for update operations

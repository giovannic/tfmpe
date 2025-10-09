# Requirements Document: Bottom-Up Training Algorithm

## Introduction

The bottom-up training algorithm is a multi-round inference strategy that iteratively refines posterior approximations by training local likelihood estimators and global posterior estimators in alternating rounds. This feature brings the proven multi-round training approach from the legacy SFMPE implementation to the new TFMPE framework, enabling more sophisticated hierarchical model training.

## Alignment with Package Vision

This feature supports TFMPE's goal of providing flexible, efficient probabilistic inference tools by enabling iterative refinement of posterior approximations. Bottom-up training represents a key training paradigm from the legacy codebase that improves posterior quality compared to single-pass training, making it essential for production inference workflows.

## Requirements

### Requirement 1: E2E Test Implementation

**User Story:** As a developer, I want an E2E test that validates bottom-up training works correctly, so that I have confidence the implementation is correct and can refactor safely.

#### Acceptance Criteria

1. WHEN the E2E test runs THEN it SHALL reuse the existing hierarchical Gaussian test data from `test_e2e_training.py`
2. WHEN training completes THEN the losses (both train and validation) SHALL be finite for all iterations
3. WHEN comparing training loss across rounds THEN the loss SHOULD generally decrease or remain stable (no catastrophic divergence)
4. WHEN the test completes THEN it SHALL return a trained TFMPE instance

### Requirement 2: Bottom-Up Algorithm Core Logic

**User Story:** As a researcher, I want to train local likelihood estimators followed by global posteriors, so that I can achieve iterative refinement of inference approximations.

#### Acceptance Criteria

1. WHEN round 0 executes THEN the system SHALL train `p(y|theta)` on sampled prior-predictive data
2. WHEN subsequent rounds execute THEN the system SHALL:
   - Decode previous posterior samples to get improved parameter estimates
   - Re-encode for training the local likelihood with new data
   - Train global posterior `p(theta,z|y)` on structured data
3. IF parameters and observations are available THEN the system SHALL handle token restructuring without unnecessary copying

### Requirement 3: Token Encoding and Decoding Between Rounds

**User Story:** As an implementer, I want to encode and decode tokens between rounds, so that I can restructure data for local likelihood vs. global posterior training.

#### Acceptance Criteria

1. WHEN decoding parameter samples THEN the system SHALL use `tokens.decode_keys()` to extract only needed parameters without full reconstruction
2. WHEN re-encoding decoded parameters into new token structure THEN the system SHALL use `Tokens.from_pytree()` with appropriate keys and labeller configuration
3. WHEN the training objective switches (local likelihood ↔ global posterior) THEN the system SHALL correctly map which keys are parameters vs. observations
4. WHEN encoding/decoding THEN the system SHALL be consistent with TFMPE's posterior sampling and log probability methods

### Requirement 4: Integration with Existing TFMPE APIs

**User Story:** As a user, I want bottom-up training to work with TFMPE, `fit_fast`, and existing test infrastructure, so that it fits naturally into the training pipeline.

#### Acceptance Criteria

1. WHEN bottom-up training executes THEN it SHALL use `fit_fast()` internally for each round's training step
2. IF loss computation is needed THEN the system SHALL reuse `cfm_loss()` from `training.py`
3. WHEN the test runs THEN it SHALL integrate with the existing test data generation and assertion patterns
4. WHEN the feature is not jittable THEN the system SHALL clearly document this limitation

## Non-Functional Requirements

### Performance
- Training should not require full dataset duplication between rounds
- Token restructuring should be efficient (leveraging views/slices where possible)

### Usability
- API should be clear about when to use bottom-up vs. single-pass `fit_fast()` training
- Documentation should explain the multi-round workflow and parameter flow

### Maintainability
- Code should follow TFMPE patterns: type annotations, numpy-style docstrings, <80 char lines
- Logic should be testable in isolation from JIT compilation concerns
- Should reuse existing `fit_fast()` and loss functions rather than reimplementing

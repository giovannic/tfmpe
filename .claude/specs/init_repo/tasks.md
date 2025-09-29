# Implementation Plan

## Task Overview

Break repository initialization into atomic, file-specific tasks that create the complete infrastructure without any implementation code. Each task creates or modifies 1-3 related files to establish the foundational structure for TFMPE development.

## Steering Document Compliance

Tasks follow structure.md conventions for package organization, tech.md standards for tooling setup, and package.md alignment for scientific computing focus. All tasks create empty structural placeholders ready for future implementation.

## Atomic Task Requirements
**Each task must meet these criteria for optimal agent execution:**
- **File Scope**: Touches 1-3 related files maximum
- **Time Boxing**: Completable in 15-30 minutes
- **Single Purpose**: One testable outcome per task
- **Specific Files**: Must specify exact files to create/modify
- **Agent-Friendly**: Clear input/output with minimal context switching

## Task Format Guidelines
- Use checkbox format: `- [ ] Task number. Task description`
- **Specify files**: Always include exact file paths to create/modify
- **Include implementation details** as bullet points
- Reference requirements using: `_Requirements: X.Y, Z.A_`
- Prefer task parallelism, avoid unnecessary `Requirements`
- Reference existing code to leverage using: `_Leverage: path/to/file_1, path/to/file_2`
- Focus only on coding tasks (no deployment, user testing, etc.)
- **Avoid broad terms**: No "system", "integration", "complete" in task titles

## Good vs Bad Task Examples
❌ **Bad Examples (Too Broad)**:
- "Implement neural network training system" (affects many files, multiple purposes)
- "Add statistical analysis features" (vague scope, no file specification)
- "Build complete data pipeline" (too large, multiple components)

✅ **Good Examples (Atomic)**:
- "Create DataLoader class in src/data/loader.py with numpy array validation"
- "Add gradient descent optimizer in src/optimization/sgd.py using JAX"
- "Create plot_results function in src/visualization/plots.py with matplotlib"

## Tasks

- [X] 1. Create project configuration in pyproject.toml
  - File: pyproject.toml
  - Define project metadata (name="tfmpe", description, authors, Python >=3.10)
  - Add default dependencies: jax, flax, optax, jaxtyping
  - Add dev optional dependencies: pytest, pyright, pytest-benchmark
  - Add examples optional dependencies: matplotlib, seaborn, hydra-core
  - Configure pytest markers: slow, speed, scale
  - Purpose: Establish dependency management and project metadata
  - _Leverage: sfmpe_legacy/pyproject.toml structure_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [X] 2. Create main package structure in tfmpe/__init__.py
  - File: tfmpe/__init__.py
  - Create empty __init__.py with basic package docstring
  - Add __version__ placeholder and __all__ list
  - Purpose: Establish importable package entry point
  - _Leverage: structure.md import patterns_
  - _Requirements: 2.1, 2.3_

- [X] 3. Create metrics module in tfmpe/metrics/__init__.py
  - File: tfmpe/metrics/__init__.py
  - Create empty __init__.py with module docstring
  - Add placeholder __all__ list for future exports
  - Purpose: Establish metrics module structure
  - _Leverage: structure.md module organization_
  - _Requirements: 2.1, 2.4_

- [X] 4. Create estimators module in tfmpe/estimators/__init__.py
  - File: tfmpe/estimators/__init__.py
  - Create empty __init__.py with module docstring
  - Add placeholder __all__ list for future exports
  - Purpose: Establish estimators module structure
  - _Leverage: structure.md module organization_
  - _Requirements: 2.1, 2.4_

- [X] 5. Create bijectors module in tfmpe/bijectors/__init__.py
  - File: tfmpe/bijectors/__init__.py
  - Create empty __init__.py with module docstring
  - Add placeholder __all__ list for future exports
  - Purpose: Establish bijectors module structure
  - _Leverage: structure.md module organization_
  - _Requirements: 2.1, 2.4_

- [X] 6. Create preprocessing module in tfmpe/preprocessing/__init__.py
  - File: tfmpe/preprocessing/__init__.py
  - Create empty __init__.py with module docstring
  - Add placeholder __all__ list for future exports
  - Purpose: Establish preprocessing module structure
  - _Leverage: structure.md module organization_
  - _Requirements: 2.1, 2.4_

- [X] 7. Create sampling module in tfmpe/sampling/__init__.py
  - File: tfmpe/sampling/__init__.py
  - Create empty __init__.py with module docstring
  - Add placeholder __all__ list for future exports
  - Purpose: Establish sampling module structure
  - _Leverage: structure.md module organization_
  - _Requirements: 2.1, 2.4_

- [X] 8. Create neural network module in tfmpe/nn/__init__.py
  - File: tfmpe/nn/__init__.py
  - Create empty __init__.py with module docstring
  - Add placeholder __all__ list for future exports
  - Purpose: Establish neural network module structure
  - _Leverage: structure.md module organization_
  - _Requirements: 2.1, 2.4_

- [X] 9. Create transformer submodule in tfmpe/nn/transformer/__init__.py
  - File: tfmpe/nn/transformer/__init__.py
  - Create empty __init__.py with module docstring
  - Add placeholder __all__ list for future exports
  - Purpose: Establish transformer submodule structure
  - _Leverage: structure.md module organization, legacy transformer structure_
  - _Requirements: 2.2, 2.4_

- [X] 10. Create test directory structure in test/__init__.py and conftest.py
  - Files: test/__init__.py, test/conftest.py
  - Create empty test package __init__.py
  - Create conftest.py with placeholder fixtures for scientific computing
  - Add sample fixture structure for future JAX/numpy array testing
  - Purpose: Establish test infrastructure foundation
  - _Leverage: sfmpe_legacy/test/conftest.py patterns_
  - _Requirements: 3.1, 3.4_

- [X] 11. Create test module directories matching package structure
  - Files: test/test_metrics/__init__.py, test/test_estimators/__init__.py, test/test_bijectors/__init__.py, test/test_preprocessing/__init__.py, test/test_sampling/__init__.py, test/test_nn/__init__.py
  - Create empty __init__.py in each test module directory
  - Mirror tfmpe/ package structure in test/ directory
  - Purpose: Establish parallel test structure for future implementations
  - _Leverage: structure.md testing organization_
  - _Requirements: 3.3_

- [X] 12. Create README.md with project documentation
  - File: README.md
  - Add TFMPE project description and purpose
  - Include installation instructions for different dependency groups
  - Add development setup section with pytest and pyright commands
  - Include placeholder sections for usage examples and contribution guidelines
  - Purpose: Provide clear project introduction and setup guidance
  - _Leverage: sfmpe_legacy/README.md structure, CLAUDE.md development commands_
  - _Requirements: 5.1_

- [X] 13. Create NEWS.md with changelog structure
  - File: NEWS.md
  - Create changelog format with version headers
  - Add categories: Added, Changed, Fixed, Removed
  - Include initial v0.1.0 placeholder entry
  - Purpose: Establish version history tracking structure
  - _Leverage: standard changelog conventions_
  - _Requirements: 5.2_

- [X] 14. Create mkdocs.yml for documentation generation
  - File: mkdocs.yml
  - Configure basic mkdocs setup with site metadata
  - Add placeholder nav structure for API documentation
  - Configure mkdocstrings plugin for automatic API docs
  - Purpose: Enable documentation generation infrastructure
  - _Leverage: legacy documentation patterns_
  - _Requirements: 5.3_

- [X] 15. Create basic docs directory structure
  - Files: docs/index.md, docs/api.md
  - Create docs/ directory with placeholder content
  - Add index.md with project overview
  - Add api.md placeholder for future API documentation
  - Purpose: Establish documentation content structure
  - _Leverage: mkdocs conventions_
  - _Requirements: 5.4_

- [X] 16. Create GitHub Actions CI workflow in .github/workflows/ci.yml
  - File: .github/workflows/ci.yml
  - Configure Python matrix testing (3.10, 3.11)
  - Add job for testing default dependencies installation
  - Add job for testing dev dependencies and running pytest
  - Add job for running pyright type checking
  - Add job for testing documentation builds with mkdocs
  - Purpose: Automate quality checks and testing
  - _Leverage: sfmpe_legacy/.github/workflows/ci.yaml patterns_
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [X] 17. Update .gitignore for additional Python project patterns
  - File: .gitignore (modify existing)
  - Add missing patterns: .pytest_cache/, site/ (mkdocs), build/, dist/
  - Add development tool ignores: .vscode/, .idea/
  - Keep existing patterns and add additional ones needed
  - Purpose: Ensure comprehensive ignore coverage for development
  - _Leverage: existing .gitignore_
  - _Requirements: None_

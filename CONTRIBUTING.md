# Contributing to TFMPE

Thank you for your interest in contributing to TFMPE! This document
outlines the development workflow and contribution standards.

## Contribution Standards

All contributions must meet the following requirements:

- **Tests**: All tests must pass (`python -m pytest test/`)
- **Type Checking**: Code must pass static type analysis (`pyright`)
- **Code Quality**: Follow the project's coding standards (see CLAUDE.md)
- **Documentation**: Include numpy-style docstrings for public APIs

## Development Workflow (Optional)

While not required, this project provides a structured spec-driven
development workflow that can help organize complex feature development.
Using this workflow is completely optional - feel free to contribute
using your preferred development process.

### Spec-Driven Development

For contributors who want to use the structured workflow, the project
includes a four-phase process:

1. **Requirements Phase** - Define functional and non-functional
   requirements
2. **Design Phase** - Create technical design with architecture diagrams
3. **Tasks Phase** - Break design into atomic, executable tasks
4. **Implementation Phase** - Execute tasks systematically with validation

#### Creating a New Feature Spec

Start the workflow with:

```bash
/create_spec <feature-name> [description]
```

This guides you through creating:
- Requirements document (`.claude/specs/{feature}/requirements.md`)
- Technical design (`.claude/specs/{feature}/design.md`)
- Task breakdown (`.claude/specs/{feature}/tasks.md`)

Each phase requires explicit approval before proceeding.

#### Executing Spec Tasks

Once tasks are approved, implement them using:

```bash
/execute_spec [feature_name] [task_id]
```

This command:
- Creates an isolated git worktree for the task
- Executes the specific task implementation
- Runs required tests and validation
- Merges changes upon approval

#### Workflow Principles

- **Structured Development**: Follow phases sequentially
- **User Approval Required**: Each phase needs explicit approval
- **Atomic Implementation**: One focused task at a time
- **Requirement Traceability**: All tasks reference specific requirements
- **Test-Driven Focus**: Write tests before implementation

For detailed workflow instructions, see `.claude/commands/create_spec.md`
and `.claude/commands/execute_spec.md`.

## Standard Contribution Process

If you prefer not to use the spec-driven workflow:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure tests pass and type checking succeeds
5. Submit a pull request

## Development Setup

See README.md for installation and development environment setup
instructions.

## Questions?

Open an issue for questions about contributing or development workflow.
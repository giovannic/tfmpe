# TFMPE

**Tokenised Flow Matching for Posterior Estimation**

TFMPE is a Python package for efficient posterior estimation using structured parameter representations and flow matching techniques. It provides modular tools for parameter estimation, data preprocessing, and neural network architectures optimized for scientific computing applications.

## Installation

### Basic Installation

Install the core package with default dependencies:

```bash
pip install -e .
```

### Development Installation

For development work including testing and type checking:

```bash
pip install -e .[dev]
```

### Examples and Plotting

For running examples with visualization capabilities:

```bash
pip install -e .[examples]
```

### All Dependencies

Install everything for full development and examples:

```bash
pip install -e .[dev,examples]
```

## Development Setup

### Prerequisites

- Python ≥ 3.10

### Testing

The project uses pytest with custom markers for different test categories:

```bash
# Run standard test suite (fast tests only)
python -m pytest test/

# Run all tests including slow ones
python -m pytest test/ -m "slow or not slow"

# Run only slow tests
python -m pytest test/ -m "slow"

# Run speed benchmarks
python -m pytest test/ -m "speed"

# Run scale benchmarks
python -m pytest test/ -m "scale"

# Run all tests (including benchmarks)
python -m pytest test/ -m ""
```

### Type Checking

Static type analysis: `pyright`

### Documentation

Build documentation locally:

```bash
pip install -e .[docs]
mkdocs build
```

Serve documentation with live reload:

```bash
mkdocs serve
```

## Package Structure

```
tfmpe/
├── metrics/          # Metrics for analysis of parameter inference
├── estimators/       # Parameter estimators
├── bijectors/        # Bijectors for structured parameter sets
├── preprocessing/    # Pipelines for processing datasets
├── sampling/         # Sampling algorithms for training estimators
└── nn/              # Neural networks
    └── transformer/  # Transformer model architectures
```

## Usage Examples

*Examples and usage documentation will be added as the package develops.*

## Contributing

*Contribution guidelines will be added as the project matures.*

## License

*License information to be determined.*
# Contributing to TorchONNX

Contributions are welcome from the community. Whether fixing bugs, adding features, improving documentation, or sharing ideas, all contributions are appreciated.

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/torchonnx.git
   cd torchonnx
   ```

2. **Install in development mode with dev dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Code Quality Standards

- **Linting:** Code must pass `ruff check` and `ruff format --check`
- **Type checking:** Code must pass `mypy` static analysis
- **Testing:** All tests must pass with `pytest tests/`
- **Coverage:** New code should include tests to maintain coverage

Run quality checks locally:
```bash
# Lint check
ruff check src/torchonnx tests
ruff format --check src/torchonnx tests

# Type check
mypy .

# Run tests
pytest tests/
```

## Pull Request Workflow

**Note:** Direct pushes to the `main` branch are restricted. All changes must go through Pull Requests.

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write code following existing patterns
   - Add tests for new functionality
   - Update documentation if needed
   - Ensure all quality checks pass

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request:**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Provide a clear description of your changes
   - Reference any related issues

6. **Address review feedback:**
   - Reviewers may request changes
   - Push additional commits to the same branch
   - The PR will update automatically

## Testing

### Testing Your Changes

Run the full test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src/torchonnx --cov-report=term-missing
```

Run specific test files:
```bash
pytest tests/test_units/test_torchonnx/test_pipeline.py -v
```

## Code Style Guidelines

- Follow PEP 8 conventions (enforced by `ruff`)
- Use type hints for all function signatures
- Write docstrings using reStructuredText format
- Keep functions focused and single-purpose
- Prefer explicit over implicit

## Reporting Issues

When reporting bugs or requesting features:
- Check if the issue already exists
- Provide a minimal reproducible example
- Include ONNX model details and error messages
- Specify your environment (Python version, PyTorch version, OS)

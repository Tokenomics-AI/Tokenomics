# Contributing to Tokenomics Platform

Thank you for your interest in contributing to the Tokenomics Platform! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/tokenomics-platform.git
   cd tokenomics-platform
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/originalowner/tokenomics-platform.git
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Install Development Dependencies

```bash
pip install pytest pytest-cov black flake8 mypy
```

### 4. Configure Environment

```bash
cp env.template .env
# Edit .env with your API keys for testing
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, readable code
- Follow the code style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run all tests
pytest tests/ -v

# Check code style
black --check tokenomics/
flake8 tokenomics/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Add: Description of your changes"
```

**Commit Message Guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Fix bug" not "Fixes bug")
- Keep first line under 50 characters
- Add detailed description if needed

**Commit Types:**
- `Add:` - New feature
- `Fix:` - Bug fix
- `Update:` - Update existing feature
- `Refactor:` - Code refactoring
- `Docs:` - Documentation changes
- `Test:` - Test additions/changes
- `Style:` - Code style changes

### 5. Keep Your Branch Updated

```bash
git fetch upstream
git rebase upstream/main
```

### 6. Push Your Changes

```bash
git push origin feature/your-feature-name
```

## Code Style

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line Length:** 100 characters (soft limit)
- **Indentation:** 4 spaces
- **Quotes:** Double quotes for strings
- **Imports:** Sorted and grouped (stdlib, third-party, local)

### Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
black tokenomics/
black tests/
```

### Linting

We use [flake8](https://flake8.pycqa.org/) for linting:

```bash
flake8 tokenomics/
```

### Type Hints

We encourage type hints for better code clarity:

```python
def query(self, query: str, token_budget: Optional[int] = None) -> Dict[str, Any]:
    ...
```

## Testing

### Writing Tests

- **Unit Tests:** Test individual functions/classes in isolation
- **Integration Tests:** Test component interactions
- **Diagnostic Tests:** Comprehensive platform validation

### Test Structure

```python
def test_feature_name():
    """Test description."""
    # Arrange
    platform = TokenomicsPlatform(config)
    
    # Act
    result = platform.query("test query")
    
    # Assert
    assert result["response"] is not None
    assert result["tokens_used"] > 0
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_memory.py -v

# Run with coverage
pytest tests/ --cov=tokenomics --cov-report=html

# Run diagnostic tests
python tests/diagnostic/extensive_diagnostic_test.py
```

### Test Coverage

We aim for >80% test coverage. Check coverage with:

```bash
pytest tests/ --cov=tokenomics --cov-report=term-missing
```

## Documentation

### Code Documentation

- **Docstrings:** Use Google-style docstrings
- **Comments:** Explain "why" not "what"
- **Type Hints:** Include type hints for function signatures

Example:

```python
def query(
    self,
    query: str,
    token_budget: Optional[int] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Process a query through the Tokenomics platform.
    
    Args:
        query: The user query string
        token_budget: Optional token budget limit
        use_cache: Whether to use caching
        
    Returns:
        Dictionary containing response, tokens_used, and metadata
        
    Raises:
        ValueError: If query is empty
    """
    ...
```

### Documentation Files

- **Architecture Docs:** `docs/architecture/`
- **Guides:** `docs/guides/`
- **Testing Docs:** `docs/testing/`
- **Results:** `docs/results/`

When adding new features:
1. Update relevant documentation
2. Add examples if applicable
3. Update architecture diagrams if needed

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass:**
   ```bash
   pytest tests/ -v
   ```

2. **Check code style:**
   ```bash
   black --check tokenomics/
   flake8 tokenomics/
   ```

3. **Update documentation** if needed

4. **Add changelog entry** if applicable

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Review Process

1. **Automated Checks:** CI will run tests and linting
2. **Code Review:** At least one maintainer will review
3. **Feedback:** Address any feedback or requested changes
4. **Approval:** Once approved, your PR will be merged

## Project Structure

```
tokenomics-platform/
├── tokenomics/          # Core package
│   ├── memory/         # Memory layer
│   ├── orchestrator/   # Token orchestrator
│   ├── bandit/         # Bandit optimizer
│   └── ...
├── tests/              # Test suite
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   ├── diagnostic/    # Diagnostic tests
│   └── benchmarks/    # Benchmark tests
├── docs/              # Documentation
├── examples/          # Usage examples
└── scripts/           # Utility scripts
```

## Areas for Contribution

### High Priority

- Performance optimizations
- Additional LLM provider support
- Enhanced compression techniques
- Quality scoring improvements
- Documentation improvements

### Medium Priority

- Additional test coverage
- Benchmark improvements
- Configuration enhancements
- Monitoring and observability

### Low Priority

- Code refactoring
- Style improvements
- Example additions

## Getting Help

- **Questions:** Open a GitHub Discussion
- **Bugs:** Open a GitHub Issue
- **Feature Requests:** Open a GitHub Issue
- **Security Issues:** Email security@example.com

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in project documentation

Thank you for contributing to the Tokenomics Platform! 🎉

# UV Quick Reference for RAG Workshop

## Initial Setup

```bash
# Create virtual environment
uv venv

# Activate virtual environment (Linux/macOS)
source .venv/bin/activate

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install all dependencies
uv pip install -e .
```

## Managing Dependencies

```bash
# Install a new package
uv pip install package-name

# Install a specific version
uv pip install package-name==1.2.3

# Update all packages
uv pip install --upgrade -e .

# Update a specific package
uv pip install --upgrade package-name

# Uninstall a package
uv pip uninstall package-name

# List installed packages
uv pip list

# Show package details
uv pip show package-name

# Check for outdated packages
uv pip list --outdated
```

## Freezing Dependencies

```bash
# Generate requirements.txt from current environment
uv pip freeze > requirements.txt

# Install from requirements.txt
uv pip install -r requirements.txt
```

## Development Dependencies

```bash
# Install with development dependencies
uv pip install -e ".[dev]"
```

## PyTorch Specific

```bash
# Install PyTorch with CPU support
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch with CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch with CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Troubleshooting

```bash
# Clear UV cache
uv cache clean

# Reinstall all packages
uv pip install --force-reinstall -e .

# Check UV version
uv --version

# Get help
uv --help
uv pip --help
```

## Virtual Environment Management

```bash
# Create virtual environment with specific Python version
uv venv --python 3.10
uv venv --python 3.11

# Remove virtual environment
rm -rf .venv  # Linux/macOS
rmdir /s .venv  # Windows

# Deactivate virtual environment
deactivate
```

## Why Use UV?

- **🚀 Fast**: 10-100x faster than pip
- **🎯 Reliable**: Deterministic dependency resolution
- **💾 Efficient**: Shared package cache
- **🔒 Safe**: Built in Rust for reliability
- **🔄 Compatible**: Drop-in replacement for pip

## Common Workflow

```bash
# 1. Start your work session
source .venv/bin/activate

# 2. Update dependencies if needed
uv pip install --upgrade -e .

# 3. Start Jupyter
jupyter notebook

# 4. When done
deactivate
```

## Updating pyproject.toml

After editing `pyproject.toml` dependencies:

```bash
# Sync environment with updated dependencies
uv pip install -e .

# Or force reinstall everything
uv pip install --force-reinstall -e .
```

## Tips

- Use `uv pip` as a drop-in replacement for `pip`
- All `pip` commands work with `uv pip`
- UV automatically uses the active virtual environment
- UV caches packages globally for reuse across projects
- UV resolves dependencies much faster than pip

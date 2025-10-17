#!/bin/bash

# RAG Workshop Setup Script
# This script automates the environment setup using UV

set -e  # Exit on error

echo "=================================="
echo "RAG Workshop Environment Setup"
echo "=================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed."
    echo "📦 Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the shell profile to make uv available
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    elif [ -f "$HOME/.zshrc" ]; then
        source "$HOME/.zshrc"
    fi
    
    echo "✅ UV installed successfully"
else
    echo "✅ UV is already installed"
fi

echo ""
echo " Installing dependencies from pyproject.toml..."
echo "   (This will create a virtual environment and install all packages)"
uv sync

echo ""
echo "⚙️  Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo "⚠️  Please edit .env and add your Azure OpenAI credentials"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "=================================="
echo "✨ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Edit .env with your credentials:"
echo "   nano .env"
echo ""
echo "3. Start Jupyter:"
echo "   jupyter notebook"
echo ""
echo "4. Navigate to RAG_hf_v4/ to access the demos"
echo ""

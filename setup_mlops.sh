#!/bin/bash

# TrendFlow-AI MLOps Setup Script
# This script creates the necessary directory structure and files

echo "🚀 Setting up TrendFlow-AI MLOps Pipeline..."

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p .github/workflows
mkdir -p tests

# Create __init__.py for tests package
echo "📝 Creating Python package files..."
touch tests/__init__.py

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Environment
.env
EOF
fi

echo "✅ Directory structure created!"
echo ""
echo "📋 Next steps:"
echo "1. Copy the content from Claude's artifacts into these files:"
echo "   - Dockerfile"
echo "   - .github/workflows/main.yml"
echo "   - tests/test_basic.py"
echo "   - api.py"
echo "   - requirements.txt"
echo "   - .dockerignore"
echo "   - pytest.ini"
echo ""
echo "2. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Run tests:"
echo "   pytest tests/ -v"
echo ""
echo "4. Build Docker image:"
echo "   docker build -t trendflow-ai:latest ."
echo ""
echo "5. Push to GitHub:"
echo "   git add ."
echo "   git commit -m 'feat: Add MLOps pipeline'"
echo "   git push origin main"
echo ""
echo "✨ Happy coding!"

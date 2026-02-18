# Activate virtual environment
source .venv/bin/activate

# Clean old builds + rebuild
rm -rf dist/*
python -m build

# Upload
python -m twine upload dist/*
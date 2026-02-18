# Activate virtual environment
./.venv/Script/Activate

# Clean old builds + rebuild
Remove-Item dist/* -ErrorAction SilentlyContinue
python -m build

# Upload
python -m twine upload dist/*
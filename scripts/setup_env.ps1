python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/check_project.py
Write-Host "If self-check passed, run: python demo/app.py"

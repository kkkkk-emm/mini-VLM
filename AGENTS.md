# AGENTS.md

## Python Environment

This project must use the Conda environment `minimind`.

Do not run bare `python`, `pip`, or `pytest` commands in this repository. Use:

```powershell
conda run -n minimind python ...
```

Examples:

```powershell
conda run -n minimind python -c "import sys; print(sys.executable)"
conda run -n minimind python -m pytest tests -q
conda run -n minimind python grpo.py --help
```

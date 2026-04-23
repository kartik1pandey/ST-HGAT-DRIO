#!/bin/bash
# ST-HGAT-DRIO WSL Runner
# Uses the Windows venv Python (accessible via /mnt/d/...) which has all packages.
# Runs under Linux so torch.compile + Triton backend are available.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu -- bash /mnt/d/CD_labs/CD_project/run_wsl.sh
#
# Or from inside WSL:
#   bash /mnt/d/CD_labs/CD_project/run_wsl.sh

set -e
PROJ="/mnt/d/CD_labs/CD_project"
PYTHON="$PROJ/venv/Scripts/python.exe"

cd "$PROJ"
export PYTHONPATH="$PROJ"
export PYTHONIOENCODING="utf-8"

echo "============================================================"
echo "  ST-HGAT-DRIO  |  WSL Linux Runner"
echo "  Python: $($PYTHON --version)"
echo "  torch.compile: checking..."
echo "============================================================"

# Check torch.compile availability under Linux
$PYTHON -c "
import torch, sys
print(f'  PyTorch: {torch.__version__}')
print(f'  Platform: {sys.platform}')
try:
    import torch._dynamo
    print('  torch.compile: AVAILABLE')
except Exception as e:
    print(f'  torch.compile: {e}')
"

echo ""
echo "  Running run_submission.py under Linux..."
echo ""

$PYTHON run_submission.py

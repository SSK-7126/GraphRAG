<#
.SYNOPSIS
Sets up portable optional acceleration for this project.

.DESCRIPTION
Installs Ray and Dask on every supported machine. On Windows with an NVIDIA
driver, it additionally installs PyTorch's CUDA 12.8 wheel into the project's
`.venv`. Computers without a supported NVIDIA GPU keep the CPU path.
#>
[CmdletBinding()]
param(
    [switch]$SkipCuda,
    [switch]$SkipRay,
    [switch]$SkipDask
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found. Run 'uv sync' from $projectRoot first."
}

Push-Location $projectRoot
try {
    if (-not $SkipRay) { uv pip install --python $python "ray>=2.40" }
    if (-not $SkipDask) { uv pip install --python $python "dask>=2024.12" }

    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($SkipCuda -or -not $nvidia) {
        Write-Host "CUDA skipped: no NVIDIA driver was detected or -SkipCuda was specified."
    } else {
        & $nvidia.Source --query-gpu=name,driver_version,memory.total --format=csv,noheader
        # CUDA 12.8 is supplied by the PyTorch wheel; a separately installed
        # CUDA Toolkit is not required. CPU-only PyTorch is replaced.
        uv pip install --python $python --upgrade --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
        uv pip install --python $python "sentence-transformers>=6.1.0"
    }
    & $python (Join-Path $projectRoot "run.py") --hardware
} finally {
    Pop-Location
}

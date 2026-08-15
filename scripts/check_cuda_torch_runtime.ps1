param(
  [string]$Target = "D:\vla_torch_cuda_pkgs"
)

$ErrorActionPreference = "Stop"

$Project = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$CheckScript = Join-Path $Project "scripts\check_runtime_capability.py"

if (-not (Test-Path $Target)) {
  throw "CUDA Torch target directory not found: $Target"
}

$env:VLA_TORCH_PACKAGE_DIR = $Target
& $Python $CheckScript

if ($LASTEXITCODE -ne 0) {
  throw "runtime capability check failed with exit code $LASTEXITCODE"
}

param(
  [string]$Target = "D:\vla_torch_cuda_pkgs",
  [ValidateSet("cu124", "cu121")]
  [string]$Cuda = "cu124",
  [switch]$Install
)

$ErrorActionPreference = "Stop"

$Project = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$CheckScript = Join-Path $Project "scripts\check_runtime_capability.py"

if ($Cuda -eq "cu124") {
  $Torch = "torch==2.6.0+cu124"
  $TorchVision = "torchvision==0.21.0+cu124"
} else {
  $Torch = "torch==2.5.1+cu121"
  $TorchVision = "torchvision==0.20.1+cu121"
}

$IndexUrl = "https://download.pytorch.org/whl/$Cuda"
$PyPiUrl = "https://pypi.org/simple"

Write-Host "runtime_cuda_torch_setup_v1"
Write-Host "Project: $Project"
Write-Host "Python:  $Python"
Write-Host "Target:  $Target"
Write-Host "PyPI index: $PyPiUrl"
Write-Host "CUDA wheel extra index: $IndexUrl"
Write-Host "Packages: $Torch, $TorchVision"

if (-not (Test-Path $Python)) {
  throw "Python executable not found: $Python"
}

if (-not $Install) {
  Write-Host ""
  Write-Host "Dry run only. To install CUDA Torch into the external target directory, run:"
  Write-Host ('& "' + $PSCommandPath + '" -Target "' + $Target + '" -Cuda ' + $Cuda + ' -Install')
  Write-Host ""
  Write-Host "After installation, test that directory with:"
  Write-Host ('$env:VLA_TORCH_PACKAGE_DIR="' + $Target + '"; & "' + $Python + '" "' + $CheckScript + '"')
  exit 0
}

New-Item -ItemType Directory -Force -Path $Target | Out-Null

& $Python -m pip install `
  --target $Target `
  --upgrade `
  --no-cache-dir `
  --index-url $PyPiUrl `
  --extra-index-url $IndexUrl `
  $Torch `
  $TorchVision

if ($LASTEXITCODE -ne 0) {
  throw "pip install failed with exit code $LASTEXITCODE"
}

$env:VLA_TORCH_PACKAGE_DIR = $Target
& $Python $CheckScript

if ($LASTEXITCODE -ne 0) {
  throw "runtime capability check failed with exit code $LASTEXITCODE"
}

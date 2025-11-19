# Usage: run in project root: `powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1`
# Creates venv `.venv`, installs CUDA PyTorch (default cu121) and then the rest of requirements.txt.

param(
    [string]$PythonExe = "C:\Users\haomi\ucl_env\Scripts\python.exe",
    [string]$VenvPath = ".venv",
    [string]$TorchVersion = "2.2.2",
    [string]$VisionVersion = "0.17.2",
    [string]$AudioVersion = "2.2.2",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu121",
    [switch]$SkipTorch
)

function Write-Info($msg) { Write-Host "[INFO]" $msg -ForegroundColor Cyan }
function Write-Err($msg) { Write-Host "[ERROR]" $msg -ForegroundColor Red }

if (-Not (Test-Path "requirements.txt")) {
    Write-Err "requirements.txt not found in current directory."
    exit 1
}

Write-Info "Creating virtual environment at $VenvPath ..."
& $PythonExe -m venv $VenvPath
if ($LASTEXITCODE -ne 0) {
    Write-Err "venv creation failed."
    exit 1
}

$venvPython = Join-Path $VenvPath "Scripts/python.exe"

Write-Info "Upgrading pip ..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Err "pip upgrade failed."
    exit 1
}

if (-Not $SkipTorch) {
    Write-Info "Installing CUDA PyTorch from $TorchIndexUrl ..."
    & $venvPython -m pip install --index-url $TorchIndexUrl `
        "torch==$TorchVersion+cu121" `
        "torchvision==$VisionVersion+cu121" `
        "torchaudio==$AudioVersion+cu121"
    if ($LASTEXITCODE -ne 0) {
        Write-Err "PyTorch CUDA install failed."
        exit 1
    }
}

Write-Info "Installing remaining requirements ..."
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Err "Dependency install failed."
    exit 1
}

Write-Info 'Done. Activate with: .\.venv\Scripts\activate'

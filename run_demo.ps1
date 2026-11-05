$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Chua co .venv. Xem README.md de cai dat dependencies."
}
Set-Location -LiteralPath $projectRoot
& $pythonExe -m uvicorn diabetes_demo.api:app --host 127.0.0.1 --port 8000
exit $LASTEXITCODE

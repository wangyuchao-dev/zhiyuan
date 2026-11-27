# 只跑测试（不联网、不需要 API Key）
#   powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1 -Filter tests/test_data_pipeline.py
param([string]$Filter = "")

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

if ($Filter) {
    & $py -m pytest $Filter -v
} else {
    & $py -m pytest --cov --cov-report=term-missing
}

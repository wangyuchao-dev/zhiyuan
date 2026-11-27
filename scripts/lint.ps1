# 只跑静态检查与自动修复（改完代码随手跑）
#   powershell -ExecutionPolicy Bypass -File .\scripts\lint.ps1
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "ruff check --fix ..." -ForegroundColor Cyan
& $py -m ruff check . --fix

Write-Host "ruff format ..." -ForegroundColor Cyan
& $py -m ruff format .

Write-Host "done" -ForegroundColor Green

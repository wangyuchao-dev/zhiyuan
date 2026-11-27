# 本地跑与 CI 完全相同的检查（提交前跑一次，避免推上去才发现红）
#   powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$failed = 0

Write-Host "[1/3] ruff check ..." -ForegroundColor Cyan
& $py -m ruff check .
if ($LASTEXITCODE -ne 0) { $failed = 1 }

Write-Host "[2/3] ruff format --check ..." -ForegroundColor Cyan
& $py -m ruff format --check .
if ($LASTEXITCODE -ne 0) { $failed = 1 }

Write-Host "[3/3] pytest --cov ..." -ForegroundColor Cyan
& $py -m pytest --cov --cov-report=term-missing
if ($LASTEXITCODE -ne 0) { $failed = 1 }

if ($failed -eq 0) {
    Write-Host "OK - 与 CI 同款检查全部通过" -ForegroundColor Green
} else {
    Write-Host "FAILED - 上面有失败的步骤" -ForegroundColor Red
    exit 1
}

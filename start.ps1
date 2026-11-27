# 启动知渊（Windows PowerShell）
#   双击 start.bat 即可，或右键「使用 PowerShell 运行」本文件
# 做的事：读 .env → 设置环境变量 → 启动 uvicorn（127.0.0.1:8000）

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "[错误] 找不到虚拟环境：$py" -ForegroundColor Red
    Write-Host "请先执行：uv venv --python 3.12 .venv" -ForegroundColor Yellow
    Write-Host "然后：uv pip install --python .venv\Scripts\python.exe -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# ---- 读 .env（KEY=VALUE，# 开头为注释）----
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line -match '^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            $k = $matches[1]
            $v = $matches[2].Trim().Trim('"').Trim("'")
            if ($v) { Set-Item -Path "Env:$k" -Value $v }
        }
    }
    Write-Host "[配置] 已载入 .env" -ForegroundColor Green
} else {
    Write-Host "[提示] 没有 .env，先复制 .env.example 为 .env 并填入 DASHSCOPE_API_KEY" -ForegroundColor Yellow
}

if (-not $env:DASHSCOPE_API_KEY) {
    Write-Host "[警告] DASHSCOPE_API_KEY 为空：服务能启动，但对话与建库会返回 503" -ForegroundColor Yellow
}

Write-Host "[启动] http://127.0.0.1:8000  （Ctrl+C 停止）" -ForegroundColor Cyan
& $py -m uvicorn api.main:app --host 127.0.0.1 --port 8000

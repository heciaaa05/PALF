param(
    [int]$Port = 8501,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " PALF Demo Launcher (v1.0-demo — Mốc 30/11)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[!] Chưa tìm thấy môi trường ảo .venv tại $PythonExe" -ForegroundColor Yellow
    Write-Host "    Hãy chạy: python -m venv .venv; .\.venv\Scripts\pip.exe install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

$BaselineDir = Join-Path $RepoRoot "experiments\results\baseline\jena"
$PalfDir = Join-Path $RepoRoot "experiments\results\palf\jena"
$CompDir = Join-Path $RepoRoot "experiments\analysis\comparison"

if (-not (Test-Path $BaselineDir)) {
    Write-Host "[!] Lưu ý: Chưa có thư mục kết quả thô Baseline '$BaselineDir' (đã .gitignore)." -ForegroundColor Yellow
    Write-Host "    -> Giao diện vẫn hiển thị đầy đủ bảng kết quả 10 seed GRU & C1-C3 từ '$CompDir'" -ForegroundColor Green
    Write-Host "       và toàn bộ 59 lần chạy PALF từ '$PalfDir'." -ForegroundColor Green
    Write-Host "    -> Để khôi phục thêm kết quả thô Baseline A0, chạy lệnh một dòng (xem experiments/RUNS.md mục 5):" -ForegroundColor Yellow
    Write-Host "       Expand-Archive -Path C:\Users\Admin\Downloads\PALF_A0_baseline_backup_all4backbones.zip -DestinationPath . -Force" -ForegroundColor White
} else {
    Write-Host "[OK] Đã tìm thấy thư mục kết quả Baseline: $BaselineDir" -ForegroundColor Green
}

if (-not (Test-Path $PalfDir)) {
    Write-Host "[!] Cảnh báo: Thiếu thư mục kết quả PALF '$PalfDir'. Hãy kiểm tra lại bản clone hoặc giải nén bản sao lưu." -ForegroundColor Yellow
} else {
    Write-Host "[OK] Đã tìm thấy thư mục kết quả PALF: $PalfDir" -ForegroundColor Green
}

$StreamlitArgs = @("-m", "streamlit", "run", "app\app.py", "--server.port", "$Port")
if ($Headless) {
    $StreamlitArgs += @("--server.headless", "true")
}

Write-Host "[*] Đang mở ứng dụng Streamlit tại http://localhost:$Port ..." -ForegroundColor Cyan
& $PythonExe @StreamlitArgs

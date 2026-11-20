param(
    [int]$Port = 8501,
    [switch]$Headless,
    [switch]$SkipTests,
    [switch]$SetupOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " PALF — Cài đặt tự động, Kiểm thử & Khởi chạy Demo" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PipExe = Join-Path $RepoRoot ".venv\Scripts\pip.exe"

# 1. Tự động tạo .venv và cài đặt toàn bộ thư viện nếu chưa có
if (-not (Test-Path $PythonExe)) {
    Write-Host "[1/4] Chưa có môi trường ảo .venv. Đang khởi tạo .venv..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[LỖI] Không thể tạo .venv. Hãy kiểm tra lại Python (>= 3.10) đã có trong PATH chưa." -ForegroundColor Red
        exit 1
    }
    Write-Host "[1/4] Đang cài đặt toàn bộ thư viện từ requirements.txt..." -ForegroundColor Cyan
    & $PythonExe -m pip install --upgrade pip
    & $PipExe install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[LỖI] Cài đặt thư viện thất bại." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Đã tạo .venv và cài đặt đầy đủ thư viện!" -ForegroundColor Green
} else {
    # Kiểm tra nhanh xem gói palf và streamlit đã được cài trong .venv chưa
    & $PythonExe -c "import palf, streamlit, torch, pandas, matplotlib" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[1/4] Phát hiện thiếu thư viện trong .venv. Đang cài đặt bổ sung từ requirements.txt..." -ForegroundColor Yellow
        & $PipExe install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[LỖI] Cài đặt thư viện thất bại." -ForegroundColor Red
            exit 1
        }
    }
    Write-Host "[1/4] [OK] Môi trường ảo .venv và toàn bộ thư viện đã sẵn sàng." -ForegroundColor Green
}

# 2. Kiểm tra và tự động khôi phục dữ liệu Baseline A0 nếu có file zip sao lưu
$BaselineDir = Join-Path $RepoRoot "experiments\results\baseline\jena"
$PalfDir = Join-Path $RepoRoot "experiments\results\palf\jena"
$CompDir = Join-Path $RepoRoot "experiments\analysis\comparison"

if (-not (Test-Path $BaselineDir)) {
    $ZipCandidates = @(
        (Join-Path $RepoRoot "PALF_A0_baseline_backup_all4backbones.zip"),
        (Join-Path $HOME "Downloads\PALF_A0_baseline_backup_all4backbones.zip")
    )
    $FoundZip = $ZipCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

    if ($FoundZip) {
        Write-Host "[2/4] Phát hiện bản sao lưu '$FoundZip'. Đang tự động giải nén..." -ForegroundColor Cyan
        Expand-Archive -Path $FoundZip -DestinationPath $RepoRoot -Force
        Write-Host "[2/4] [OK] Đã giải nén dữ liệu Baseline A0 vào '$BaselineDir'." -ForegroundColor Green
    } else {
        Write-Host "[2/4] [Lưu ý] Chưa có thư mục kết quả thô Baseline '$BaselineDir' (đã .gitignore)." -ForegroundColor Yellow
        Write-Host "      -> Ứng dụng vẫn hiển thị đầy đủ bảng kết quả 10 seed GRU & C1-C3 từ '$CompDir'" -ForegroundColor Green
        Write-Host "         và toàn bộ 70 lần chạy PALF từ '$PalfDir'." -ForegroundColor Green
        Write-Host "      -> (Tùy chọn) Nếu muốn khôi phục thêm đường cong từng epoch của Baseline A0," -ForegroundColor Yellow
        Write-Host "         tải file zip từ Google Drive (trong experiments/RUNS.md mục 5) về Downloads rồi chạy lại script này." -ForegroundColor Yellow
    }
} else {
    Write-Host "[2/4] [OK] Đã có đầy đủ thư mục kết quả Baseline & PALF." -ForegroundColor Green
}

# 3. Chạy kiểm thử tự động pytest để xác nhận toàn bộ chương trình hoạt động tốt
if (-not $SkipTests) {
    Write-Host "[3/4] Đang chạy bộ kiểm thử tự động (pytest -q)..." -ForegroundColor Cyan
    & $PythonExe -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[LỖI] Một số bài kiểm thử không đạt. Vui lòng kiểm tra thông báo phía trên." -ForegroundColor Red
        exit 1
    }
    Write-Host "[3/4] [OK] Toàn bộ bài kiểm thử đều đạt (PASS)!" -ForegroundColor Green
} else {
    Write-Host "[3/4] Bỏ qua bước chạy pytest (-SkipTests)." -ForegroundColor DarkGray
}

if ($SetupOnly) {
    Write-Host "[4/4] Đã hoàn tất cài đặt và kiểm thử (-SetupOnly)." -ForegroundColor Green
    exit 0
}

# 4. Mở ứng dụng Streamlit Demo
$StreamlitArgs = @("-m", "streamlit", "run", "app\app.py", "--server.port", "$Port")
if ($Headless) {
    $StreamlitArgs += @("--server.headless", "true")
}

Write-Host "[4/4] Đang mở giao diện PALF Demo tại http://localhost:$Port ..." -ForegroundColor Cyan
& $PythonExe @StreamlitArgs

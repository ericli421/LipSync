$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$services = @()

function Wait-ForService([string]$Url, $Process, [string]$Name) {
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) { throw "$Name stopped. Check the logs in .tools\startup." }
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) { return }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    throw "$Name did not become ready. Check the logs in .tools\startup."
}

try {
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        $ffmpegTool = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg*" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($ffmpegTool) { $env:PATH = $ffmpegTool.DirectoryName + ';' + $env:PATH }
    }
    foreach ($port in @(8000, 3000)) {
        if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
            throw "Port $port is already in use. Close the existing server before starting LipSync."
        }
    }
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    $python = Join-Path $projectRoot '.venv-local\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        throw 'Python environment missing. Follow the first-time setup in docs/local-android-demo.md.'
    }
    # Refresh dependencies after pulls, including newly added packages.
    $backendStamp = Join-Path $projectRoot '.venv-local\lipsync-requirements.sha256'
    $backendHash = (Get-FileHash 'processor\requirements.txt' -Algorithm SHA256).Hash
    if (-not (Test-Path $backendStamp) -or (Get-Content $backendStamp -Raw).Trim() -ne $backendHash) {
        Write-Host 'Updating backend dependencies...'
        & $python -m pip install -r processor/requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed.' }
        Set-Content -LiteralPath $backendStamp -Value $backendHash
    }
    & $python -c 'import processor.server'
    if ($LASTEXITCODE -ne 0) { throw 'Backend import failed. Check the error above and reinstall processor/requirements.txt.' }
    $torchStatus = & $python scripts/check_torch.py
    $gpu = $null
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
    }
    if ($torchStatus -like 'gpu*') {
        Write-Host "Separation will run on the GPU: $torchStatus" -ForegroundColor Green
    } elseif ($gpu) {
        Write-Host "WARNING: this PC has an NVIDIA GPU ($gpu) but PyTorch cannot use it: $torchStatus" -ForegroundColor Red
        Write-Host 'Close LipSync and double-click Setup-Models.cmd to install CUDA PyTorch. Test mode refuses to run on the CPU here.' -ForegroundColor Red
    } elseif ($torchStatus -eq 'missing') {
        Write-Host 'Separation models are not installed yet: double-click Setup-Models.cmd once.' -ForegroundColor Yellow
    } else {
        Write-Host "No NVIDIA GPU found; separation will run on the CPU: $torchStatus" -ForegroundColor Yellow
    }
    $dashboardStamp = Join-Path $projectRoot 'node_modules\.lipsync-package-lock.sha256'
    $dashboardHash = (Get-FileHash 'package-lock.json' -Algorithm SHA256).Hash
    if (-not (Test-Path 'node_modules\next\package.json') -or -not (Test-Path $dashboardStamp) -or (Get-Content $dashboardStamp -Raw).Trim() -ne $dashboardHash) {
        Write-Host 'Installing dashboard dependencies...'
        & $npm ci
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
        Set-Content -LiteralPath $dashboardStamp -Value $dashboardHash
    }

    $logDirectory = Join-Path $projectRoot '.tools\startup'
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    Write-Host 'Starting the backend...'
    $backend = Start-Process -FilePath $python -ArgumentList '-m', 'uvicorn', 'processor.server:app', '--host', '0.0.0.0', '--port', '8000', '--ws-max-size', '750000', '--ws-max-queue', '4' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$logDirectory\backend.log" -RedirectStandardError "$logDirectory\backend-error.log"
    $services += $backend
    Wait-ForService 'http://127.0.0.1:8000/' $backend 'Backend'

    Write-Host 'Starting the dashboard...'
    $dashboard = Start-Process -FilePath $npm -ArgumentList 'run', 'dev', '--', '--port', '3000' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$logDirectory\dashboard.log" -RedirectStandardError "$logDirectory\dashboard-error.log"
    $services += $dashboard
    Wait-ForService 'http://localhost:3000/dashboard' $dashboard 'Dashboard'

    Write-Host "`nLipSync is ready." -ForegroundColor Green
    Write-Host 'Dashboard: http://localhost:3000/dashboard'
    Write-Host 'Separation models load when Test mode is enabled. Run Setup-Models.cmd once to install their dependencies. Falcon stays offline.'
    Write-Host "Debug log (share this file): $projectRoot\.tools\startup\lipsync-debug.log"
    Write-Host 'Phone addresses (use the Wi-Fi/Ethernet connection shared with your phone):'
    Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | ForEach-Object {
        $adapter = $_
        foreach ($address in $adapter.IPv4Address) {
            Write-Host ("  {0}: {1}:8000" -f $adapter.InterfaceAlias, $address.IPAddress)
        }
    }
    Write-Host "`nKeep this window open. Press Enter to stop both services."
    Write-Host "Logs: $logDirectory"
    Start-Process 'http://localhost:3000/dashboard'
    Read-Host | Out-Null
} catch {
    Write-Host "`nStartup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    foreach ($service in $services) {
        if (-not $service.HasExited) {
            # Include npm/Next and Python child processes, but only for our own servers.
            & taskkill.exe /PID $service.Id /T /F 2>&1 | Out-Null
        }
    }
}

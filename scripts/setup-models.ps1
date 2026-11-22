param([ValidateSet('auto','cpu','cuda')][string]$Device = 'auto', [switch]$ReplaceTorch)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
function Run-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Executable $Arguments" }
}
try {
    $replace = $ReplaceTorch.IsPresent
    if ($Device -eq 'auto') {
        # Use the NVIDIA GPU whenever the driver reports one (e.g. the A3000 laptop).
        $gpu = $null
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
        }
        if ($gpu) { $Device = 'cuda'; Write-Host "NVIDIA GPU found ($gpu): installing CUDA PyTorch." -ForegroundColor Green }
        else { $Device = 'cpu'; Write-Host 'No NVIDIA GPU found: installing CPU PyTorch.' }
    }
    $basePython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
    if (-not (Test-Path $basePython)) {
        $basePython = (& py -3.11 -c 'import sys; print(sys.executable)')
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 first.' }
    }
    foreach ($environment in @('.venv-local')) {
        $python = Join-Path $projectRoot "$environment\Scripts\python.exe"
        if (-not (Test-Path $python)) { Run-Checked $basePython @('-m','venv',$environment) }
        $installed = & $python -c "import importlib.util; print(bool(importlib.util.find_spec('torch')))"
        if ($installed -eq 'True') {
            $cudaBuild = & $python -c 'import torch; print(torch.version.cuda is not None)'
            if ($Device -eq 'cuda' -and $cudaBuild -ne 'True') {
                Write-Host "$environment has CPU PyTorch; replacing it with the CUDA build so the GPU is used."
                $replace = $true
            } elseif (-not $replace) {
                Write-Host "Keeping existing Torch in $environment."
            }
        }
        if ($installed -ne 'True' -or $replace) {
            $wheel = if ($Device -eq 'cuda') { 'cu126' } else { 'cpu' }
            Run-Checked $python @('-m','pip','install','--upgrade','--force-reinstall','torch==2.9.1','torchaudio==2.9.1','--index-url',"https://download.pytorch.org/whl/$wheel")
        }
        if ($environment -eq '.venv-local') {
            Run-Checked $python @('-m','pip','install','-r','processor/requirements.txt','speechbrain==1.1.2','soundfile==0.14.0','huggingface_hub==2.4.0')
        } else {
            Run-Checked $python @('-m','pip','install','-r','requirements-falcon-comparison.txt')
        }
        if ($Device -eq 'cuda') {
            Run-Checked $python @('-c',"import torch; assert torch.cuda.is_available(), 'CUDA unavailable; check driver and wheel'; print(torch.cuda.get_device_name())")
        }
    }
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
        $ffmpegTool = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg*" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $ffmpegTool) {
            Run-Checked 'winget' @('install','--id','Gyan.FFmpeg','--exact','--scope','user','--silent','--accept-package-agreements','--accept-source-agreements','--disable-interactivity')
            $ffmpegTool = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg*" -Recurse -Filter ffmpeg.exe | Select-Object -First 1
        }
        $env:PATH = $ffmpegTool.DirectoryName + ';' + $env:PATH
    }
    New-Item -ItemType Directory -Force '.tools' | Out-Null
    @{ device = $Device } | ConvertTo-Json | Set-Content '.tools\models.json'
    Write-Host 'Loading and smoke-testing the separator (first run downloads weights). Falcon stays off...'
    Run-Checked '.\.venv-local\Scripts\python.exe' @('scripts/smoke_models.py','--separator','--device',$Device)
    Write-Host 'Model setup complete. Double-click Start-LipSync.cmd.' -ForegroundColor Green
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

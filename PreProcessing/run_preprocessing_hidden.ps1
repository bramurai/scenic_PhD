# VGGSound Preprocessing - Fully Background (No Window)
# Runs completely hidden with only log files

$ErrorActionPreference = "Continue"

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$logFile = Join-Path $scriptPath "preprocessing.log"
$pidFile = Join-Path $scriptPath "preprocessing.pid"

Write-Host "Starting VGGSound preprocessing (hidden)..."
Write-Host "Log file: $logFile"
Write-Host "PID file: $pidFile"
Write-Host ""
Write-Host "Monitor progress:"
Write-Host "  Get-Content preprocessing.log -Wait -Tail 20"
Write-Host ""
Write-Host "Check progress file:"
Write-Host "  Get-Content PreProcessing\tfrecords\train\.progress.json"
Write-Host ""

# Create a script block to run
$command = @"
Set-Location '$scriptPath'
conda activate scenic_preprocessing
`$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
python download_and_preprocess_vggsound.py --csv_path=PreProcessing\vggsound_train.csv --output_path=PreProcessing\tfrecords\train --num_shards=100 --temp_dir=C:\temp\vggsound *>&1 | Tee-Object -FilePath '$logFile'
"@

# Start the process completely hidden
$process = Start-Process powershell -ArgumentList "-NoProfile", "-WindowStyle", "Hidden", "-Command", $command -PassThru -WorkingDirectory $scriptPath

# Save PID for later reference
$process.Id | Out-File $pidFile

Write-Host "Process started with PID: $($process.Id)"
Write-Host "Process is running in the background."
Write-Host ""
Write-Host "To stop the process:"
Write-Host "  Stop-Process -Id $($process.Id)"
Write-Host "Or:"
Write-Host "  Get-Process -Id (Get-Content preprocessing.pid) | Stop-Process"

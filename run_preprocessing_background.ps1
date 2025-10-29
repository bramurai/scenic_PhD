# VGGSound Preprocessing Background Runner
# This script runs the preprocessing in the background with logging

$ErrorActionPreference = "Continue"

# Set OpenMP workaround
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Activate conda environment and run
$scriptPath = $PSScriptRoot
$logFile = Join-Path $scriptPath "preprocessing.log"
$errorLogFile = Join-Path $scriptPath "preprocessing_errors.log"

Write-Host "Starting VGGSound preprocessing in background..."
Write-Host "Log file: $logFile"
Write-Host "Error log: $errorLogFile"
Write-Host ""
Write-Host "Monitor progress with:"
Write-Host "  Get-Content preprocessing.log -Wait -Tail 20"
Write-Host ""
Write-Host "To stop the process, find it with:"
Write-Host "  Get-Process python | Where-Object {`$_.CommandLine -like '*download_and_preprocess*'}"
Write-Host ""

# Run in background with output redirection
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "conda activate scenic_preprocessing; `$env:KMP_DUPLICATE_LIB_OK='TRUE'; python download_and_preprocess_vggsound.py --csv_path=PreProcessing\vggsound_train.csv --output_path=PreProcessing\tfrecords\train --num_shards=100 --temp_dir=C:\temp\vggsound 2>&1 | Tee-Object -FilePath '$logFile'"
) -WorkingDirectory $scriptPath

Write-Host "Process started! Check the new PowerShell window or monitor the log file."

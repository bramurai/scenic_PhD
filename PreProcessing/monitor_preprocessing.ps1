# Monitor VGGSound Preprocessing Progress

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$logFile = Join-Path $scriptPath "preprocessing.log"
$progressFile = Join-Path $scriptPath "PreProcessing\tfrecords\train\.progress.json"
$pidFile = Join-Path $scriptPath "preprocessing.pid"

Write-Host "=== VGGSound Preprocessing Monitor ===" -ForegroundColor Cyan
Write-Host ""

# Check if process is running
if (Test-Path $pidFile) {
    $pid = Get-Content $pidFile
    $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "Status: RUNNING (PID: $pid)" -ForegroundColor Green
        Write-Host "Started: $($process.StartTime)"
        Write-Host "CPU Time: $($process.TotalProcessorTime)"
        Write-Host "Memory: $([math]::Round($process.WorkingSet64 / 1MB, 2)) MB"
    } else {
        Write-Host "Status: NOT RUNNING (PID file exists but process not found)" -ForegroundColor Yellow
    }
} else {
    Write-Host "Status: NOT RUNNING (no PID file)" -ForegroundColor Red
}

Write-Host ""

# Check progress
if (Test-Path $progressFile) {
    $progress = Get-Content $progressFile | ConvertFrom-Json
    $processed = $progress.processed_indices.Count
    $successful = $progress.successful
    $failed = $progress.failed
    $lastUpdated = $progress.last_updated
    
    Write-Host "Progress:" -ForegroundColor Cyan
    Write-Host "  Processed: $processed / 183971 ($([math]::Round($processed/183971*100, 2))%)"
    Write-Host "  Successful: $successful"
    Write-Host "  Failed: $failed"
    Write-Host "  Success Rate: $([math]::Round($successful/($successful+$failed)*100, 2))%"
    Write-Host "  Last Updated: $lastUpdated"
    
    # Estimate time remaining
    if ($processed -gt 0) {
        $startTime = [DateTime]::Parse($progress.last_updated)
        $now = Get-Date
        # This is approximate - we'd need the actual start time
        Write-Host ""
        Write-Host "  Estimated time per video: ~3-5 seconds"
        Write-Host "  Estimated total time: ~6-15 days for full dataset"
    }
} else {
    Write-Host "No progress file found yet" -ForegroundColor Yellow
}

Write-Host ""

# Check output directory
$outputPath = Join-Path $scriptPath "PreProcessing\tfrecords\train"
if (Test-Path $outputPath) {
    $files = Get-ChildItem "$outputPath\*.tfrecord"
    $totalSize = ($files | Measure-Object -Property Length -Sum).Sum
    Write-Host "Output:" -ForegroundColor Cyan
    Write-Host "  TFRecord files: $($files.Count)"
    Write-Host "  Total size: $([math]::Round($totalSize / 1GB, 2)) GB"
}

Write-Host ""
Write-Host "Recent log entries:" -ForegroundColor Cyan
if (Test-Path $logFile) {
    Get-Content $logFile -Tail 10
} else {
    Write-Host "  No log file found yet" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Commands:" -ForegroundColor Cyan
Write-Host "  Watch log: Get-Content $logFile -Wait -Tail 20"
Write-Host "  Stop process: Stop-Process -Id (Get-Content $pidFile)"
Write-Host "  Rerun this monitor: .\monitor_preprocessing.ps1"

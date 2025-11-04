# PowerShell script to extract all TFRecord tar archives
# Maintains the batch_* directory structure that the model expects

param(
    [string]$Split = "train"  # "train" or "test"
)

$SourceDir = "${Split}_tfrecords_local"
$TargetDir = "${Split}_tfrecords_local"

Write-Host "============================================"
Write-Host "Extracting TFRecord archives for: $Split"
Write-Host "============================================"

# Count total archives
$Archives = Get-ChildItem -Path $SourceDir -Filter "*.tar.gz"
$TotalArchives = $Archives.Count
Write-Host "Found $TotalArchives tar archives to extract"
Write-Host ""

$ExtractedCount = 0
$SkippedCount = 0

foreach ($Archive in $Archives) {
    $ArchiveName = $Archive.Name
    $ArchivePath = $Archive.FullName
    
    # Extract batch ID from filename (e.g., train_tar00_batch_00000.tar.gz -> batch_00000)
    if ($ArchiveName -match 'batch_(\d+)\.tar\.gz$') {
        $BatchDir = "batch_$($Matches[1])"
        $TargetPath = Join-Path $TargetDir $BatchDir
        
        # Check if already extracted
        if (Test-Path $TargetPath) {
            $TFRecordCount = (Get-ChildItem -Path $TargetPath -Filter "*.tfrecord" -ErrorAction SilentlyContinue).Count
            if ($TFRecordCount -gt 0) {
                Write-Host "  [SKIP] $ArchiveName -> $BatchDir (already extracted: $TFRecordCount files)"
                $SkippedCount++
                continue
            }
        }
        
        Write-Host "  [EXTRACT] $ArchiveName -> $BatchDir"
        
        # Extract using tar (available in Windows 10+)
        try {
            tar -xzf $ArchivePath -C $TargetDir 2>$null
            
            if ($LASTEXITCODE -eq 0) {
                $TFRecordCount = (Get-ChildItem -Path $TargetPath -Filter "*.tfrecord" -ErrorAction SilentlyContinue).Count
                Write-Host "    ✓ Extracted $TFRecordCount TFRecord files"
                $ExtractedCount++
            } else {
                Write-Host "    ✗ Extraction failed!" -ForegroundColor Red
            }
        } catch {
            Write-Host "    ✗ Error: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "  [SKIP] $ArchiveName (unexpected filename format)"
        $SkippedCount++
    }
}

Write-Host ""
Write-Host "============================================"
Write-Host "Extraction Complete!"
Write-Host "============================================"
Write-Host "Extracted: $ExtractedCount archives"
Write-Host "Skipped: $SkippedCount archives (already extracted)"
Write-Host ""

# Count total TFRecord files
$TotalTFRecords = (Get-ChildItem -Path $TargetDir -Recurse -Filter "*.tfrecord").Count
$TotalBatches = (Get-ChildItem -Path $TargetDir -Directory -Filter "batch_*").Count

Write-Host "Total batch directories: $TotalBatches"
Write-Host "Total TFRecord files: $TotalTFRecords"
Write-Host ""
Write-Host "Dataset ready at: $TargetDir"
Write-Host ""
Write-Host "You can now delete the .tar.gz files to save space:"
Write-Host "  Remove-Item ${SourceDir}\*.tar.gz"

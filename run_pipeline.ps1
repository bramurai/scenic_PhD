# PowerShell wrapper to run the VGGSound preprocessing pipeline
# Usage: .\run_pipeline.ps1 [split] [start_tar]
# Example: .\run_pipeline.ps1 train 0

param(
    [string]$Split = "train",
    [int]$StartTar = 0
)

Write-Host "Starting VGGSound Pipeline"
Write-Host "Split: $Split"
Write-Host "Starting from tar: $StartTar"
Write-Host ""

# Use WSL bash if available, otherwise suggest manual approach
if (Get-Command wsl -ErrorAction SilentlyContinue) {
    Write-Host "Using WSL bash..."
    wsl bash -c "cd /mnt/c/Users/bravhee/Uta_PhD/scenic_PhD/scenic_PhD && ./auto_hf_pipeline.sh $Split $StartTar"
} else {
    Write-Host "WSL not available. Running via SSH to cluster..."
    Write-Host ""
    Write-Host "NOTE: This will run the pipeline ON the cluster instead of from your local machine."
    Write-Host "The pipeline is designed to run locally and SSH to cluster for each step."
    Write-Host ""
    $confirm = Read-Host "Continue anyway? (y/n)"
    
    if ($confirm -eq 'y') {
        ssh bravhee@mentat001.dccn.nl "cd scenic_PhD && nohup bash auto_hf_pipeline.sh $Split $StartTar > pipeline_output.log 2>&1 &"
        Write-Host ""
        Write-Host "Pipeline started on cluster in background."
        Write-Host "Monitor with: ssh bravhee@mentat001.dccn.nl 'tail -f scenic_PhD/pipeline_output.log'"
    } else {
        Write-Host "Cancelled."
    }
}

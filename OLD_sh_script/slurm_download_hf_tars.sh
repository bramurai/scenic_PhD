#!/bin/bash
#SBATCH --job-name=vgg_download_hf
#SBATCH --output=/home/mpla/bravhee/scenic_PhD/logs/download_hf_%A_%a.out
#SBATCH --error=/home/mpla/bravhee/scenic_PhD/logs/download_hf_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-19

# Download and extract VGGSound tar files from HuggingFace
# Each array job downloads and extracts one tar file

TAR_ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
BASE_URL="https://huggingface.co/datasets/Loie/VGGSound/resolve/main"
TAR_FILE="vggsound_${TAR_ID}.tar.gz"
DOWNLOAD_DIR="$HOME/scenic_PhD/vggsound_data"
EXTRACT_DIR="$DOWNLOAD_DIR/videos"

mkdir -p $DOWNLOAD_DIR
mkdir -p $EXTRACT_DIR
mkdir -p $HOME/scenic_PhD/logs

echo "=== Downloading and extracting tar file ${TAR_ID} ==="
echo "URL: ${BASE_URL}/${TAR_FILE}"
echo "Download dir: $DOWNLOAD_DIR"
echo "Extract dir: $EXTRACT_DIR"
echo "Space check before download:"
df -h $HOME | tail -1

cd $DOWNLOAD_DIR

# Download with wget (resume if interrupted)
if [ ! -f "${TAR_FILE}" ]; then
    echo "Downloading ${TAR_FILE}..."
    wget -c "${BASE_URL}/${TAR_FILE}" -O "${TAR_FILE}"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download ${TAR_FILE}"
        exit 1
    fi
else
    echo "${TAR_FILE} already exists, skipping download"
fi

echo "Download complete. File size:"
ls -lh "${TAR_FILE}"

# Extract tar file
echo "Extracting ${TAR_FILE}..."
tar -xzf "${TAR_FILE}" -C "$EXTRACT_DIR" --strip-components=6

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to extract ${TAR_FILE}"
    exit 1
fi

echo "Extraction complete!"
echo "Space check after extraction:"
df -h $HOME | tail -1

# Optionally delete tar file to save space (uncomment if needed)
# echo "Deleting tar file to save space..."
# rm -f "${TAR_FILE}"

echo "=== Completed tar file ${TAR_ID} ==="
echo "Extracted videos are in: ${EXTRACT_DIR}"
ls -lh "$EXTRACT_DIR" | head -20

# VGGSound Download and Preprocessing Pipeline - Step by Step

This guide walks you through the complete pipeline from raw VGGSound CSV to TFRecords ready for MBT training.

---

## Prerequisites

### 1. Check your conda environment is active:
```powershell
conda activate scenic_preprocessing
```

### 2. Install required packages:
```powershell
# If not already installed:
pip install yt-dlp librosa soundfile ffmpeg-python
```

### 3. Verify installations:
```powershell
# Check yt-dlp
yt-dlp --version

# Check Python packages
python -c "import librosa, ffmpeg, tensorflow; print('All packages OK')"
```

**Expected output:**
- yt-dlp version number (e.g., 2024.08.06)
- "All packages OK"

---

## Step 1: Clean and Format VGGSound CSV

### What this does:
- Converts VGGSound CSV format to preprocessing format
- Creates train/test splits
- Generates label mapping

### Command:
```powershell
python clean_vggsound_csv.py `
  --input_csv=PreProcessing\vggsound.csv `
  --output_dir=PreProcessing
```

### Expected output:
```
I1029 10:00:00.000 Reading PreProcessing\vggsound.csv
I1029 10:00:01.000 Loaded 199468 entries
I1029 10:00:01.000 Train: 183494, Test: 15974
I1029 10:00:01.000 Unique labels: 309
I1029 10:00:02.000 Converting to preprocessing format
I1029 10:00:03.000 Saved train CSV: PreProcessing\vggsound_train.csv (183494 entries)
I1029 10:00:03.000 Saved test CSV: PreProcessing\vggsound_test.csv (15974 entries)
I1029 10:00:03.000 Saved label mapping: PreProcessing\vggsound_labels.txt (309 classes)
```

### Output files created:
- `PreProcessing/vggsound_train.csv` - Training set
- `PreProcessing/vggsound_test.csv` - Test set
- `PreProcessing/vggsound_combined.csv` - All data
- `PreProcessing/vggsound_labels.txt` - Label to index mapping

### Verify output:
```powershell
# Check first few lines of train CSV
Get-Content PreProcessing\vggsound_train.csv -Head 5
```

**Expected format:**
```
video_path,start,end,label,clip_id
--0PQM4-hqg_000030.mp4,30,40,waterfall burbling,--0PQM4-hqg_000030
--56QUhyDQM_000185.mp4,185,195,playing tennis,--56QUhyDQM_000185
```

---

## Step 2: Test with a Small Subset (Recommended!)

Before processing all 200k videos, test with a small subset to verify everything works.

### Create a test subset:
```powershell
# Get first 10 entries
Get-Content PreProcessing\vggsound_train.csv -Head 11 | Set-Content PreProcessing\vggsound_test_small.csv
```

### Run preprocessing on test subset:
```powershell
python download_and_preprocess_vggsound.py `
  --csv_path=PreProcessing\vggsound_test_small.csv `
  --output_path=PreProcessing\tfrecords_test `
  --num_shards=1 `
  --temp_dir=C:\temp\vggsound_test
```

### What happens:
1. Script reads the CSV
2. For each video:
   - Downloads from YouTube using yt-dlp
   - Extracts RGB frames at 25fps
   - Extracts audio and computes log mel spectrogram
   - Writes to TFRecord
   - Deletes temporary video file
3. Outputs progress every 10 videos

### Expected output:
```
I1029 10:10:00.000 Reading CSV from PreProcessing\vggsound_test_small.csv
I1029 10:10:00.000 Processing 10 examples
I1029 10:10:00.000 Using temporary directory: C:\temp\vggsound_test
I1029 10:10:05.000 Processing 0/10 (success: 0, failed: 0, skipped: 0)
I1029 10:10:15.000 Processing 10/10 (success: 8, failed: 2, skipped: 0)
I1029 10:10:15.000 
=== Processing Complete ===
I1029 10:10:15.000 Successful: 8
I1029 10:10:15.000 Failed: 2
I1029 10:10:15.000 Output: PreProcessing\tfrecords_test
```

**Note:** Some videos may fail (deleted from YouTube, region-locked, etc.). This is normal.

### Verify TFRecord was created:
```powershell
dir PreProcessing\tfrecords_test
```

**Expected:**
```
data-00000-of-00001.tfrecord  (size: ~40-80MB for 8-10 videos)
```

---

## Step 3: Verify TFRecord Contents

Let's make sure the TFRecord contains the correct data.

### Create verification script:
```powershell
# Create verify_tfrecord.py
@"
import tensorflow as tf
import sys

tfrecord_path = sys.argv[1]
dataset = tf.data.TFRecordDataset([tfrecord_path])

for i, raw_record in enumerate(dataset.take(1)):
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    
    # Check RGB frames
    num_frames = len(example.feature_lists.feature_list['image/encoded'].feature)
    print(f'Video {i+1}:')
    print(f'  RGB frames: {num_frames}')
    
    # Check spectrogram
    if 'WAVEFORM/feature/floats' in example.feature_lists.feature_list:
        num_spec_frames = len(example.feature_lists.feature_list['WAVEFORM/feature/floats'].feature)
        n_mels = example.context.feature['WAVEFORM/num_mel_bins'].int64_list.value[0]
        print(f'  Spectrogram frames: {num_spec_frames}')
        print(f'  Mel bins: {n_mels}')
    
    # Check label
    if 'clip/label/string' in example.context.feature:
        label = example.context.feature['clip/label/string'].bytes_list.value[0].decode('utf-8')
        print(f'  Label: {label}')
    
    print('  ✓ TFRecord structure looks good!')
"@ | Set-Content verify_tfrecord.py
```

### Run verification:
```powershell
python verify_tfrecord.py PreProcessing\tfrecords_test\data-00000-of-00001.tfrecord
```

### Expected output:
```
Video 1:
  RGB frames: 250
  Spectrogram frames: 1000
  Mel bins: 128
  Label: waterfall burbling
  ✓ TFRecord structure looks good!
```

**What this means:**
- ✓ 250 RGB frames = 10 seconds × 25fps
- ✓ 1000 spectrogram frames = 10 seconds × 100 frames/second
- ✓ 128 mel bins (frequency dimension)
- ✓ Label is preserved

---

## Step 4: Full Dataset Processing

Now that everything is verified, process the full dataset.

### Important decisions:

**A. Storage location:**
- TFRecords will be ~1-2TB for full VGGSound
- Choose a location with enough space

**B. Number of shards:**
- Recommended: 100 shards for train, 10 for test
- More shards = easier to parallelize later

**C. Temporary directory:**
- Should have ~1-5GB free space
- Files are deleted as they're processed

### Process training set:
```powershell
python download_and_preprocess_vggsound.py `
  --csv_path=PreProcessing\vggsound_train.csv `
  --output_path=D:\VGGSound\tfrecords\train `
  --num_shards=100 `
  --temp_dir=C:\temp\vggsound `
  --batch_size=100
```

### Process test set:
```powershell
python download_and_preprocess_vggsound.py `
  --csv_path=PreProcessing\vggsound_test.csv `
  --output_path=D:\VGGSound\tfrecords\test `
  --num_shards=10 `
  --temp_dir=C:\temp\vggsound `
  --batch_size=100
```

### Expected time:
- **Single process:** 5-15 days for 200k videos
- **Reason:** ~1-3 seconds per video (download + process)

### Monitoring progress:
The script outputs progress every 10 videos:
```
Processing 1000/183494 (success: 850, failed: 150, skipped: 0)
Processing 2000/183494 (success: 1720, failed: 280, skipped: 0)
```

**Success rate:** Typically 80-90% (some videos deleted/unavailable)

---

## Step 5: Resume Interrupted Processing

If the process is interrupted (power loss, error, etc.), you can resume:

```powershell
python download_and_preprocess_vggsound.py `
  --csv_path=PreProcessing\vggsound_train.csv `
  --output_path=D:\VGGSound\tfrecords\train `
  --num_shards=100 `
  --skip_existing=True
```

The `--skip_existing=True` flag makes it skip shards that already exist.

---

## Step 6: Parallel Processing (Optional, Faster)

To speed up processing, split the work across multiple terminals/machines.

### Split CSV into chunks:
```powershell
python -c "
import pandas as pd
df = pd.read_csv('PreProcessing/vggsound_train.csv')
chunk_size = len(df) // 4
for i in range(4):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < 3 else len(df)
    df[start:end].to_csv(f'PreProcessing/vggsound_train_chunk{i}.csv', index=False)
print('Created 4 chunks')
"
```

### Run in 4 separate terminals:
```powershell
# Terminal 1
python download_and_preprocess_vggsound.py --csv_path=PreProcessing\vggsound_train_chunk0.csv --output_path=D:\VGGSound\tfrecords\train --num_shards=100

# Terminal 2
python download_and_preprocess_vggsound.py --csv_path=PreProcessing\vggsound_train_chunk1.csv --output_path=D:\VGGSound\tfrecords\train --num_shards=100

# Terminal 3
python download_and_preprocess_vggsound.py --csv_path=PreProcessing\vggsound_train_chunk2.csv --output_path=D:\VGGSound\tfrecords\train --num_shards=100

# Terminal 4
python download_and_preprocess_vggsound.py --csv_path=PreProcessing\vggsound_train_chunk3.csv --output_path=D:\VGGSound\tfrecords\train --num_shards=100
```

**Result:** 4x faster processing

---

## Step 7: Verify Final Output

### Check number of shards:
```powershell
(Get-ChildItem D:\VGGSound\tfrecords\train\*.tfrecord).Count
```

**Expected:** 100 (or your specified num_shards)

### Check total size:
```powershell
(Get-ChildItem D:\VGGSound\tfrecords\train\*.tfrecord | Measure-Object -Property Length -Sum).Sum / 1GB
```

**Expected:** ~1-2 TB for full VGGSound training set

### Verify a few TFRecords:
```powershell
python verify_tfrecord.py D:\VGGSound\tfrecords\train\data-00000-of-00100.tfrecord
python verify_tfrecord.py D:\VGGSound\tfrecords\train\data-00050-of-00100.tfrecord
```

---

## Troubleshooting

### "yt-dlp not found"
```powershell
pip install yt-dlp
```

### "librosa not installed"
```powershell
pip install librosa soundfile
```

### "Many videos failing to download"
- **Normal:** 10-20% failure rate (videos deleted/restricted)
- **High failure (>50%):** May be IP rate-limiting. Wait and retry later.

### "Out of disk space"
- Check `temp_dir` has space
- Reduce `--batch_size` to clean temp files more frequently
- Check final output location has enough space

### "Very slow downloads"
- YouTube may be rate-limiting
- Use VPN or different network
- Process in smaller batches over multiple days

---

## Next Steps

Once preprocessing is complete:

1. **Update MBT config** to point to your TFRecords:
   ```python
   config.dataset_configs.base_dir = 'D:/VGGSound/tfrecords'
   config.dataset_configs.tables = {
       'train': 'train/data@100',
       'test': 'test/data@10',
   }
   ```

2. **Test loading the dataset** with a small training run

3. **Train your MBT model!**

---

## Summary

✓ **Step 1:** Clean CSV → Format conversion  
✓ **Step 2:** Test subset → Verify pipeline works  
✓ **Step 3:** Verify TFRecords → Check data structure  
✓ **Step 4:** Full processing → Download + process + save  
✓ **Step 5:** Resume if needed → Skip existing shards  
✓ **Step 6:** Parallel processing → Speed up 4x  
✓ **Step 7:** Final verification → Ready for training  

**Total storage:** Only TFRecords (~1-2TB), no raw videos needed!

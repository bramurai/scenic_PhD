"""Test video download and processing locally."""

import sys
import os
sys.path.insert(0, 'PreProcessing')

import subprocess
import tempfile
import generate_audiovisual_from_file as gen

# Test with a known video
video_id = "-G5GTTzXP6g"  # One that failed on cluster
start_time = 0
url = f"https://www.youtube.com/watch?v={video_id}"

print(f"Testing video: {video_id}")
print(f"URL: {url}")

# Download video to temp file
temp_video = os.path.join(tempfile.gettempdir(), f"test_{video_id}.mp4")

# Clean up any existing file
if os.path.exists(temp_video):
    os.remove(temp_video)

try:
    print(f"\n1. Downloading video...")
    cmd = [
        'yt-dlp',
        '--format', 'worst[ext=mp4]/worst',
        '--output', temp_video,
        '--no-playlist',
        url
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)
    
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"Stdout: {result.stdout[:500]}")
    if result.stderr:
        print(f"Stderr: {result.stderr[:500]}")
    
    if result.returncode == 0 and os.path.exists(temp_video):
        file_size = os.path.getsize(temp_video)
        print(f"✓ Downloaded: {file_size} bytes")
        
        print(f"\n2. Extracting frames...")
        try:
            sequence_example = gen.create_sequence_example(
                video_path=temp_video,
                start_time=start_time,
                end_time=start_time + 10,
                label="test",
                clip_id=f"{video_id}_test",
                target_fps=25,
                decode_audio=True,
                audio_sample_rate=16000,
                n_mels=128,
                win_length_ms=25.0,
                hop_length_ms=10.0
            )
            
            print(f"✓ Processing successful!")
            print(f"  Context features: {len(sequence_example.context.feature)}")
            print(f"  Feature lists: {len(sequence_example.feature_lists.feature_list)}")
            
        except Exception as e:
            print(f"✗ Processing failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ Download failed")
        if result.stderr:
            print(f"Error: {result.stderr.decode()}")

finally:
    # Cleanup
    if os.path.exists(temp_video):
        os.remove(temp_video)
        print(f"\n3. Cleaned up temp file")

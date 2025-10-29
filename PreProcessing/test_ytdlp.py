"""Test if yt-dlp can download a known working video."""

import subprocess
import os

# Test with a Creative Commons video (always available)
test_video_id = "--0PQM4-hqg_000030"  # "Me at the zoo" - first YouTube video
test_url = f"https://www.youtube.com/watch?v={test_video_id}"
output = "test_download.mp4"

print(f"Testing yt-dlp with: {test_url}")

try:
    cmd = [
        'yt-dlp',
        '--format', 'worst',  # Download smallest format for speed
        '--output', output,
        '--no-playlist',
        test_url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    
    if result.returncode == 0 and os.path.exists(output):
        file_size = os.path.getsize(output) / (1024 * 1024)
        print(f"✓ Success! Downloaded {file_size:.2f} MB")
        #os.remove(output)
    else:
        print(f"✗ Failed!")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        
except Exception as e:
    print(f"✗ Error: {e}")

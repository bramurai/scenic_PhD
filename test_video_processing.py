import csv
from pathlib import Path

csv_file = "PreProcessing/vggsound_train.csv"
video_dir = Path("vggsound_temp/videos_00")

found = 0
missing = 0
checked = 0

with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_filename = row["video_path"].strip()
        video_path = video_dir / video_filename
        checked += 1
        if video_path.exists():
            found += 1
        else:
            missing += 1
        if checked >= 20:  # Only check first 20 for a quick test
            break

print(f"Checked: {checked}")
print(f"Found: {found}")
print(f"Missing: {missing}")
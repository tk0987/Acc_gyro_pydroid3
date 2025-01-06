import os
import re
import numpy as np
from datetime import datetime
import shutil

# Specify the directory and data file
directory = "/storage/emulated/0/DCIM/Camera/"
data_fname = "/storage/emulated/0/skanfon/res.txt"
cp_dir="/storage/emulated/0/skanfon/img/"

# Read and parse the data file
data = []
with open(data_fname, 'r') as f:
    for line in f:
        data.append(line.strip().split('\t'))

# Extract initial angles
anglex0, angley0, anglez0 = map(float, data[0][:3])

# Parse the rest of the data
parsed_data = []
for line in data[1:]:
    date_time_str, dx, dy, dz = line[0], line[1], line[2], line[3]
    datetime_obj = datetime.strptime(date_time_str, "%Y_%m_%d-%H_%M_%S_%f")
    parsed_data.append([
        datetime_obj.year, datetime_obj.month, datetime_obj.day,
        datetime_obj.hour, datetime_obj.minute, datetime_obj.second,
        int(datetime_obj.microsecond / 1000),  # Milliseconds
        round(anglex0 + float(dx), 2),  # Consistent precision
        round(angley0 + float(dy), 2),
        round(anglez0 + float(dz), 2)
    ])

# Parse image filenames and match them directly
matched_pairs = []  # Stores matched parsed data, image metadata, and filename
used_img_data_indices = set()
prevm = None

files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
for file in files:
    match = re.search(r"IMG_(\d{8})_(\d{9})", file)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSSmmm
        datetime_obj = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S%f")
        img_entry = [
            datetime_obj.year, datetime_obj.month, datetime_obj.day,
            datetime_obj.hour, datetime_obj.minute, datetime_obj.second,
            int(datetime_obj.microsecond / 1000)  # Milliseconds
        ]

        # Match the parsed data with the current image entry
        for i, entry in enumerate(parsed_data):
            if i in used_img_data_indices:
                continue

            # Check if the date and time match (excluding milliseconds)
            if np.array_equal(entry[:6], img_entry[:6]):
                diff = abs(entry[6] - img_entry[6])

                # Check if the difference in milliseconds is within the threshold
                if 0 <= diff <= 100 / 4:
                    matched_pairs.append((tuple(entry), tuple(img_entry), file))
                    used_img_data_indices.add(i)  # Mark this entry as used
                    break  # Proceed to the next image file

# Print unique matches
for match in matched_pairs:
    if prevm is None or not np.array_equal(match[1], prevm):
        prevm = match[1]
        print("Matched:", match[0], match[1], "Filename:", match[2])
        new_fname=f"{match[0][7]}_"+f"{match[0][8]}_"+f"{match[0][9]}"+".jpg"
        shutil.copy(os.path.join(directory,str(match[2])),os.path.join(cp_dir,new_fname))

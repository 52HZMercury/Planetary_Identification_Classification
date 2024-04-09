# Muhammad Azeem #  Number = 03244064060
import os
dic = []
f = 1
image_dir = r"D:\Program\Planetary_Identification_Classification\image_acquisition\dataset\土星"  # Add Your Folder Path
all_files = os.listdir(image_dir)
dic = all_files

for i in all_files:
    path = os.path.join(image_dir,i)
    a = f"{f}.jpg"  # Add Extension
    p = os.path.join(image_dir, a)
    f = f+1
    os.rename(path, p)
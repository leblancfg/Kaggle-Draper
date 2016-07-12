from os import listdir, makedirs
import numpy as np
from os.path import isfile, join, exists
import cv2

from chipy2 import im_stitcher
# from stitchDirector import train_set, test_set

'''
This script assumes standard Kaggle Script directory hierarchy:
/input/train/
/input/test/
/scripts/Director2.py <- YOU ARE HERE
'''

path = "../input/train/"
files_per_set = 5
pixels = 32

only_files = [f for f in listdir(path) if isfile(join(path, f))]
composite_list = [only_files[x:x + files_per_set] for x in range(0, len(only_files), files_per_set)]

if not exists('{0}npy'.format(path)):
    print("Creating npy subdirectory")
    makedirs('{0}npy'.format(path))

# Initialize the stack of training slices
train_stack = np.zeros((pixels, pixels, 3), dtype=np.uint8)

for idx, sets in enumerate(composite_list):
    print("{0}|{1}".format(idx, len(composite_list)))
    for i in range(len(sets)):
        j = i + 1
        while j < len(sets):
            image1 = cv2.imread(path + sets[i])
            image2 = cv2.imread(path + sets[j])
            stitched = im_stitcher(image1, image2)

            cols = len(stitched)
            rows = len(stitched[0])
            splitted = []

            for x, sx in enumerate(range(0, len(stitched), pixels)):
                for y, sy in enumerate(range(0, len(stitched[0]), pixels)):
                    splitted = stitched[sx:sx + pixels, sy:sy + pixels, :]

                    # TODO: Use matrix from stitchDirector to keep track of TARGET for each one of these slices.

                    # Keep only if > 50% of slice is non-zero (
                    if len(splitted[np.where(splitted > 0)]) > (pixels ** 2) * 0.50:
                        print('{4}npy/BRISK_matching_{0}_vs_{1}/{2}_{3}.npy'.
                              format(sets[i][:-4], sets[j][:-4], x, y, path))  # debugging
                        np.append(train_stack, splitted, axis=1)
            j += 1

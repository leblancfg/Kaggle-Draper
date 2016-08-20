from os import listdir, makedirs
import numpy as np
from os.path import isfile, join, exists
import cv2
import re
import csv
import imutils

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
only_files.sort()
composite_list = [only_files[x:x + files_per_set] for x in range(0, len(only_files), files_per_set)]


if not exists('{0}npy'.format(path)):
    print("Creating npy subdirectory")
    makedirs('{0}npy'.format(path))

# Initialize the stack of training slices

for idx, sets in enumerate(composite_list):
    setNumber = "".join(re.findall(r'set(.*?)_', sets[0]))
    print("{0}|{1}".format(idx, len(composite_list)))
    for i in range(len(sets)):
        j = i + 1
        while j < len(sets):
            image1 = cv2.imread(path + sets[i])
            image2 = cv2.imread(path + sets[j])
            stitched = im_stitcher(image1, image2)
            # this is for test purpose, resize to 600*450
            # stitched = im_stitcher(imutils.resize(image1,width=600), imutils.resize(image2,width=600))
            train_stack = [np.zeros((pixels, pixels, 3), dtype=np.uint8)]
            firstrun = True
            cols = len(stitched)
            rows = len(stitched[0])
            splitted = np.zeros((pixels, pixels, 3), dtype=np.uint8)

            for x, sx in enumerate(range(0, len(stitched), pixels)):
                for y, sy in enumerate(range(0, len(stitched[0]), pixels)):
                    widthOfImage = len(stitched[sx:sx + pixels, sy:sy + pixels, :][0])
                    heightOfImage = len(stitched[sx:sx + pixels, sy:sy + pixels, :])
                    splitted[:heightOfImage, :widthOfImage] = stitched[sx:sx + pixels, sy:sy + pixels, :]

                    # TODO: Use matrix from stitchDirector to keep track of TARGET for each one of these slices.

                    # Keep only if > 50% of slice is non-zero (
                    if len(splitted[np.where(splitted > 0)]) > (pixels ** 2) * 0.50:
                        filename = '{4}npy/set{5}_{0}_vs_{1}/{2}_{3}.jpg'.format(i+1, j+1, x, y, path, setNumber)
                        if firstrun is True:
                            firstrun = False
                            train_stack[0] = splitted
                        train_stack = np.concatenate((train_stack, [splitted]))
                        cv2.imwrite(filename, splitted)
            # np.save('{0}npy/set{1}_{2}_vs_{3}.npy'.format(path, setNumber, i+1, j+1), train_stack)
            j += 1

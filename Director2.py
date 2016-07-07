from os import listdir
import numpy as np
import imutils
from os.path import isfile, join
import os
import cv2
import time
import math
'''
This script assumes standard Kaggle Script directory hierarchy:
/input/train/
/input/test/
/scripts/Director2.py <- YOU ARE HERE
'''
from chipy2 import im_stitcher
path = "../input/train/"
numbersoffileperset = 5
onlyfiles = [f for f in listdir(path) if isfile(join(path,f))]
composite_list = [onlyfiles[x:x+numbersoffileperset] for x in range(0,len(onlyfiles),numbersoffileperset)]
for idx,sets in enumerate(composite_list):
    print("{0}|{1}".format(idx,len(composite_list)))
    for i in range(len(sets)):
        j=i+1
        while(j<len(sets)):
            image1 = cv2.imread("{0}".format(path+sets[i]))
            image2 = cv2.imread(path+sets[j])
            #image1 = imutils.resize(image1,width=600)
            #image2 = imutils.resize(image2, width=600)
            #-----------------------First compared to second----------------------------#1
            stitched = im_stitcher(image1, image2)
            print("{0} was tested against {1}".format(sets[i], sets[j]))

            #----------------------Second compared to first-----------------------------#2
            #stitched = im_stitcher(image2, image1)
            #print("{0} was tested against {1}".format(sets[j],sets[i]))
            #print('BRISK_matching_{0}_vs_{1}.npy'.format(sets[j][:-4],sets[i][:-4]))
            steps = 32
            cols = len(stitched)
            rows = len(stitched[0])
            splitted = []
            for x,sx in enumerate(range(0,len(stitched),steps)):
                for y,sy in enumerate(range(0,len(stitched[0]),steps)):
                    splitted = stitched[sx:sx+steps,sy:sy+steps,:]
                    if np.any(splitted):
                        #1
                        if not os.path.exists('../input/train/npy/1/BRISK_matching_{0}_vs_{1}'.format(sets[i][:-4], sets[j][:-4])):
                            os.makedirs('../input/train/npy/1/BRISK_matching_{0}_vs_{1}'.format(sets[i][:-4], sets[j][:-4]))
                            print '../input/train/npy/1/BRISK_matching_{0}_vs_{1}'.format(sets[i][:-4], sets[j][:-4]), ' created'
                        np.save('../input/train/npy/1/BRISK_matching_{0}_vs_{1}/{2}_{3}.npy'.format(sets[i][:-4], sets[j][:-4],x,y),splitted)


                        #2
                        #if not os.path.exists('../input/train/npy/0/BRISK_matching_{0}_vs_{1}'.format(sets[j][:-4],sets[i][:-4])):
                        #    os.makedirs('../input/train/npy/0/BRISK_matching_{0}_vs_{1}'.format(sets[j][:-4],sets[i][:-4]))
                        #    print '../input/train/npy/0/BRISK_matching_{0}_vs_{1}'.format(sets[j][:-4],sets[i][:-4]), ' created'
                        #np.save('../input/train/npy/0/BRISK_matching_{0}_vs_{1}/{2}_{3}.npy'.format(sets[j][:-4],sets[i][:-4],x,y),splitted)
            j += 1

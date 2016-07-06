import time
start = time.time()

from panorama import Stitcher
#import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
imageA = cv2.imread('/mnt/hgfs/Kaggle Drapper/train/set4_1.tif')
imageB = cv2.imread('/mnt/hgfs/Kaggle Drapper/train/set4_5.tif')
imageA = imutils.resize(imageA,width=600)
imageB = imutils.resize(imageB,width=600)
reading = time.time()
print('Read Images, time {0}'.format(reading-start))

#cv2.imshow("image a", imageA)
#cv2.waitKey(0)
#imageA = imutils.resize(imageA,width=500)
#imageB = imutils.resize(imageB,width=500)

# Init Stitcher class, feed images to function
stitcher_ = Stitcher()
result = stitcher_.stitch([imageA, imageB])
end = time.time()
print("Finished stitching in {0}".format(end-reading))

# show the images
print("Time to show")
#cv2.imshow("image a", imageA)
#cv2.imshow("image b", imageB)
#cv2.imwrite("test.png",imutils.resize(result,width=500))
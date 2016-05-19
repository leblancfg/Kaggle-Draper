from panorama import Stitcher
import argparse
import imutils
import cv2

#construct the argument parse and parse the arguments
imageA = cv2.imread('/mnt/hgfs/Kaggle Drapper/train/set12_1.tif')
imageB = cv2.imread('/mnt/hgfs/Kaggle Drapper/train/set12_3.tif')
imageA = imutils.resize(imageA,width=500)
imageB = imutils.resize(imageB,width=500)
stitcher = Stitcher()
(result) = stitcher.stitch([imageA,imageB])
print("Time to show")
# show the images
#cv2.imshow("image a", imageA)
#cv2.imshow("image b", imageB)
cv2.imshow("Result",result)
cv2.waitKey(0)
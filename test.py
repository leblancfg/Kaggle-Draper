import numpy as np
import cv2
import heapq
import imutils
import matplotlib.pyplot as plt
import glob, os
from scipy.stats import threshold as th


img1 = cv2.imread('/mnt/hgfs/Kaggle Drapper/train/set4_1.tif')
img2 = cv2.imread('/mnt/hgfs/Kaggle Drapper/train/set4_2.tif')
MIN_MATCH_COUNT=10

detector = cv2.AKAZE_create()
(kp1,desc1) = detector.detectAndCompute(img1,None)
(kp2,desc2) = detector.detectAndCompute(img2,None)
print("keypoints : {},descriptors : {}".format(len(kp1),desc1.shape))
print("keypoints : {},descriptors : {}".format(len(kp2),desc2.shape))

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(desc1,desc2, k=2)
good=[]
for m,n in matches:
     if m.distance < 0.75*n.distance:
         good.append(m)
'''         good.apprend([m])
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[40:50], None, flags=2)
'''
#to drawMatchesKnn we need to use the brackets
#Down there is to print the image with the matches
'''cv2.namedWindow('win1',flags=0)
cv2.resizeWindow('win1',1000,600)
cv2.imshow('win1', img3)
cv2.waitKey(0)'''
plt.subplot(211),plt.imshow(img1),plt.title('img1')
plt.subplot(212),plt.imshow(img2), plt.title('img2')
plt.show()
########################################################################
#
# Taking the1owls Image Matching script and experimenting;
# - downsizing images to speed up
#
# Want to extract the warp parameters
#
########################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt

print(cv2.__version__)
import time
beg=time.time()

def im_stitcher(imp1, imp2, pcntDownsize = 1.0, withTransparency=False):
    start=time.time()
    #Read image1
    image1 = cv2.imread(imp1)
    
    # perform the resizing of the image by pcntDownsize and create a Grayscale version
    dim1 = (int(image1.shape[1] * pcntDownsize), int(image1.shape[0] * pcntDownsize))
    img1 = cv2.resize(image1, dim1, interpolation = cv2.INTER_AREA)
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    #Read image2
    image2 = cv2.imread(imp2)
    
    # perform the resizing of the image by pcntDownsize and create a Grayscale version
    dim2 = (int(image2.shape[1] * pcntDownsize), int(image2.shape[0] * pcntDownsize))
    img2 = cv2.resize(image2, dim2, interpolation = cv2.INTER_AREA)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    #use BRISK to create keypoints in each image
    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(img1Gray,None)
    kp2, des2 = brisk.detectAndCompute(img2Gray,None)
    
    # use BruteForce algorithm to detect matches among image keypoints 
    dm = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    
    matches = dm.knnMatch(des1,des2, 2)
    matches_ = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches_.append((m[0].trainIdx, m[0].queryIdx))
    
    kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1,1,2)
    kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1,1,2)
    
    
    H, mask = cv2.findHomography(kp2_,kp1_, cv2.RANSAC, 4.0)
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    
    t = [-xmin,-ymin]
    
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    
    #warp the colour version of image2
    im = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    
    #overlay colur version of image1 to warped image2
    # if withTransparency == True:
    h3,w3 = im.shape[:2]
    bim = np.zeros((h3,w3,3), np.uint8)
    bim[t[1]:h1+t[1],t[0]:w1+t[0]] = img1

    #   #imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #   #imColor = cv2.applyColorMap(imGray, cv2.COLORMAP_JET)

    #   #im =(im[:,:,2] - bim[:,:,2])
    #   im = cv2.addWeighted(im,0.6,bim,0.6,0)
    im = np.dstack((im, bim))
    # else:
    #   im[t[1]:h1+t[1],t[0]:w1+t[0]] = img1  # Here be the crux. Left side is "position of img1", right?
    print("Image took {0} s to complete.".format(round(time.time()-start,1)))
    return(im)

##########################################################
#
# Match all combinations of one set of images
#
##########################################################

#img104_1_166_1 = im_stitcher("../input/test/set104_1.tif", "../input/test/set166_1.tif", 0.4, True)
#img104_2_166_4 = im_stitcher("../input/test/set104_2.tif", "../input/test/set166_4.tif", 0.4, True)
#img104_3_166_5 = im_stitcher("../input/test/set104_3.tif", "../input/test/set166_5.tif", 0.4, True)
#img104_4_166_3 = im_stitcher("../input/test/set104_4.tif", "../input/test/set166_3.tif", 0.4, True)
#img104_5_166_2 = im_stitcher("../input/test/set104_5.tif", "../input/test/set166_2.tif", 0.4, True)

#plt.imsave('Set104_1_166_1_BRISK_matching.tif',img104_1_166_1) 
#plt.imsave('Set104_2_166_4_BRISK_matching.tif',img104_2_166_4) 
#plt.imsave('Set104_3_166_5_BRISK_matching.tif',img104_3_166_5) 
#plt.imsave('Set104_4_166_3_BRISK_matching.tif',img104_4_166_3) 
#plt.imsave('Set104_6_166_2_BRISK_matching.tif',img104_5_166_2) 

#img1_1_85_5 = im_stitcher("../input/test/set1_1.tif", "../input/test/set85_5.tif", 0.4, True)
#img1_2_85_4 = im_stitcher("../input/test/set1_2.tif", "../input/test/set85_4.tif", 0.4, True)
#img1_3_85_2 = im_stitcher("../input/test/set1_3.tif", "../input/test/set85_2.tif", 0.4, True)
#img1_4_85_3 = im_stitcher("../input/test/set1_4.tif", "../input/test/set85_3.tif", 0.4, True)
#img1_5_85_1 = im_stitcher("../input/test/set1_5.tif", "../input/test/set85_1.tif", 0.4, True)

#plt.imsave('Set1_1_85_5_BRISK_matching.tif',img1_1_85_5) 
#plt.imsave('Set1_2_85_4_BRISK_matching.tif',img1_2_85_4) 
#plt.imsave('Set1_3_85_2_BRISK_matching.tif',img1_3_85_2) 
#plt.imsave('Set1_4_85_3_BRISK_matching.tif',img1_4_85_3) 
#plt.imsave('Set1_5_85_1_BRISK_matching.tif',img1_5_85_1) 

#img3_1_22_1 = im_stitcher("../input/test/set3_1.tif", "../input/test/set22_1.tif", 0.4, True)
#img3_2_22_2 = im_stitcher("../input/test/set3_2.tif", "../input/test/set22_2.tif", 0.4, True)
#img3_3_22_5 = im_stitcher("../input/test/set3_3.tif", "../input/test/set22_5.tif", 0.4, True)
#img3_4_22_3 = im_stitcher("../input/test/set3_4.tif", "../input/test/set22_3.tif", 0.4, True)
#img3_5_22_4 = im_stitcher("../input/test/set3_5.tif", "../input/test/set22_4.tif", 0.4, True)

#plt.imsave('Set3_1_22_1_BRISK_matching.tif',img3_1_22_1) 
#plt.imsave('Set3_2_22_2_BRISK_matching.tif',img3_2_22_2) 
#plt.imsave('Set3_3_22_5_BRISK_matching.tif',img3_3_22_5) 
#plt.imsave('Set3_4_22_3_BRISK_matching.tif',img3_4_22_3) 
#plt.imsave('Set3_5_22_4_BRISK_matching.tif',img3_5_22_4) 

#img5_1_68_3 = im_stitcher("../input/train/set5_1.tif", "../input/test/set68_3.tif", 0.4, True)
#plt.imsave('Set5_1_68_3_BRISK_matching.tif',img5_1_68_3) 

img160_5_74_1 = im_stitcher("/mnt/hgfs/Kaggle Drapper/train/set78_1.tif", "/mnt/hgfs/Kaggle Drapper/train/set78_5.tif")
end=time.time()
# img160_5_74_2 = im_stitcher("../input/train/set160_5.tif", "../input/test/set74_2.tif")
# img160_5_74_3 = im_stitcher("../input/train/set160_5.tif", "../input/test/set74_3.tif")
# img160_5_74_4 = im_stitcher("../input/train/set160_5.tif", "../input/test/set74_4.tif")
# img160_5_74_5 = im_stitcher("../input/train/set160_5.tif", "../input/test/set74_5.tif")

np.save('Set160_5_74_1_BRISK_matching.npy',img160_5_74_1)

# This can save the file as a human-readable .txt or .csv, but it's 50x slower.
# text = ''
# for row in img160_5_74_1:
#     for e in row:
#         text += '{} {} {} {} {} {},'.format(e[0], e[1], e[2], e[3], e[4], e[5])
#     text += '\n'
#
# # Write the string to a file.
# with open('image.csv', 'w') as f:
#     f.write(text)
#
print "Image took {0} s to save as npy".format(round(time.time()-end),)

# np.savetxt('Set160_5_74_2_BRISK_matching.csv',img160_5_74_2, delimiter=",", fmt="%s")
# np.savetxt('Set160_5_74_3_BRISK_matching.csv',img160_5_74_3, delimiter=",", fmt="%s")
# np.savetxt('Set160_5_74_4_BRISK_matching.csv',img160_5_74_4, delimiter=",", fmt="%s")
# np.savetxt('Set160_5_74_5_BRISK_matching.csv',img160_5_74_5, delimiter=",", fmt="%s")






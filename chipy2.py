from osgeo import ogr
import numpy as np
import cv2
# import matplotlib.pyplot as plt
# import time

print('OpenCV version: ' + cv2.__version__)


def flattenImage(im):
    ret, thresh = cv2.threshold(im, 1, 255, cv2.THRESH_BINARY)
    flat = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY, 0)
    return flat


def im_stitcher(image1, image2, pcntDownsize=1.0, withTransparency=False):
    # start = time.time()

    # perform the resizing of the image by pcntDownsize and create a Grayscale version
    dim1 = (int(image1.shape[1] * pcntDownsize), int(image1.shape[0] * pcntDownsize))
    img1 = cv2.resize(image1, dim1, interpolation=cv2.INTER_AREA)
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # perform the resizing of the image by pcntDownsize and create a Grayscale version
    dim2 = (int(image2.shape[1] * pcntDownsize), int(image2.shape[0] * pcntDownsize))
    img2 = cv2.resize(image2, dim2, interpolation=cv2.INTER_AREA)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # use BRISK to create keypoints in each image
    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(img1Gray, None)
    kp2, des2 = brisk.detectAndCompute(img2Gray, None)

    # use BruteForce algorithm to detect matches among image keypoints
    dm = cv2.DescriptorMatcher_create("BruteForce-Hamming")

    matches = dm.knnMatch(des1, des2, 2)
    matches_ = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches_.append((m[0].trainIdx, m[0].queryIdx))
    kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1, 1, 2)
    kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(kp2_, kp1_, cv2.RANSAC, 4.0)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)

    # calculate the height and the width of the final image
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # warp the colour version of image2
    im = cv2.warpPerspective(img2, Ht.dot(H), (xmax - xmin, ymax - ymin))

    # overlay image1 to warped image2
    h3, w3 = im.shape[:2]
    bim = np.zeros((h3, w3, 3), np.uint8)
    bim[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    # Creating the mask
    # Getting the points in wkt format
    wkt1 = "POLYGON (({0} {1}, {2} {3}, {4} {5}, {6} {7}, {0} {1}))".\
        format(pts1[0][0][0]+t[0], pts1[0][0][1]+t[1], pts1[1][0][0]+t[0], pts1[1][0][1]+t[1],
               pts1[2][0][0]+t[0], pts1[2][0][1]+t[1], pts1[3][0][0]+t[0], pts1[3][0][1]+t[1])
    wkt2 = "POLYGON (({0} {1}, {2} {3}, {4} {5}, {6} {7}, {0} {1}))".\
        format(pts2_[0][0][0]+t[0], pts2_[0][0][1]+t[1], pts2_[1][0][0]+t[0], pts2_[1][0][1]+t[1],
               pts2_[2][0][0]+t[0], pts2_[2][0][1]+t[1], pts2_[3][0][0]+t[0],  pts2_[3][0][1]+t[1])

    # Creating the polygons
    poly1 = ogr.CreateGeometryFromWkt(wkt1)
    poly2 = ogr.CreateGeometryFromWkt(wkt2)

    # Finding the intersected area
    intersection = poly1.Intersection(poly2)
    overlapWKTformat = intersection.ExportToWkt()

    # Parsing
    overlapCoordinate = np.array(np.int32(np.float32(
        [points.split(' ') for points in overlapWKTformat[10:len(overlapWKTformat)-2].split(',')]))
    )
    maskB = np.zeros((h3, w3, 3), np.uint8)
    cv2.fillConvexPoly(maskB, overlapCoordinate, (255, 255, 255))

    xmin = min(x for x, y in overlapCoordinate)
    xmax = max(x for x, y in overlapCoordinate)
    ymin = min(y for x, y in overlapCoordinate)
    ymax = max(y for x, y in overlapCoordinate)

    # im3 = cv2.addWeighted(im,0.6,bim,0.6,0)
    im = cv2.bitwise_and(im, maskB)
    bim = cv2.bitwise_and(bim, maskB)
    im = im[ymin:ymax, xmin:xmax]
    bim = bim[ymin:ymax, xmin:xmax]

    # Subtract the two images, to use as a single, RGB-like, 3-channel matrix
    # cv2.imwrite('im1.jpg', im)
    # hist = cv2.calcHist([im], [0], None, [256], [0, 256])
    # cv2.imwrite('hist1.jpg', hist)
    im = cv2.absdiff(im, bim)
    # cv2.imwrite('im2.jpg', im)
    # print("Image took {0} s to complete.".format(round(time.time() - start, 1)))
    print ".",
    return im



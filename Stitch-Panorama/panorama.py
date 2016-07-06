import numpy as np
#import imutils
import cv2
from osgeo import ogr

class Stitcher:
    #def __init__(self):
        #self.isv3 = imutils.is_cv3()
        # determine if we are using OpenCV 3.x
    def stitch(self,images, ratio=0.75, reprojThresh=4.0):
    # unpack the images, then detect keypoints and extract
    # local invariant descriptors from them
        (imageB,imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        #match features between the two images
        M = self.matchKeypoints(kpsA,kpsB, featuresA,featuresB,ratio,reprojThresh)
        #if the match is None, then there aren't enough matched keypoints to create a panorama
        (matches, H, status) = M
        #(hA, wA) = imageA.shape[:2]
        #(hB, wB) = imageB.shape[:2]

        # Transparency
        #imageATran = np.zeros((hA,wA,3),np.uint8)
        #imageBTran = np.zeros((hB, wB, 3), np.uint8)
        #cv2.addWeighted(imageATran,0.6,imageA,0.6,-1,imageATran)
        #cv2.addWeighted(imageBTran, 0.6, imageB, 0.6, -1, imageBTran)

        #/Transparency
        result = cv2.warpPerspective(imageA, H,(imageA.shape[1],imageA.shape[0]))
        #result[0:imageB.shape[0],0:imageB.shape[1]] = imageBTran
        #result = cv2.warpPerspective(imageATran, H,(imageA.shape[1]+imageB.shape[1],imageA.shape[0]))
        #result[0:imageB.shape[0],0:imageB.shape[1]] = imageBTran
        (hC, wC) = result.shape[:2]
        resultTran = np.zeros((hC,wC,3), np.uint8)
        resultTran[0:imageB.shape[0],0:imageB.shape[1]] = imageB

        # Creating the mask
        # Getting the points in wkt format
        wkt1 = "POLYGON (({0} {1}, {2} {3}, {4} {5}, {6} {7}, {0} {1}))".format(pts1[0][0][0], pts1[0][0][1], pts1[1][0][0],
                                                                                pts1[1][0][1], pts1[2][0][0], pts1[2][0][1],
                                                                                pts1[3][0][0], pts1[3][0][1])
        wkt2 = "POLYGON (({0} {1}, {2} {3}, {4} {5}, {6} {7}, {0} {1}))".format(pts2_[0][0][0], pts2_[0][0][1],
                                                                                pts2_[1][0][0], pts2_[1][0][1],
                                                                                pts2_[2][0][0], pts2_[2][0][1],
                                                                                pts2_[3][0][0], pts2_[3][0][1])
        # Creating the polygons
        poly1 = ogr.CreateGeometryFromWkt(wkt1)
        poly2 = ogr.CreateGeometryFromWkt(wkt2)
        # Finding the intersected area
        intersection = poly1.Intersection(poly2)
        overlapWKTformat = intersection.ExportToWkt()
        # Parsing
        overlapCoordinate = np.array(np.int32(
            np.float32([points.split(' ') for points in overlapWKTformat[10:len(overlapWKTformat) - 2].split(',')])))
        maskB = np.zeros((imageA.shape[0], imageA.shape[1]))
        out = cv2.fillConvexPoly(maskB, overlapCoordinate, 1)
        cv2.imshow("resultTran",resultTran)
        cv2.imshow("result",result)
        cv2.imshow("out",out)

        #result = cv2.addWeighted(result,0.6,resultTran,0.6,0)
        #cv2.imshow("final",result)
        cv2.waitKey(0)
        return result

    def detectAndDescribe(self, image):
        #convert the image to grayscale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps,features) = descriptor.detectAndCompute(gray,None)
        #convert the keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps])
        #return a tuple of keypoints and features
        '''detector = cv2.AKAZE_create()
        (kps, features) = detector.detectAndCompute(image, None)'''
        return (kps, features)

    def matchKeypoints(self, kpsA,kpsB,featuresA, featuresB, ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA,featuresB,2)
        matches=[]

        # loop over the raw matches
        for m in rawMatches:
            #ensure the distance is within a certain ration of each other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx,m[0].queryIdx))
        if(len(matches)<5):
            return None

        ptsA = np.float32([kpsA[i] for (_,i) in matches])
        ptsB = np.float32([kpsB[i] for (i,_) in matches])
        (H, status) = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,reprojThresh)

        #return the matches along with the homography matrix and status of each matched points
        return (matches,H,status)

    def drawMatches(self,imageA,imageB,kpsA,kpsB,matches,status):
        #initialize the output visualization image
        (hA,wA) = imageA.shape[:2]
        (hB,wB) = imageA.shape[:2]
        vis = np.zeros((max(hA,hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        #loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches,status):
            #only process the match if the keypoint was successfully matched
            if s == 1:
                #draw the match
                ptA = (int(kpsA[queryIdx][0]),int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0])+wA,int(kpsB[trainIdx][1]))
                cv2.line(vis,ptA,ptB,(0,255,0),1)
            gc.collect()

        #return the visualisation
        return vis


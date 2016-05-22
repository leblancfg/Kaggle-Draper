from os import listdir
import numpy as np
import imutils
from os.path import isfile, join
import cv2
from chipy import  im_stitcher
path = "/mnt/hgfs/Kaggle Drapper/train/"
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
            image1 = imutils.resize(image1,width=600)
            image2 = imutils.resize(image2, width=600)
            #stitched = im_stitcher(image1, image2)
            #print("{0} was tested against {1}".format(sets[i], sets[j]))
            #np.save('/mnt/hgfs/Kaggle Drapper/train/npy/1/BRISK_matching_{0}_vs_{1}.npy'.format(sets[i][:-4], sets[j][:-4]),stitched)
            #print('BRISK_matching_{0}_vs_{1}.npy'.format(sets[i][:-4], sets[j][:-4]))
            stitched = im_stitcher(image2,image1)
            print("{0} was tested against {1}".format(sets[i],sets[j]))
            #np.save('/mnt/hgfs/Kaggle Drapper/train/npy/0/BRISK_matching_set223_5_vs_set223_3.npy')
            np.save('/mnt/hgfs/Kaggle Drapper/train/npy/0/BRISK_matching_{0}_vs_{1}.npy'.format(sets[j][:-4],sets[i][:-4]), stitched)
            print('BRISK_matching_{0}_vs_{1}.npy'.format(sets[j][:-4],sets[i][:-4]))
            j += 1

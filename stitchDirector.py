# -*- coding: utf-8 -*-
'''
Created on Thu May 12 12:22:09 2016
@author: leblancfg
This script assumes standard Kaggle Script directory hierarchy:
/input/train/
/input/test/
/scripts/stitchDirector.py <- YOU ARE HERE
THE LEFT-MOST, OR TOP-MOST IMAGE BEING YOUNGER IS 1
IF IT IS OLDER, THEN 0
Also, start by scrolling to bottom. =)
'''

import glob, re, csv  # Standard modules

# This guy creates a 4-column matrix, rows = image pairs, last column being TARGET
def train_set(dummy=1):
    # Constants
    numPerSet = 5  # Number of images per imageset
    # numSets = 344  # Number of imagesets in (train+test)
    trainTarget= []

    for filename in glob.glob('../input/train/*.tif'):
        #print filename
        setNum = re.search('(\d)+(?=_)', filename)
        setNum = int(setNum.group())
        dayNum = re.search('(\d)(?=\.)', filename)  # this guy is the left-most one
        dayNum = int(dayNum.group())
        #print(dayNum.group())
        for i in range(1,numPerSet+1):
            if dayNum < i:
                more = [setNum,dayNum, i, 1]
                trainTarget.append(more)
                #print(more)
            elif dayNum > i:
                more = [setNum, dayNum, i, 0]
                trainTarget.append(more)
                #print(more)
            else:
                pass

    # Sort that sucker
    for i in reversed(range(0,2)):
        trainTarget.sort(key=lambda x: x[i])
    #print(len(trainTarget))
    if dummy != 1:  # If stitchDirector.py is invoked directly, save to csv.
        with open('train-set.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(trainTarget)
            print('Successfully wrote train-set.csv, containing {0} rows of data:'.format(len(trainTarget)))
        print('{0} ...\n'.format(trainTarget[0:5]))
    return(trainTarget)


# This fella creates a 3-column matrix, rows = imagepairs
def test_set(dummy=1):
    # Constants
    numPerSet = 5  # Number of images per imageset
    # numSets = 344  # Number of imagesets in (train+test)
    testTarget = []

    for filename in glob.glob('../input/test/*.tif'):
        #print filename
        setNum = int(re.search('(\d)+(?=_)', filename).group())
        dayNum = int(re.search('(\d)(?=\.)', filename).group())  # this guy is the left-most one
        #print(dayNum.group())
        for i in range(1,numPerSet+1):
            if dayNum < i:
                more = [setNum,dayNum, i]
                testTarget.append(more)
                #print(more)
            elif dayNum > i:
                more = [setNum, dayNum, i]
                testTarget.append(more)
                # print(more)
            else:
                pass

    # Sort that sucker
    for i in reversed(range(0,2)):
        testTarget.sort(key=lambda x: x[i])
    #print(len(testTarget))

    if dummy != 1:  # If stitchDirector.py is invoked directly, save to csv.
        with open('test-set.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(testTarget)
            print('Successfully wrote test-set.csv, containing {0} rows of data:'.format(len(testTarget)))
        print('{0} ...\n'.format(testTarget[0:6]))
    return(testTarget)


'''Juste un petit switchbox pour le fichier:
* Si stitchDirector.py est invoque directement, roule les fonctions et sauve en .csv
* Autrement si on 'import stitchDirector',
    fais juste offrir les fonctions, qui retournent les listes-de-listes
    trainTarget:    [setNum, leftImageNum, rightImageNum, target], et
    testTarget:     [setNum, leftImageNum, rightImageNum].
C'est un cool petit truc python qui est devenu standard avec le temps, voir:
http://stackoverflow.com/questions/419163/what-does-if-name-main-do
'''
if __name__ == '__main__':
    trainTarget = train_set(0)
    testTarget = test_set(0)
    print('Total image-pairs: {0}.'.format(len(trainTarget)+len(testTarget)))


# # List every 2-member permutation of set images
# iterSet = [''.join(p) for p in itertools.permutations(''.join(str(n) for n in range(1,numPerSet+1)),2)]
# #print '{0} permutations per image set'.format(len(iterSet))
# #print iterSet
#
# # Iterate AKAZE stitching for every 2-member images, for each set.
# for i in (1:numSets):
#   for j in iterSet:
#     leftImage = 'set{0}_{1}.tif'.format(i,j[0])
#     rightImage = 'set{0}_{1}.tif'.format(i,j[-1])
#     outputName = 'set{0}_{1}_{2}.tif'.format(i,j[0],j[-1])
#
#     # Send to AKAZE_stitching.py
#
#     if i in trainSets:
#       if j[0] < j[-1]:
#         # trainTarget$target becomes 1
#       else:
#         # trainTarget$target becomes 0
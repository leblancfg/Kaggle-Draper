# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:22:09 2016
@author: leblancfg
This script assumes standard Kaggle Script directory hierarchy:
/input/train/
/input/test/
/scripts/stitchDirector.py <- YOU ARE HERE

THE LEFT-MOST, OR TOP-MOST IMAGE BEING YOUNGER IS 1
IF IT IS OLDER, THEN 0
"""
import csv
import glob
import re


def train_set(dummy=1):
    # Creates a 4-column matrix:
    #
    # setID,
    # top dayNum,
    # bottom dayNum,
    # TARGET

    # Constants
    numPerSet = 5  # Number of images per imageset
    trainTarget= []

    # Create matrix by grepping filenames
    for filename in glob.glob('../input/train/*.tif'):
        setNum = re.search('(\d)+(?=_)', filename)  # Get number before '_'
        setNum = int(setNum.group())
        dayNum = re.search('(\d)(?=\.)', filename)  # Get number after '_'
        dayNum = int(dayNum.group())
        for i in range(1, numPerSet + 1):
            if dayNum < i:
                more = [setNum, dayNum, i, 1]
                trainTarget.append(more)
            elif dayNum > i:
                more = [setNum, dayNum, i, 0]
                trainTarget.append(more)
            else:
                pass

    # Sort that sucker
    for i in reversed(range(0, 2)):
        trainTarget.sort(key=lambda x: x[i])

    if dummy != 1:  # If stitchDirector.py is invoked directly, save to csv.
        with open('train-set.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(trainTarget)
            print('Successfully wrote train-set.csv, containing {0} rows of data:'.format(len(trainTarget)))
    return trainTarget


def test_set(dummy=1):
    # Creates a 3-column matrix:
    #
    # setID,
    # top dayNum,
    # bottom dayNum,

    # Constants
    numPerSet = 5  # Number of images per imageset
    testTarget = []

    # Create matrix by grepping filenames
    for filename in glob.glob('../input/test/*.tif'):
        setNum = int(re.search('(\d)+(?=_)', filename).group())  # Get number before '_'
        dayNum = int(re.search('(\d)(?=\.)', filename).group())  # Get number after '_'
        for i in range(1, numPerSet + 1):
            if dayNum < i:
                more = [setNum, dayNum, i]
                testTarget.append(more)
            elif dayNum > i:
                more = [setNum, dayNum, i]
                testTarget.append(more)
            else:
                pass

    # Sort that sucker
    for i in reversed(range(0, 2)):
        testTarget.sort(key=lambda x: x[i])

    if dummy != 1:  # If stitchDirector.py is invoked directly, save to csv.
        with open('test-set.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(testTarget)
            print('Successfully wrote test-set.csv, containing {0} rows of data:'.format(len(testTarget)))
    return testTarget


''' Switchbox:
* Si stitchDirector.py est invoque directement, roule les fonctions et sauve en .csv
* Autrement si on 'import stitchDirector',
    fait juste offrir les fonctions, qui retournent les listes-de-listes
    trainTarget:    [setNum, leftImageNum, rightImageNum, TARGET], et
    testTarget:     [setNum, leftImageNum, rightImageNum].
'''
if __name__ == '__main__':
    trainTarget = train_set(0)
    testTarget = test_set(0)
    print('Total image-pairs: {0}.'.format(len(trainTarget) + len(testTarget)))

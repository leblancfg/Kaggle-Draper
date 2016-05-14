# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:22:09 2016

@author: leblancfg
"""
import glob, os
import itertools

# Constants
numPerSet = 5  # Number of images per imageset
numSets = 344  # Number of imagesets in (train+test)

# Remember what imagesets are part of train
# Create trainTarget, associates "boolean" chronological value for every 2-member image permutation
trainTarget = []
os.chdir('..')
for file in glob.glob('*.tif'):
    #print file
    setValue = [re.search(INSERT REGEX HERE,file),0]  # trainTarget has columns 'set#' and 'target'
    trainTarget.append(setValue)

# List every 2-member permutation of set images
iterSet = [''.join(p) for p in itertools.permutations(''.join(str(n) for n in range(1,numPerSet+1)),2)]
#print '{0} permutations per image set'.format(len(iterSet))
#print iterSet

# Iterate AKAZE stitching for every 2-member images, for each set.
for i in (1:numSets):
  for j in iterSet:
    leftImage = 'set{0}_{1}.tif'.format(i,j[0])
    rightImage = 'set{0}_{1}.tif'.format(i,j[-1])
    outputName = 'set{0}_{1}_{2}.tif'.format(i,j[0],j[-1])
    
    # Send to AKAZE_stitching.py
    
    if i in trainSets:
      if j[0] < j[-1]:
        # trainTarget$target becomes 1
      else:
        # trainTarget$target becomes 0
    
    
    

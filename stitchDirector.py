# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:22:09 2016

@author: leblancfg
"""
import glob
import itertools

# Number of images in set
numSet = 5

# List every 2-member permutation of set images
iterSet = [''.join(p) for p in itertools.permutations(''.join(str(n) for n in range(1,numSet+1)),2)]
print '{0} permutations per image set'.format(len(iterSet))
#print iterSet


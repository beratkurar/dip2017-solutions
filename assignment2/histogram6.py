# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:20:21 2017

@author: B
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt



def computeImageHistogram (image):
    rows=int(image.shape[0])
    cols=int(image.shape[1])
    histogram=[0]*256
    for i in range (0,rows):
        for j in range (0,cols):
            histogram[image[i,j]]+=1
    n=float(rows*cols)
    histogram=[h/n for h in histogram]
    plt.plot(histogram)
    plt.xlim([-1,260])
    plt.show()
    return histogram

def histogramEqualization (histogram):
    trans=[0]*256
    trans[0]=histogram[0]
    for i in range (1,256):
        trans[i]= histogram[i] + trans[i-1]
    trans=[t*255 for t in trans]
    trans=np.around(trans).astype('uint8')
    plt.plot(trans)
    plt.xlim([-1,260])
    plt.show()

    eq_hist=[0]*256
    for i in range(0,256):
        indices=list(np.where(trans==i)[0])
        hist = [histogram[index] for index in indices]
        eq_hist[i]=sum(hist)
   
    plt.plot(eq_hist)
    plt.xlim([-1,260])
    plt.show()
 
    return  trans, eq_hist


def matchHistogram(src_hist, dst_hist):

    strans, seq_hist=histogramEqualization (src_hist)
    dtrans, deq_hist=histogramEqualization (dst_hist)
    
    trans=list(strans)
    for i in range(0,256):
        check=strans[i]
        smallest=257
        z=check
        for j in range (0,256):
           temp=abs(dtrans[j]-check)
           if temp<smallest:
               smallest=temp
               z=j
        trans[i]=z
           
    plt.plot(trans)
    plt.xlim([-1,260])
    plt.show()    
    return trans


def equalizeHistogram(image):
    rows=int(image.shape[0])
    cols=int(image.shape[1])
    histogram=computeImageHistogram (image)
    trans, eq_hist=histogramEqualization (histogram)
    
    dst_image =np.array(image)
    for i in range (0,rows):
            for j in range (0,cols):
                dst_image[i,j] = trans[image[i,j]]   
    cv2.imwrite('eout.tif', dst_image)
    return dst_image

def matchImage(src_image, dst_hist):
    rows=int(src_image.shape[0])
    cols=int(src_image.shape[1])
    src_hist=computeImageHistogram (src_image)
    trans=matchHistogram(src_hist, dst_hist)
    dst_image =np.array(src_image)
    for i in range (0,rows):
            for j in range (0,cols):
                dst_image[i,j] = trans[src_image[i,j]]   
    cv2.imwrite('outmatched.tif', dst_image)
    
    return dst_image
    

si=cv2.imread('3161.tif',0)
di=cv2.imread('307.tif',0)
cv2.imwrite('outsource.tif', si)
cv2.imwrite('outdestin.tif', di)
dst_hist=computeImageHistogram (di)
dst_image=matchImage(si, dst_hist)





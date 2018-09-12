# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:39:21 2017

@author: B
"""

import numpy as np
import cv2
from scipy import interpolate
import math
np.set_printoptions(threshold=np.nan)



def interpol(zs,x,y):
    xs=[0,1,2,3,
        0,1,2,3,
        0,1,2,3,
        0,1,2,3]
    ys=[0,0,0,0,
        1,1,1,1,
        2,2,2,2,
        3,3,3,3]
    f = interpolate.interp2d(xs, ys, zs, kind='cubic')
    z=f(x,y)
    return z

def buildImagePyramid(image,colors):
    x,y=image.shape
    element=set()
    for i in range (0,x):        
        for j in range (0,y):
            if image[i,j] in colors:
                element.add((i,j))
    images = [image]
    elements=[element]
    
    level=1
    while(len(element)>=4):
        x,y=image.shape
        sx=x/2
        sy=y/2
        simage=np.zeros(shape=(sx,sy), dtype='uint8')
        for i in range(0,x-1,2):
            for j in range (0,y-1,2):
                simage[i/2,j/2]=(int(image[i,j])+int(image[i,j+1])+int(image[i+1,j])+int(image[i+1,j+1]))/4
         
        belement=elements[level-1]
        element=set()
        for [i,j] in belement:
            if(i/2<sx and j/2<sy):
                element.add((i/2,j/2))
   
        images.append(simage)
        elements.append(element)
        
        image=simage
        level=level+1
    return images,elements



def neighbors(image,i,j,q):
    if q==0:

      zs=[image[i+1,j-2],image[i+1,j-1],image[i+1,j],image[i+1,j+1],
          image[i,j-2],image[i,j-1],image[i,j],image[i,j+1],
          image[i-1,j-2],image[i-1,j-1],image[i-1,j],image[i-1,j+1],
          image[i-2,j-2],image[i-2,j-1],image[i-2,j],image[i-2,j+1]]
    
    if q==1:

      zs=[image[i+1,j-1],image[i+1,j],image[i+1,j+1],image[i+1,j+2],
          image[i,j-1],image[i,j],image[i,j+1],image[i,j+2],
          image[i-1,j-1],image[i-1,j],image[i-1,j+1],image[i-1,j+2],
          image[i-2,j-1],image[i-2,j],image[i-2,j+1],image[i-2,j+2]]                    

    if q==2:

      zs=[image[i+2,j-2],image[i+2,j-1],image[i+2,j],image[i+2,j+1],
    		 image[i+1,j-2],image[i+1,j-1],image[i+1,j],image[i+1,j+1],
    		 image[i,j-2],image[i,j-1],image[i,j],image[i,j+1],
    		 image[i-1,j-2],image[i-1,j-1],image[i-1,j],image[i-1,j+1]]

    if q==3:

      zs=[image[i+2,j-1],image[i+2,j],image[i+2,j+1],image[i+2,j+2],
    		 image[i+1,j-1],image[i+1,j],image[i+1,j+1],image[i+1,j+2],
    		 image[i,j-1],image[i,j],image[i,j+1],image[i,j+2],
    		 image[i-1,j-1],image[i-1,j],image[i-1,j+1],image[i-1,j+2]]
    return zs
   

def removeElements(image, colors):
    images,elements=buildImagePyramid(image,colors)
    n=len(images)
 
    for level in range(n-1,0,-1):
        
        image=images[level]
        element=elements[level]
        bimage=images[level-1]
        belement=elements[level-1]

        stop=False
        while(stop==False):
                stop=True
                for (i,j) in element:
                    
                    rimage = cv2.copyMakeBorder(image,2,2,2,2,cv2.BORDER_REPLICATE)
                    
                    zs=neighbors(rimage,i+2,j+2,0)
                    z0=int(round(np.clip(interpol(zs,1.25,1.75),0,255)))
                            
                    zs=neighbors(rimage,i+2,j+2,1)
                    z1=int(round(np.clip(interpol(zs,1.75,1.75),0,255)))
                            
                    zs=neighbors(rimage,i+2,j+2,2)
                    z2=int(round(np.clip(interpol(zs,1.25,1.25),0,255)))
                            
                    zs=neighbors(rimage,i+2,j+2,3)
                    z3=int(round(np.clip(interpol(zs,1.75,1.25),0,255)))
                         
                    zt=float(z0+z1+z2+z3)/4
                       
                    if(zt>64.0):
                       z=int(math.ceil(zt))
                    else:
                       z=int(math.floor(zt))
                       

                    
                    if rimage[i+2,j+2]!=z:
                            stop=False
                            image[i,j]=z
                            if((2*i,2*j) in belement):
                                bimage[2*i,2*j]=z
                            if((2*i,2*j+1) in belement):
                                bimage[2*i,2*j+1]=z
                            if((2*i+1,2*j) in belement):
                                bimage[2*i+1,2*j]=z
                            if((2*i+1,2*j+1) in belement):
                                bimage[2*i+1,2*j+1]=z
                    else:
                            image[i,j]=z
                            if((2*i,2*j) in belement):
                                bimage[2*i,2*j]=z
                            if((2*i,2*j+1) in belement):
                                bimage[2*i,2*j+1]=z
                            if((2*i+1,2*j) in belement):
                                bimage[2*i+1,2*j]=z
                            if((2*i+1,2*j+1) in belement):
                                bimage[2*i+1,2*j+1]=z
            

        images[level-1]=bimage
     
    image=images[0]
    element=elements[0]

    stop=False
    while(stop==False):
            stop=True
            for (i,j) in element:
                    
                rimage = cv2.copyMakeBorder(image,2,2,2,2,cv2.BORDER_REPLICATE)
     
                zs=neighbors(rimage,i+2,j+2,0)
                z0=int(round(np.clip(interpol(zs,1.25,1.75),0,255)))
                            
                zs=neighbors(rimage,i+2,j+2,1)
                z1=int(round(np.clip(interpol(zs,1.75,1.75),0,255)))
                            
                zs=neighbors(rimage,i+2,j+2,2)
                z2=int(round(np.clip(interpol(zs,1.25,1.25),0,255)))
                            
                zs=neighbors(rimage,i+2,j+2,3)
                z3=int(round(np.clip(interpol(zs,1.75,1.25),0,255)))
                
                
                zt=float(z0+z1+z2+z3)/4
                       
                if(zt>64.0):
                   z=int(math.ceil(zt))
                else:
                   z=int(math.floor(zt))
            
                if rimage[i+2,j+2]!=z:
                     stop=False
                     image[i,j]=z
                else:
                     image[i,j]=z
    	

    images[0]=image
    
    return images



#image=np.zeros((32,32))
#image[12:20,12:20]=255
#image=np.array(image,dtype='uint8')

#image=np.zeros((64,64))
#image[:,32:64]=128
#image[24:40,24:40]=255
#image=np.array(image,dtype='uint8')

image = cv2.imread('pic6.png',0)
colors=[0]
images=removeElements(image,colors)

cv2.imwrite('DIPpic6.png', images[0])



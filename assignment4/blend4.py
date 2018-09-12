import math
import numpy as np
import cv2


def kernelMaker():
  #kernel=cv2.getGaussianKernel(7,2)
  #kernel=np.outer(kernel,kernel)
  kernel=np.array([[1,2,1],[2,4,2],[1,2,1]])
  kernel=kernel/16.
  return kernel
 

def halfimage(image):
  out = None
  kernel = kernelMaker()
  #gaussian lowpass smoothing filter
  outimage= cv2.filter2D(image,-1,kernel)
  out = outimage[::2,::2]
  return out
 

def doubleimage(image):
  out = None
  kernel = kernelMaker()
  newx,newy = 2*image.shape[1],2*image.shape[0] 
  #bilinear interpolation
  out = cv2.resize(image,(newx,newy))
  out= cv2.filter2D(out,-1,kernel)
  return out
 
def gpyramid(image, levels):
  output = []
  output.append(image)
  tmp = image
  for i in range(0,levels):
    tmp = halfimage(tmp)
    output.append(tmp)
  return output
 
def lpyramid(gpyr):
  output = []
  k = len(gpyr)
  for i in range(0,k-1):
    gi = gpyr[i]
    dgi = doubleimage(gpyr[i+1])
    if dgi.shape[0] > gi.shape[0]:
       dgi = np.delete(dgi,(-1),axis=0)
    if dgi.shape[1] > gi.shape[1]:
      dgi = np.delete(dgi,(-1),axis=1)
    output.append(gi - dgi)
  output.append(gpyr.pop())
  return output

def blendpyramid(lpyra, lpyrb, gpyrm):
  blendedpyr = []
  k= len(gpyrm)
  for i in range(0,k):
   p1= gpyrm[i]*lpyra[i]
   p2=(1 - gpyrm[i])*lpyrb[i]
   blendedpyr.append(p1 + p2)
  return blendedpyr

def reconstruct(lpyr):
  output = None
  output = np.zeros((lpyr[0].shape[0],lpyr[0].shape[1]), dtype=np.float64)
  for i in range(len(lpyr)-1,0,-1):
    lap = doubleimage(lpyr[i])
    lapb = lpyr[i-1]
    if lap.shape[0] > lapb.shape[0]:
      lap = np.delete(lap,(-1),axis=0)
    if lap.shape[1] > lapb.shape[1]:
      lap = np.delete(lap,(-1),axis=1)
    tmp = lap + lapb
    lpyr.pop()
    lpyr.pop()
    lpyr.append(tmp)
    output = tmp
  return output

def blend (base_img, blend_img,x,y):


    mask = blend_img
    ret, otsu = cv2.threshold(mask,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    mask = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, se)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
    cv2.imwrite('mask.png',mask)
    
    i=np.where(mask==255)[0][0]
    j=np.where(mask==255)[1][0]
    
    h,w=base_img.shape

    if (x>=i and y>=j):
        newimage1=np.zeros([h,w])
        newimage1[0:h,0:w]=base_img
        newimage2=np.zeros([h,w])
        newimage2[x-i:,y-j:]=blend_img[0:h-(x-i),0:w-(y-j)]
        newimage3=np.zeros([h,w])
        newimage3[x-i:,y-j:]=mask[0:h-(x-i),0:w-(y-j)]
    if (x<i and y<j):
        newimage1=np.zeros([h,w])
        newimage1[0:h,0:w]=base_img
        newimage2=np.zeros([h,w])
        newimage2[0:h-(i-x),0:w-(j-y)]=blend_img[i-x:,j-y:]
        newimage3=np.zeros([h,w])
        newimage3[0:h-(i-x),0:w-(j-y)]=mask[i-x:,j-y:]
    if (x>=i and y<j):
        newimage1=np.zeros([h,w])
        newimage1[0:h,0:w]=base_img
        newimage2=np.zeros([h,w])
        newimage2[x-i:,0:w-(j-y)]=blend_img[0:h-(x-i),j-y:]
        newimage3=np.zeros([h,w])
        newimage3[x-i:,0:w-(j-y)]=mask[0:h-(x-i),j-y:]
    if (x<i and y>=j):
        newimage1=np.zeros([h,w])
        newimage1[0:h,0:w]=base_img
        newimage2=np.zeros([h,w])
        newimage2[0:h-(i-x),y-j:]=blend_img[i-x:,0:w-(y-j)]
        newimage3=np.zeros([h,w])
        newimage3[0:h-(i-x),y-j:]=mask[i-x:,0:w-(y-j)]  
    
    newimage1=newimage1.astype(float)
    newimage2=newimage2.astype(float)
    newimage3=newimage3/255.
    
    minsize = min(base_img.shape)
    depth = int(math.floor(math.log(minsize, 2))) - 4 
    
    gpyrm = gpyramid(newimage3, depth)
    
    gpyrimage1 = gpyramid(newimage1, depth)
        
    gpyrimage2 = gpyramid(newimage2, depth)
        
    lpyrimage1  = lpyramid(gpyrimage1)
        
    lpyrimage2 = lpyramid(gpyrimage2)
        
    outimg = reconstruct(blendpyramid(lpyrimage2, lpyrimage1, gpyrm))
    
    outimg[outimg < 0] = 0
    outimg[outimg > 255] = 255
    outimg = outimg.astype(np.uint8)
    
    cv2.imwrite('blended.png', outimg)



image1 = cv2.imread('desert.png',0)
image2 = cv2.imread('pyramid.png',0)

blend(image1,image2,70,308)

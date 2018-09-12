# -*- coding: utf-8 -*-
"""
Created on Thu May 25 23:22:59 2017

@author: B
"""
from math import log, ceil,sqrt
import numpy as np
import cv2
from scipy import ndimage
from scipy.misc import imsave,imread

def dft(f):
    N=f.shape[0]
    x=np.arange(N)
    u=x.reshape((N,1))
    W=np.exp(-2j*np.pi*u*x/N)
    return(np.dot(W,f))

f=np.random.rand(4)
F=dft(f)
oF=np.fft.fft(f)
print(np.allclose(F,oF))

def idft(F):
    # Transform again
    Ft =dft(F)
    # Reverse and normalize
    f = np.divide(Ft[-np.arange(Ft.shape[0])],F.shape[-1])
    return f

f=idft(F)
of=np.fft.ifft(F)
print(np.allclose(f,of))

def dft2(f):
    (m,n)=f.shape
    Fh=np.zeros(shape=(m,n),dtype=complex)
    for i in range(m):
        Fh[i]=dft(f[i,:])
   
    Fv=np.zeros(shape=(m,n),dtype=complex)
    for i in range(n):
        Fv[:,i]=dft(Fh[:,i])
    return Fv

f=np.random.rand(4,4)
F=dft2(f)
oF=np.fft.fft2(f)
print(np.allclose(F,oF))

def idft2(F):
    (m,n)=F.shape
    Fv=np.zeros(shape=(m,n),dtype=complex)
    for i in range(n):
        Fv[:,i]=idft(F[:,i])
   
    Fh=np.zeros(shape=(m,n))
    for i in range(m):
        Fh[i,:]=idft(Fv[i,:])
    return Fh

f=idft2(F)
of=np.fft.ifft2(F)
print(np.allclose(f,of))

def fft(f):

    N=f.shape[0]
    if N & (N - 1) <> 0 and N>8:
        raise ValueError("length must be a power of 2")
    if N <= 8:
        return dft(f)
    else:
        F_even = fft(f[::2])
        F_odd = fft(f[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([F_even + factor[:N / 2] * F_odd,
                               F_even + factor[N / 2:] * F_odd])

f=np.random.rand(4)
F=fft(f)
oF=np.fft.fft(f)
print(np.allclose(F,oF))

def ifft(F):
    # Transform again
    Ft =fft(F)
    # Reverse and normalize
    f = np.divide(Ft[-np.arange(Ft.shape[0])],F.shape[-1])
    return f

f=ifft(F)
of=np.fft.ifft(F)
print(np.allclose(f,of))

def fftpad(f):
    m, n = f.shape
    M, N = 2 ** int(ceil(log(m, 2))), 2 ** int(ceil(log(n, 2)))
    pf =np.zeros(shape=(M,N))
    for i in range(0, m):
        for j in range(0, n):
            pf[i][j] = f[i][j]
    return pf


def fft2(f):
    (m,n)=f.shape
    Fh=np.zeros(shape=(m,n),dtype=complex)
    for i in range(m):
        Fh[i]=fft(f[i,:])
   
    Fv=np.zeros(shape=(m,n),dtype=complex)
    for i in range(n):
        Fv[:,i]=fft(Fh[:,i])
    return Fv

f=np.random.rand(4,4)
F=fft2(f)
oF=np.fft.fft2(f)
print(np.allclose(F,oF))

def ifft2(F):
    (m,n)=F.shape
    Fv=np.zeros(shape=(m,n),dtype=complex)
    for i in range(n):
        Fv[:,i]=ifft(F[:,i])
   
    Fh=np.zeros(shape=(m,n))
    for i in range(m):
        Fh[i,:]=ifft(Fv[i,:])
    return Fh

f=ifft2(F)
of=np.fft.ifft2(F)
print(np.allclose(f,of))

f=imread('dip2.png',flatten=True)
pf=fftpad(f)
pF=fft2(pf)
pf=ifft2(pF)
imsave('dip2padded.png', pf)

def convolve(image, filt):
    m, n = filt.shape
    r=m/2
    rimage = cv2.copyMakeBorder(image,r,r,r,r,cv2.BORDER_REPLICATE)

    if (m == n):
        y, x = rimage.shape
        y = y - m + 1
        x = x - m + 1
        cimage = np.zeros((y,x)).astype(np.float)
        for i in range(y):
            for j in range(x):
                cimage[i][j] = np.sum(rimage[i:i+m, j:j+m]*filt)
    return cimage


#image = imread('dip2.png',flatten=True)
#sobel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#simage=convolve(image,sobel)
#imsave('dip2sobel.png', simage)

image = imread('dip2.png',flatten=True)
image = image.astype(np.float)
ocimage = ndimage.sobel(image, 1)
imsave('dip2convolve.png', ocimage)

def pad(image):
    m,n=image.shape
    pimage=np.zeros((2*m,2*n))
    pimage[0:m,0:n]=image
    return pimage

def unpad(pimage):
    p,q=pimage.shape
    image=pimage[0:p/2,0:q/2]
    return image

def center(image):
    m,n=image.shape
    cimage=np.zeros((m,n))
    for i in range (m):
        for j in range (n):
            cimage[i,j]=((-1)**(i+j))*image[i,j]
    return cimage

def ILP(image,d):
    
    pimage=pad(image)
    
    cpimage=center(pimage)

    cpF = fft2(cpimage)
    
    rows, cols = pimage.shape
    crow, ccol = rows/2 , cols/2
    
    ilpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            if sqrt((i-crow)**2+(j-ccol)**2)<d:
                ilpf[i,j]=1

    #imsave('ilpfFrequency.png',ilpf)
 
    rows, cols = image.shape
    crow,ccol = rows/2 , cols/2
    
    silpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            if sqrt((i-crow)**2+(j-ccol)**2)<d:
                silpf[i,j]=1
                     
    #imsave('silpfFrequency.png',silpf)


    cilpf=center(silpf)
    cilpff = ifft2(cilpf)
    ilpff=center(cilpff)

    #imsave('ilpfSpatial.png', ilpff)

    sf=ilpff[243:270,243:270]
    #imsave('silpfSpatial.png', sf)

    nsf=sf/np.sum(sf)
    convolvedImage=convolve(image,nsf)
    imsave('dip1ILPSpatial.png', convolvedImage)

    
    fcpF=cpF*ilpf
    
    fcpf = ifft2(fcpF)
   
    fpf=center(fcpf)
    
    fpf = np.abs(fpf)
    
    ff=unpad(fpf)

    imsave('dip1ILPFrequency.png', ff)


def IHP(image,d):
    
    pimage=pad(image)
    
    cpimage=center(pimage)

    cpF = fft2(cpimage)
    
    rows, cols = pimage.shape
    crow,ccol = rows/2 , cols/2
    
    ihpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            if sqrt((i-crow)**2+(j-ccol)**2)>d:
                ihpf[i,j]=1

    #imsave('ihpfFrequency.png',ihpf)

    
    rows, cols = image.shape
    crow,ccol = rows/2 , cols/2
    
    sihpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            if sqrt((i-crow)**2+(j-ccol)**2)>d:
                sihpf[i,j]=1
    
    #imsave('sihpfFrequency.png',sihpf)


    cihpf=center(sihpf)
    cihpff = ifft2(cihpf)
    ihpff=center(cihpff)
    #imsave('ihpfSpatial.png', ihpff)
    
    sf=ihpff[243:270,243:270]
    #imsave('sihpfSpatial.png', sf)

    convolvedImage=convolve(image,sf)
    cv2.imwrite('dip1IHPSpatial.png', convolvedImage)

    fcpF=cpF*ihpf
    
    fcpf = ifft2(fcpF)
   
    fpf=center(fcpf)
    
    fpf = np.abs(fpf)
    
    ff=unpad(fpf)

    imsave('dip1IHPFrequency.png', ff)
    
    
   

def BLP(image,d):
    
    pimage=pad(image)
    
    cpimage=center(pimage)

    cpF = fft2(cpimage)
    
    rows, cols = pimage.shape
    crow,ccol = rows/2 , cols/2
    
    blpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            blpf[i,j]=1./(1+(sqrt((i-crow)**2+(j-ccol)**2)/d)**4)

    #imsave('blpfFrequency.png',blpf)
 
    rows, cols = image.shape
    crow,ccol = rows/2 , cols/2
    
    sblpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            sblpf[i,j]=1./(1+(sqrt((i-crow)**2+(j-ccol)**2)/d)**4)

    #imsave('sblpfFrequency.png', sblpf)

    cblpf=center(sblpf)
    cblpff = ifft2(cblpf)
    blpff=center(cblpff)
    
    #imsave('blpfSpatial.png', blpff)

    sf=blpff[243:270,243:270]
    #imsave('sblpfSpatial.png', sf)

    nsf=sf/np.sum(sf)
    convolvedImage=convolve(image,nsf)
    imsave('dip1BLPSpatial.png', convolvedImage)


    fcpF=cpF*blpf
    
    fcpf = ifft2(fcpF)
   
    fpf=center(fcpf)
    
    fpf = np.abs(fpf)
    
    ff=unpad(fpf)

    imsave('dip1BLPFrequency.png', ff)

def BHP(image,d):
    
    pimage=pad(image)
    
    cpimage=center(pimage)

    cpF = fft2(cpimage)
    
    rows, cols = pimage.shape
    crow,ccol = rows/2 , cols/2
    
    bhpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            if(i<>crow or j<>ccol):
                bhpf[i,j]=1./(1+(d/sqrt((i-crow)**2+(j-ccol)**2))**4)
    
    
    
    #imsave('bhpfFrequency.png',bhpf)
 
    rows, cols = image.shape
    crow,ccol = rows/2 , cols/2
    
    sbhpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            if(i<>crow or j<>ccol):
                sbhpf[i,j]=1./(1+(d/sqrt((i-crow)**2+(j-ccol)**2))**4)
    #imsave('sbhpfFrequency.png',sbhpf)

    cbhpf=center(sbhpf)
    cbhpff = ifft2(cbhpf)
    bhpff=center(cbhpff)
    #imsave('bhpfSpatial.png',bhpff)

    sf=bhpff[243:270,243:270]
    #imsave('sbhpfSpatial.png', sf)

    convolvedImage=convolve(image,sf)
    cv2.imwrite('dip1BHPSpatial.png', convolvedImage)


    fcpF=cpF*bhpf
    
    fcpf = ifft2(fcpF)
   
    fpf=center(fcpf)
    
    fpf = np.abs(fpf)
    
    ff=unpad(fpf)

    imsave('dip1BHPFrequency.png', ff)

def GLP(image,d):
    
    pimage=pad(image)
    
    cpimage=center(pimage)

    cpF = fft2(cpimage)
    
    rows, cols = pimage.shape
    crow,ccol = rows/2 , cols/2
    
    glpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            glpf[i,j]=np.exp(-((i-crow)**2+(j-ccol)**2)/(2*d**2))


    #imsave('glpfFrequency.png',glpf)
 
    rows, cols = image.shape
    crow,ccol = rows/2 , cols/2
    
    sglpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            sglpf[i,j]=np.exp(-((i-crow)**2+(j-ccol)**2)/(2*d**2))
    
    #imsave('sglpfFrequency.png',sglpf)

    cglpf=center(sglpf)
    cglpff = ifft2(cglpf)
    glpff=center(cglpff)
    
    #imsave('glpfSpatial.png', glpff)

    sf=glpff[243:270,243:270]
    #imsave('sglpfSpatial.png', sf)

    nsf=sf/np.sum(sf)
    convolvedImage=convolve(image,nsf)
    imsave('dip1GLPSpatial.png', convolvedImage)

    fcpF=cpF*glpf
    
    fcpf = ifft2(fcpF)
   
    fpf=center(fcpf)
    
    fpf = np.abs(fpf)
    
    ff=unpad(fpf)

    imsave('dip1GLPFrequency.png', ff)

def GHP(image,d):
    
    pimage=pad(image)
    
    cpimage=center(pimage)

    cpF = fft2(cpimage)
    
    rows, cols = pimage.shape
    crow,ccol = rows/2 , cols/2
    
    ghpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            ghpf[i,j]=1-np.exp(-((i-crow)**2+(j-ccol)**2)/(2*d**2))
      
    
    #imsave('ghpfFrequency.png',ghpf)
 
    rows, cols = image.shape
    crow,ccol = rows/2 , cols/2
    
    sghpf = np.zeros((rows,cols))
    for i in range (rows):
        for j in range (cols):
            sghpf[i,j]=1-np.exp(-((i-crow)**2+(j-ccol)**2)/(2*d**2))
      
    #imsave('sghpfFrequency.png',sghpf)

    cghpf=center(sghpf)
    cghpff = ifft2(cghpf)
    ghpff=center(cghpff)
       
    #imsave('ghpfSpatial.png',ghpff)

    sf=ghpff[243:270,243:270]
    #imsave('sghpfSpatial.png', sf)

    convolvedImage=convolve(image,sf)
    cv2.imwrite('dip1GHPSpatial.png', convolvedImage)
    

    fcpF=cpF*ghpf
    
    fcpf = ifft2(fcpF)
   
    fpf=center(fcpf)
    
    fpf = np.abs(fpf)
    
    ff=unpad(fpf)

    imsave('dip1GHPFrequency.png', ff)



def sample(sf):
    m,n=sf.shape
    c=m/2
    down=c-40
    up=c+41
    ssf=sf[down:up,down:up]
    sample=np.zeros((27,27))
    for i in range(0,81,3):
        for j in range(0,81,3):
            sample[i/3,j/3]=ssf[i,j]
            

image = imread('dip1.png',flatten=True)
d=60.
pimage=fftpad(image)
#IHP(pimage,d)
#ILP(pimage,d)
#BHP(pimage,d)
#BLP(pimage,d)
#GHP(pimage,d)
#GLP(pimage,d)
















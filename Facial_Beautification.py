# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:31:31 2018

@author: Zehao Huang
"""
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
t=1
        
def image_show_plt(img):
    b,g,r=cv.split(img)
    img2=cv.merge([r,g,b])
    plt.imshow(img2)
    plt.show()
    
def my_bilateralFilter(img,d,sigma_d,sigma_c):
    r,c,k = img.shape
    B=np.zeros([r,c,3])
    for row in range(r):
        for col in range(c):
            rmin=max(row-d,1)
            rmax=min(row+d,r)
            cmin=max(col-d,1)
            cmax=min(col+d,c)
            A=img[rmin:rmax,cmin:cmax,:]
            
            db=A[:,:,0]-img[row,col,0]
            dg=A[:,:,1]-img[row,col,1]
            dr=A[:,:,2]-img[row,col,2]
            H=np.exp(-(np.square(db)+np.square(dg)+np.square(dr))/(2*np.square(sigma_c)))
     
            G=np.zeros(H.shape,dtype=np.uint8)
            G_x=G
            G_y=G
            a=0
            b=0
            for x in range(rmin-row+d+1,rmax-row+d+1):
                G_x[a,:]=x
                a=a+1
            for y in range(cmin-col+d+1,cmax-col+d+1):
                G_y[:,b]=y
                b=b+1
            G=np.exp(-(np.square(G_x)+np.square(G_y))/(2*np.square(sigma_c)))
            F=np.multiply(H,G)
            norm_F=sum(sum(F))
            B[row,col,0]=sum(sum(np.multiply(F,A[:,:,0])))/norm_F
            B[row,col,1]=sum(sum(np.multiply(F,A[:,:,1])))/norm_F
            B[row,col,2]=sum(sum(np.multiply(F,A[:,:,2])))/norm_F
    return B

def image_show_cv(name1,img1,name2,img2):
    cv.namedWindow(name1)
    cv.imshow(name1,img1)
    cv.namedWindow(name2)
    cv.imshow(name2,img2)
    print('tap any key to exit')
    cv.waitKey(0)
    cv.destroyAllWindows()
    
img = cv.imread("example.jpg")
img2 = my_bilateralFilter(img, 15, 37.5, 37.5)
img3 = img2 - img - 128
img4 = cv.GaussianBlur(img3,(1,1),0)
img5 = img + 2 * img4 - 255
img5 = img5.astype('uint8')
dst = cv.addWeighted(img,0.5,img5, 0.5, 10)

image_show_cv('original image',img,'beautified image',dst)

"""plt.figure(1)
plt.axis('off')
image_show_plt(img)
plt.figure(2)
plt.axis('off')
image_show_plt(dst)"""
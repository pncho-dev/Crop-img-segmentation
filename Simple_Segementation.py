# -*- coding: utf-8 -*-
"""
This file contains  simple approaches that used to solve the problem, it only
uses the color of the images to segmentate the rice crops:
1. The most basic segmentation algortihm that was tuned manually and used as a learning excrsise.
2. The  implementation of K means used to segmentate the image by color
"""

import cv2 as cv
import numpy as np




img1 = cv.imread("Your path to the images")

hsv_img1 = cv.cvtColor(img1, cv.COLOR_RGB2HSV)#convertir la imagen

light_green = np.array([49, 35,35 ])
dark_green = np.array([80, 255,255])

mask = cv.inRange(hsv_img1, light_green, dark_green)


#Mostrar las imágenes
cv.imshow("Imagen original", img1)
cv.imshow("Mask", mask)
cv.imwrite('mask_img1_Result.jpg',mask)

cv.waitKey()
cv.destroyAllWindows()

def K_Means(img, K):# This function returns the segmentated image, using the image and the number of clusters 
    img1=img
    img1 = cv.bilateralFilter(img,9,75,75)#preliminar filter
    img_reshape = img1.reshape((-1,3))#resizing the image
    img_reshape_fl=np.float32(img_reshape)#converting the image
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)#defining the K-Means parameters
    K = K#Number of clusters
    ret,label,center=cv.kmeans(img_reshape_fl,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)#aplying K-means
    center = np.uint8(center)#converting each center cluster
    res = center[label.flatten()]#flattening the array
    img_kmeans = res.reshape((img1.shape))#reshaping the image
    ret,mask = cv.threshold(img_kmeans[:,:,0],100,255,cv.THRESH_BINARY_INV)#segmentating the image with umbralization
    size=(15,15)#Kernel size
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,size)
    mask_open = cv.morphologyEx(mask,cv.MORPH_OPEN, kernel)#Using morphology to reduce noise
    kernel_cl=np.ones((10,10),np.uint8)#Rectangular Kernel
    mask_close = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernel_cl)
    result = cv.bitwise_and(img, img,mask = mask_close)#apliying the mask
    return result
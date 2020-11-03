# -*- coding: utf-8 -*-
"""
This is the most basic segmentation algortihm it was tuned manually so you shouldn't 
expect high performance. It only uses the supposed color range of the canopy of rice 
crops with a simple mask.
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

import cv2
import matplotlib.pyplot as plt
import numpy as np
from metric import *

img_dir = "data/img.tif"
gt_dir = "data/gt.tif"
img = cv2.imread(img_dir)
gt = cv2.imread(gt_dir)
ret,thresh = cv2.threshold(gt[:,:,0],127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = [i+1 for i in contours]
# cv2.drawContours(img, contours, -1, (0,255,0), 2)
# fill_img = np.zeros_like(img)
# cv2.drawContours(fill_img, contours, -1, (255,255,255), cv2.FILLED)
# cv2.drawContours(fill_img, contours, -1, (0,255,0), 2)
# f, ax = plt.subplots(1,2)
# ax[0].imshow(img)
# ax[1].imshow(fill_img)
# plt.show()

def fillContour(ref_img, contours):
    mask = np.zeros_like(ref_img, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (255,255,255), cv2.FILLED)
    return mask
ref = gt[:,:,0]
pred = fillContour(img, contours)[:,:,0]
print("dc: ", dc(ref, pred))
print("assd: ", assd(ref, pred))
print("hd: ", hd(ref, pred))
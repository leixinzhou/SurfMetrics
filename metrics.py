import cv2
import matplotlib.pyplot as plt

img_dir = "data/img.tif"
gt_dir = "data/gt.tif"
img = cv2.imread(img_dir)
imgray = img[:,:,0]
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
plt.imshow(img)
plt.show()
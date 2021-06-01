import cv2

img16 = cv2.imread('./man_seg001.tif')
img8 = (img16/256).astype('uint8')
cv2.imwrite('./tmp.tif',img8)
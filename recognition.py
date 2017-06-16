import numpy as np
import cv2
import matplotlib.image as mpimg
import camera
import pickle
from image_processor import ImageProcessor
import os.path
import matplotlib.pyplot as plt
from line import Line

if os.path.exists(camera.CAL_FILE_NAME): 
    (mtx, dist) = pickle.load(open( "cal_data.p", "rb" ) )
else:
    mtx, dist = camera.calibrate_camera(9,6)
    pickle.dump( (mtx, dist), open( "cal_data.p", "wb" ) )

src = [(590, 460), (720, 460), (1100, 690), (260, 690)]
dst = [(260, 80), (1000, 80), (1000, 700), (260, 700)]
M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))

img = mpimg.imread('test_images/test3.jpg')

left = Line()
right = Line()
proc = ImageProcessor(mtx, dist, M)
proc.process_image(img, left, right)

cv2.line(img, src[0], src[1], (255,0,0), 5)
cv2.line(img, src[1], src[2], (255,0,0), 5)
cv2.line(img, src[2], src[3], (255,0,0), 5)
cv2.line(img, src[3], src[0], (255,0,0), 5)

# cv2.line(warped, dst[0], dst[1], (255,0,0), 5)
# cv2.line(warped, dst[1], dst[2], (255,0,0), 5)
# cv2.line(warped, dst[2], dst[3], (255,0,0), 5)
# cv2.line(warped, dst[3], dst[0], (255,0,0), 5)


fig = plt.figure()
a=fig.add_subplot(2,2,1)
plt.imshow(img)
a=fig.add_subplot(2,2,2)
plt.imshow(proc.warped)
a=fig.add_subplot(2,2,3)
plt.imshow(proc.binary_warped, cmap='gray')

a=fig.add_subplot(2,2,4)
plt.imshow(proc.out_img)
plt.plot(proc.left_fitx, proc.ploty, color='yellow')
plt.plot(proc.right_fitx, proc.ploty, color='yellow')

fig.show()
input()
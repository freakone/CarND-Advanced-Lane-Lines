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

fig = plt.figure()
a=fig.add_subplot(2,3,1)
plt.imshow(img)
a=fig.add_subplot(2,3,2)
plt.imshow(proc.warped)
a=fig.add_subplot(2,3,3)
plt.imshow(proc.binary_warped, cmap='gray')

a=fig.add_subplot(2,3,4)
plt.imshow(proc.out_img)
plt.plot(left.fitx, left.ploty, color='yellow')
plt.plot(right.fitx, left.ploty, color='yellow')

a=fig.add_subplot(2,3,5)
plt.imshow(proc.overlay_route(img, left, right))

fig.show()
input()
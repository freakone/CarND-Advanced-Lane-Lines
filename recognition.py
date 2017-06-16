import numpy as np
import cv2
import matplotlib.image as mpimg
import camera
import pickle
from image_processor import ImageProcessor
import os.path
import matplotlib.pyplot as plt
from line import Line
from moviepy.editor import VideoFileClip

if os.path.exists(camera.CAL_FILE_NAME): 
    (mtx, dist) = pickle.load(open( "cal_data.p", "rb" ) )
else:
    mtx, dist = camera.calibrate_camera(9,6)
    pickle.dump( (mtx, dist), open( "cal_data.p", "wb" ) )

src = [(600, 460), (710, 460), (1100, 690), (260, 690)]
dst = [(260, 80), (1000, 80), (1000, 700), (260, 700)]
M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))



left = Line()
right = Line()
proc = ImageProcessor(mtx, dist, M)

def process_image(img):
    proc.process_image(img, left, right)
    return proc.overlay_route(img, left, right)

img = mpimg.imread('test_images/vlc.png')
proc.process_image(img, left, right)
overlayed = proc.overlay_route(img, left, right)

fig = plt.figure()
a=fig.add_subplot(4,2,1)
plt.imshow(img)
a=fig.add_subplot(4,2,2)
plt.imshow(proc.warped)

a=fig.add_subplot(4,2,3)
plt.imshow(proc.s_binary, cmap='gray')

a=fig.add_subplot(4,2,4)
plt.imshow(proc.sxbinary, cmap='gray')

a=fig.add_subplot(4,2,5)
plt.imshow(proc.binary_warped, cmap='gray')

a=fig.add_subplot(4,2,6)
plt.imshow(proc.out_img)
plt.plot(left.fitx, left.ploty, color='yellow')
plt.plot(right.fitx, left.ploty, color='yellow')

a=fig.add_subplot(4,2,7)
plt.imshow(proc.warp_fill)

a=fig.add_subplot(4,2,8)
plt.imshow(overlayed)

fig.show()
input()

# white_output = 'output_images/project_video.mp4'
# clip1 = VideoFileClip("materials/project_video.mp4") 
# white_clip = clip1.fl_image(process_image)
# white_clip.write_videofile(white_output, audio=False)
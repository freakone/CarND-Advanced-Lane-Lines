## **Advanced Lane Finding Project**
### Kamil Górski

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

--

### Camera Calibration

#### 1. Briefly, state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in file `camera.py`.

At the beginning, the `objp` variable was defined to store the chessboard corners in global coordinates.

I'm loading the list of calibration images from the `camera_cal` folder.
For each of those images the following steps are performed:
* load file
* convert to grayscale
* use `findChessboardCorners` function to retrieve the corners.
* if the corners are successfully retrieved add them to global list

With obtained corners list the `calibrateCamera` function can be called.
Below there is an example of the undistorted image.

![]('./output_images/figure_6.png')

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

![]('./output_images/figure_7.png')

Image is undistorted by `cv2.undistort` function. File `image_processor.py` line 54.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 17 through 41 in `image_processor.py`). The image is converted to HSV color space. S channel is used for color threshold, V channel is used for gradient threshold.

![]('./output_images/figure_8.png')

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transform matrix is calculated in `recognition.py` file in lines 18-20.

`src` and `dst` points were mapped manually on images. Then the transformation is performed in `image_processor.py` file (line 55).

![]('./output_images/figure_1.png')

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To detect the lines on warped and filtered image the histogram with detection window was used. The moving windows have been applied to the both halves of the picture, then the peaks in histogram were detected, and indices for peaks was saved to the array.
For the next frame of the video, the histogram peaks are being searched near previously detected lines (without moving window) to increase performance. These steps are performed in `image_processor.py` lines 58-120.

Then for each half of the picture is being processed in function `calculate_fit` in `line.py` file. A polyfit function is used for the function approximation for detected points.

![]('./output_images/figure_4.png')

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to the center.

I did this in lines 33 through 35 in my code in `line.py`

The polynomial of the detected points has been recalculated with coefficients to scale the pixel to the real meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 43 through 51 in my code in `image_processor.py` in the function `overlay_route()`.  Here is an example of my result on a test image:

![]('./output_images/figure_9.png')

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

All steps of the process you can see in the image below.

![]('./output_images/figure_5.png')

To ensure that the detection process will be flawless and continuous few treatments were done.

1. Window line detection is skipped only if the line was successfully detected.
2. The line is detected correctly if has a curvature greater than 400m and lower than 1500m.
3. The lines are detected correctly if the difference of the curvature is lower than 20% of their average value.
4. The curvature and pixel polynomial have been filtered with moving average with the window size of 20.
5. If the polynomial cannot be calculated the measurement is being skipped.
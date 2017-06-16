import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class ImageProcessor():
    def __init__(self, mtx, dist, M):
        self.mtx = mtx
        self.dist = dist
        self.M = M
        self.Minv = np.linalg.inv(M)

        self.warped = []
        self.binary_warped = []
        self.out_img = []

    def hls_select(self, img, thresh_color=(0, 255), tresh_sobel=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        s_channel = hls[:,:,2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel > thresh_color[0]) & (s_channel <= thresh_color[1])] = 1

        self.s_binary = s_binary

        sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= tresh_sobel[0]) & (scaled_sobel <= tresh_sobel[1])] = 1
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        self.sxbinary = sxbinary
        self.combined_binary = combined_binary
        return combined_binary

    def overlay_route(self, img, left, right):
        warp_fill = np.copy(self.warped)
        pts_left = np.array([np.transpose(np.vstack([left.fitx, left.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right.fitx, right.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(warp_fill, np.int_([pts]), (255, 0, 0))
        self.warp_fill = warp_fill
        newwarp = cv2.warpPerspective(warp_fill, self.Minv, (img.shape[1], img.shape[0])) 
        return cv2.addWeighted(img, 1, newwarp, 0.5, 0)

    def process_image(self, img, left, right, thresh_color=(120, 230), tresh_sobel=(20, 100)):
        img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        self.warped = cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
        self.binary_warped = self.hls_select(self.warped, thresh_color, tresh_sobel)

        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//2:,:], axis=0)
        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        if not left.detected or not right.detected:
            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(self.binary_warped.shape[0]/nwindows)
            # Create empty lists to receive left and right lane pixel indices
            left.lane_inds = []
            right.lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
                win_y_high = self.binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left.lane_inds.append(good_left_inds)
                right.lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left.lane_inds = np.concatenate(left.lane_inds)
            right.lane_inds = np.concatenate(right.lane_inds)
        else:
            left.lane_inds = ((nonzerox > (left.fit[0]*(nonzeroy**2) + left.fit[1]*nonzeroy + left.fit[2] - margin)) & (nonzerox < (left.fit[0]*(nonzeroy**2) + left.fit[1]*nonzeroy + left.fit[2] + margin))) 
            right.lane_inds = ((nonzerox > (right.fit[0]*(nonzeroy**2) + right.fit[1]*nonzeroy + right.fit[2] - margin)) & (nonzerox < (right.fit[0]*(nonzeroy**2) + right.fit[1]*nonzeroy + right.fit[2] + margin)))  

        # Extract left and right line pixel positions
        leftx = nonzerox[left.lane_inds]
        lefty = nonzeroy[left.lane_inds] 
        rightx = nonzerox[right.lane_inds]
        righty = nonzeroy[right.lane_inds] 

        # Fit a second order polynomial to each
        left.fit = np.polyfit(lefty, leftx, 2)
        right.fit = np.polyfit(righty, rightx, 2)

        left.ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )
        right.ploty = left.ploty
        left.fitx = left.fit[0]*left.ploty**2 + left.fit[1]*left.ploty + left.fit[2]
        right.fitx = right.fit[0]*left.ploty**2 + right.fit[1]*left.ploty + right.fit[2]

        out_img[nonzeroy[left.lane_inds], nonzerox[left.lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right.lane_inds], nonzerox[right.lane_inds]] = [0, 0, 255]

        y_eval = np.max(left.ploty)
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        left.radius_of_curvature = left_curverad
        right.radius_of_curvature = right_curverad

        self.out_img = out_img

        left.detected = True
        right.detected = True

        if left.radius_of_curvature < 500 or left.radius_of_curvature > 1500:
            left.detected = False
        
        if right.radius_of_curvature < 500 or right.radius_of_curvature > 1500:
            right.detected = False

        return out_img
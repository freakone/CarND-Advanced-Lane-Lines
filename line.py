import numpy as np

class Line():
    def __init__(self):
        self.shape = (0,0)
        
        self.detected = False       
        self.line_base_pos = None 
        self.radius_of_curvature = None 
        self.lane_inds = []
        self.fitx = [] #equation for pixels y(x)
        
        self.fit_pix = [] #polynomal history in pixels
        self.fit = [] #polynomal history in meters
        self.ploty = []
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension

    def add_history(self, arr, item, maximum):
        arr.append(item)
        if len(arr) > maximum:
            arr.pop(0)
        return arr

    def get_from_history(self, arr):
        return sum(arr) / len(arr)

    def calculate_fit(self, nonzero):
        x = np.array(nonzero[1])[self.lane_inds]
        y = np.array(nonzero[0])[self.lane_inds]
        y_eval = np.max(self.ploty)

        self.fit = self.add_history(self.fit, np.polyfit(y*self.ym_per_pix, x*self.xm_per_pix, 2), 20)
        fix = self.get_from_history(self.fit)
        self.radius_of_curvature = ((1 + (2*fix[0]*y_eval*self.ym_per_pix + fix[1])**2)**1.5) / np.absolute(2*fix[0])

        self.fit_pix = self.add_history(self.fit_pix, np.polyfit(y, x, 2), 20)
        pix = self.get_from_history(self.fit_pix)

        self.fitx = pix[0]*self.ploty**2 + pix[1]*self.ploty + pix[2]
        self.line_base_pos = pix[0]*y_eval**2 + pix[1]*y_eval + pix[2]

        self.detected = True

        if self.radius_of_curvature < 400 or self.radius_of_curvature > 1500:
            self.detected = False
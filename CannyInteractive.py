from __future__ import print_function
import cv2 as cv
import numpy as np

class CannyInteractive:

    def __init__(self, image):
        self.img = image
        self.gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        self.max_lowThreshold = 100
        self.window_name = 'Edge Map'
        self.title_trackbar = 'Min Threshold:'
        self.title_trackbar2 = 'Canny Kern Size:'
        self.ratio = 3
        self.low_threshold = 0
        self.canny_ksize = 3
        # added low-pass, highpass attributes for bandpass filtering
        self.lowpass = 0
        self.highpass = 0
        self.hough_threshold = 1003

        # additional trackbars for bandpass, Hough transform
        self.title_trackbar3 = 'Low-pass Filter'
        self.title_trackbar4 = 'Highpass Filter'
        self.title_trackbar5 = 'Hough Transform'

        cv.namedWindow (self.window_name)
        # thresholding trackbar
        cv.createTrackbar(self.title_trackbar, self.window_name, 0, self.max_lowThreshold, self._canny_threshold)
        # kernel size trackbar
        cv.createTrackbar(self.title_trackbar2, self.window_name, 0, 2, self._cannyKsize)
        # bandpass filter trackbars
        cv.createTrackbar(self.title_trackbar3, self.window_name, 1, 15, self.bandpass)
        cv.createTrackbar(self.title_trackbar4, self.window_name, 1, 15, self.bandpass)
        # Hough line transform trackbar
        cv.createTrackbar(self.title_trackbar5, self.window_name, 0, 100, self.hough)

        self._canny_threshold(0)

    def _update_window(self, img):
        cv.imshow(self.window_name, img)

    def blur(self, img):
        img_blur = cv.GaussianBlur(img, (25, 25), 0)
        return img_blur

    def _canny_threshold(self, val):
        self.low_threshold = val
        upper_threshold = self.low_threshold * self.ratio
        img_blur = self.blur(self.gray)
        detected_edges = cv.Canny(img_blur, self.low_threshold, upper_threshold, apertureSize=self.canny_ksize)
        mask = detected_edges != 0
        dst = self.img * (mask[:, :, None].astype(self.img.dtype))

        # copy bandpass to ensure filtering
        self.lowpass = (2 * val) + 1  # low-pass filter threshold
        self.highpass = (4 * val) + 1  # highpass filter threshold
        # create Gaussian filter with low variance (high freq can pass)
        blur_high_pass = cv.GaussianBlur(dst, (self.lowpass, self.lowpass), 0)
        # create Gaussian filter with high variance (low freq can pass)
        blur_low_pass = cv.GaussianBlur(dst, (self.highpass, self.highpass), 0)
        # subtract (low pass - high pass) filters to create bandpass filter
        dst = blur_low_pass - blur_high_pass

        self._update_window(dst)

    def _cannyKsize(self, val):
        self.canny_ksize = 2*val + 3
        upper_threshold = self.low_threshold * self.ratio
        img_blur = self.blur(self.gray)
        detected_edges = cv.Canny(img_blur, self.low_threshold, upper_threshold, apertureSize=self.canny_ksize)
        mask = detected_edges != 0
        dst = self.img * (mask[:, :, None].astype(self.img.dtype))

        # copy bandpass here too
        self.lowpass = (2 * val) + 1  # low-pass filter threshold
        self.highpass = (4 * val) + 1  # highpass filter threshold
        # create Gaussian filter with low variance (high freq can pass)
        blur_high_pass = cv.GaussianBlur(dst, (self.lowpass, self.lowpass), 0)
        # create Gaussian filter with high variance (low freq can pass)
        blur_low_pass = cv.GaussianBlur(dst, (self.highpass, self.highpass), 0)
        # subtract (low pass - high pass) filters to create bandpass filter
        dst = blur_low_pass - blur_high_pass

        self._update_window (dst)

    def bandpass(self, val):
        # must repeat below 5 lines to ensure consistent filtering when changing ksize or threshold
        upper_threshold = self.low_threshold * self.ratio
        img_blur = self.blur(self.gray)
        detected_edges = cv.Canny(img_blur, self.low_threshold, upper_threshold, apertureSize=self.canny_ksize)
        mask = detected_edges != 0
        dst = self.img * (mask[:, :, None].astype(self.img.dtype))

        self.lowpass = (2 * val) + 1  # low-pass filter threshold
        self.highpass = (4 * val) + 1  # highpass filter threshold
        # create Gaussian filter with low variance (high freq can pass)
        blur_high_pass = cv.GaussianBlur(dst, (self.lowpass, self.lowpass), 0)
        # create Gaussian filter with high variance (low freq can pass)
        blur_low_pass = cv.GaussianBlur(dst, (self.highpass, self.highpass), 0)
        # subtract (low pass - high pass) filters to create bandpass filter
        dst = blur_low_pass - blur_high_pass

        self._update_window(dst)

    def hough(self, val):
        # must repeat below 5 lines to ensure consistent filtering when changing ksize or threshold
        upper_threshold = self.low_threshold * self.ratio
        img_blur = self.blur(self.gray)
        detected_edges = cv.Canny(img_blur, self.low_threshold, upper_threshold, apertureSize=self.canny_ksize)
        mask = detected_edges != 0
        dst = self.img * (mask[:, :, None].astype(self.img.dtype))
        # repeat bandpass filtering
        self.lowpass = (2 * self.lowpass) + 1  # low-pass filter threshold
        self.highpass = (4 * self.highpass) + 1  # highpass filter threshold
        # create Gaussian filter with low variance (high freq can pass)
        blur_high_pass = cv.GaussianBlur(dst, (self.lowpass, self.lowpass), 0)
        # create Gaussian filter with high variance (low freq can pass)
        blur_low_pass = cv.GaussianBlur(dst, (self.highpass, self.highpass), 0)
        # subtract (low pass - high pass) filters to create bandpass filter
        dst = blur_low_pass - blur_high_pass

        dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        dst = dst.astype(self.img.dtype)

        # run probabilistic Hough lines transform function
        lines = cv.HoughLinesP(dst, rho=1, theta=np.pi / 180, threshold=val, minLineLength=100, maxLineGap=100)
        # extract start and stop (x,y) cds of each calculated line
        n = lines.shape[0]
        for i in range(n):
            x1 = lines[i][0][0]
            y1 = lines[i][0][1]
            x2 = lines[i][0][2]
            y2 = lines[i][0][3]
            cv.line(dst, (x1, y1), (x2, y2), (255, 0, 0), 2)
        self._update_window(dst)

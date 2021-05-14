from __future__ import print_function
import cv2 as cv
import sys
import os


def get_input_args():
    # error if input is more than 2 (program, file1)
    if len(sys.argv) > 2:
        print(f'\nFormat:\n  {sys.argv[0]} {"{image path/filename1}"} \n')
        exit()
    # error if file1 does not exist
    if not os.path.isfile(sys.argv[1]):
        print(f'\nInvalid file1:  {sys.argv[1]}\n')
        exit()
    # return the file
    return sys.argv[1]


class CannyInteractive:

    def __init__(self, image):
        self.img = image
        self.gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        self.max_lowThreshold = 200
        self.window_name = 'Edge Map'
        self.title_trackbar = 'Min Threshold:'
        self.title_trackbar2 = 'Canny Kern Size:'
        # 2 additional trackbars for bandpass filter
        self.title_trackbar3 = 'Low-pass Filter'
        self.title_trackbar4 = 'Highpass Filter'
        self.ratio = 3
        self.low_threshold = 0
        self.canny_ksize = 3
        # added low-pass and highpass thresholds for bandpass filtering
        self.lowpass = 0
        self.highpass = 0

        cv.namedWindow(self.window_name)
        # thresholding trackbar
        cv.createTrackbar(self.title_trackbar, self.window_name, 0, self.max_lowThreshold, self._canny_threshold)
        # kernel size trackbar
        cv.createTrackbar(self.title_trackbar2, self.window_name, 0, 2, self._cannyKsize)
        # bandpass filter trackbars
        cv.createTrackbar(self.title_trackbar3, self.window_name, 1, 15, self.bandpass)
        cv.createTrackbar(self.title_trackbar4, self.window_name, 1, 15, self.bandpass)

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
        self._update_window(dst)

    def _cannyKsize(self, val):
        self.canny_ksize = 2*val + 3
        upper_threshold = self.low_threshold * self.ratio
        img_blur = self.blur(self.gray)
        detected_edges = cv.Canny(img_blur, self.low_threshold, upper_threshold, apertureSize=self.canny_ksize)
        mask = detected_edges != 0
        dst = self.img * (mask[:, :, None].astype(self.img.dtype))
        self._update_window(dst)

    def bandpass(self, val):
        self.lowpass = (2 * val) + 1  # low-pass filter threshold
        self.highpass = (2 * val) + 1  # highpass filter threshold
        # create Gaussian filter with low variance (high freq can pass)
        blur_high_pass = cv.GaussianBlur(self.img, (self.lowpass, self.lowpass), 0)
        # create Gaussian filter with high variance (low freq can pass)
        # blur_low_pass = cv.GaussianBlur(self.img, (self.highpass, self.highpass), 0)
        # subtract (low pass - high pass) filters to create bandpass filter
        # bandpass = blur_low_pass - blur_high_pass
        self._update_window(blur_high_pass)

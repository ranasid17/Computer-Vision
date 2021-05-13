from __future__ import print_function
import cv2 as cv
import argparse

class CannyInteractive():

    def __init__ (self, image):
        self.gray = cv.cvtColor (image, cv.COLOR_BGR2GRAY)

        self.max_lowThreshold = 200
        self.window_name = 'Edge Map'
        self.title_trackbar = 'Min Threshold:'
        self.title_trackbar2 = 'Canny Kern Size:'
        self.ratio = 3
        self.low_threshold = 0
        self.canny_ksize = 3

        cv.namedWindow (self.window_name)
        cv.createTrackbar (self.title_trackbar, self.window_name , 0, self.max_lowThreshold, self._cannyThreshold)
        cv.createTrackbar (self.title_trackbar2, self.window_name , 0, 2, self._cannyKsize)

        self._cannyThreshold(0)


    def _updateWindow (self, img):
        cv.imshow (self.window_name, img)


    def blur (self, img):
        img_blur = cv.GaussianBlur (img, (25,25), 0)
        return img_blur


    def _cannyThreshold (self, val):
        self.low_threshold = val
        upper_threshold = self.low_threshold * self.ratio
        img_blur = self.blur (self.gray)
        detected_edges = cv.Canny (img_blur, self.low_threshold, upper_threshold, apertureSize=self.canny_ksize)
        mask = detected_edges != 0
        dst = src * (mask[:,:,None].astype (src.dtype))
        self._updateWindow (dst)


    def _cannyKsize (self, val):
        self.canny_ksize = 2*val + 3
        upper_threshold = self.low_threshold * self.ratio
        img_blur = self.blur (self.gray)
        detected_edges = cv.Canny (img_blur, self.low_threshold, upper_threshold, apertureSize=self.canny_ksize)
        mask = detected_edges != 0
        dst = src * (mask[:,:,None].astype (src.dtype))
        self._updateWindow (dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser (description='Code for Canny Edge Detector tutorial.')
    parser.add_argument ('--input', help='Path to input image.', default='RATHN_000000.jpg')
    args = parser.parse_args()

    src = cv.imread (cv.samples.findFile (args.input))
    if src is None:
        print ('Could not open or find the image: ', args.input)
        exit(0)

    CannyInteractive (src)

    cv.waitKey()


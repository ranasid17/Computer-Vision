import numpy as np
import cv2
from featureDetector import FeatureMapper
import sys
import os


def get_input_args():
    # error if input is more than 3 (program, file1, file2)
    if len(sys.argv) != 3:
        print(f'\nFormat:\n  {sys.argv[0]} {"{image path/filename1}"} {"{image path/filename2}"}\n')
        exit()
    # error if file1 does not exist
    if not os.path.isfile (sys.argv[1]):
        print(f'\nInvalid file1:  {sys.argv[1]}\n')
        exit()
    # error if file2 does not exist
    if not os.path.isfile (sys.argv[2]):
        print(f'\nInvalid file2:  {sys.argv[2]}\n')
        exit()
    # return the two files
    return sys.argv[1], sys.argv[2]


if __name__ == "__main__":
    # Get input args from user
    filename, filename2 = get_input_args()
    # Read args as images
    img1 = cv2.imread (filename)
    img2 = cv2.imread (filename2)

    def f(x=None):
        pass

    # Create Window
    cv2.namedWindow('Feature Matching', cv2.WINDOW_NORMAL)

    # Create Trackbars
    det_mthd_select = 'Detector  (HARRIS/FAST/ORB/SIFT)'
    cv2.createTrackbar(det_mthd_select, 'Feature Matching', 0, 3, f)

    descr_mthd_select = 'Descriptor  (BRIEF/ORB/SIFT)'
    cv2.createTrackbar(descr_mthd_select, 'Feature Matching', 0, 2, f)

    match_mthd_select = 'MatchType  (BF_NORM_HAMMING/BF_NORM_2SQR/FLANN)'
    cv2.createTrackbar(match_mthd_select, 'Feature Matching', 0, 2, f)

    ratio_switch_select = 'Ratio  (ON/OFF)'
    cv2.createTrackbar(ratio_switch_select, 'Feature Matching', 0, 1, f)

    # Get trackbar information
    det_mthd = cv2.getTrackbarPos(det_mthd_select, 'Feature Matching')
    descr_mthd = cv2.getTrackbarPos(descr_mthd_select, 'Feature Matching')
    match_mthd = cv2.getTrackbarPos(match_mthd_select, 'Feature Matching')
    ratio_mthd = cv2.getTrackbarPos(ratio_switch_select, 'Feature Matching')

    # Initialize class
    feature_detector = FeatureMapper (img1, img2, det_mthd, descr_mthd, match_mthd, ratio_mthd)

    # Detect using class
    kp1 = feature_detector.detect(img1, det_mthd)
    kp2 = feature_detector.detect(img2, det_mthd)

    # Description using class
    kp1, descr1 = feature_detector.compute (img1, descr_mthd)
    kp2, descr2 = feature_detector.compute (img2, descr_mthd)

    # Match using class
    matches = feature_detector.match(img1, img2, match_mthd, ratio_mthd)
    # Draw matches using class
    feature_detector.plot_matches(img1, img2, matches, kp1, kp2)

    cv2.waitKey()
    cv2.destroyAllWindows()

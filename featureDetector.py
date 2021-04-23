import numpy as np
import cv2


class FeatureMapper:
    PREDEFINED = 3
    # Detector parameters (det_mthd)
    HARRIS_DET = 0
    FAST_DET = 1
    ORB_DET = 2
    SIFT_DET = 3

    # Descriptor parameters (descr_mthd)
    BRIEF_DESCR = 0
    ORB_DESCR = 1
    SIFT_DESCR = 3

    # Matcher parameters (match_mthd)
    BF_MATCHING_NORM_HAMMING = 3
    BF_MATCHING_NORM_L2SQR = 2
    FLANN_MATCHING = 1

    # Ratio test (ratio_mthd)
    OFF = 0
    ON = 3

    def __init__(self, img1, img2, PREDEFINED, det_mthd=SIFT_DET, descr_mthd=SIFT_DESCR,
                 match_mthd=FLANN_MATCHING, ratio_mthd=ON):
        self.det_ver = det_mthd  # load SIFT as default feature detector
        self.descr_ver = descr_mthd  # load SIFT as default feature descriptor
        self.match_ver = match_mthd  # load brute force as default matcher
        self.ratio_ver = ratio_mthd
        self.PREDEFINED = PREDEFINED
        self.img1 = img1
        self.img2 = img2

    # class mutators for trackbar inputs
    def set_det_ver(self, detector):
        self.det_ver = detector

    def set_descr_ver(self, descriptor):
        self.descr_ver = descriptor

    def set_match_ver(self, matcher):
        self.match_ver = matcher

    def set_ratio(self, ratio):
        self.ratio_ver = ratio

    def detect(self, img, detector, method=PREDEFINED):
        # set det_ver == initialized det_ver
        det_ver = self.det_ver

        # update det_ver with trackbar
        FeatureMapper.set_det_ver(self, detector)

        # convert image to greyscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # check if input method has different value from PREDEFINED (default)
        if self.det_ver != self.PREDEFINED:
            # Harris corner detector
            if self.det_ver == 0:
                mod_grey = np.float32(grey)  # not sure if necessary to convert type to float32
                key_points = cv2.cornerHarris(mod_grey, 2, 3, 0.04)  # detect corners, given params
            # FAST corner detector
            elif self.det_ver == 1:
                fast = cv2.FastFeatureDetector()  # create FAST object (default params)
                key_points = fast.detect(grey, None)  # detect key points
            # ORB corner detector
            elif self.det_ver == 2:
                orb = cv2.ORB_create()  # create ORB object (default params)
                key_points = orb.detect(grey, None)  # detect key points
        else:
            # SIFT corner detector
            sift = cv2.SIFT.create()  # create SIFT object
            key_points = sift.detect(grey, None)  # detect key points
        return key_points

    def compute(self, img, descriptor, method=PREDEFINED):
        descr_ver = self.descr_ver

        FeatureMapper.set_descr_ver(self, descriptor)
        # Convert image to greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect key points using detect()
        key_points = FeatureMapper.detect(self, img, self.det_ver)

        if self.descr_ver != self.PREDEFINED:
            if self.descr_ver == 0:
                # Initiate BRIEF object
                brief = cv2.xfeatures2d.BriefDescriptorExtractor_create ()
                # Compute descriptors
                key_points, descriptors = brief.compute(gray, key_points)
            if self.descr_ver == 1:
                # Initiate ORB object
                orb = cv2.ORB ()
                # Compute descriptors
                key_points, descriptors = orb.compute(gray, key_points)
        else:
            # Create SIFT object
            sift = cv2.SIFT.create()
            # Compute descriptors
            key_points, descriptors = sift.compute (gray, key_points)
        # convert descriptors to uint8 type for future matching
        descriptors = descriptors.astype (np.uint8)
        return key_points, descriptors

    def match(self, tr_img, q_img, match, ratio, method=PREDEFINED):
        FeatureMapper.set_match_ver(self, match)
        FeatureMapper.set_ratio(self, ratio)
        # Detect key points of training and test images using input method
        kp1 = FeatureMapper.detect(self, tr_img, self.det_ver)
        kp2 = FeatureMapper.detect(self, q_img, self.det_ver)
        # Calculate kp and descriptors for both images using input method
        kp1, descriptors1 = FeatureMapper.compute(self, tr_img, self.descr_ver)
        kp2, descriptors2 = FeatureMapper.compute(self, q_img, self.descr_ver)

        if self.match_ver != self.PREDEFINED:
            if self.match_ver == 2:
                # initialize brute-force matching object
                bf = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=True)
                # match using brute force matcher
                matches = bf.match(descriptors1, descriptors2)
                # Sort them in the order of their distance.
                matches = sorted(matches, key=lambda x: x.distance)
            elif self.match_ver == 1:
                # FLANN parameters
                flann_index_tree = 0
                index_params = dict(algorithm=flann_index_tree, trees=5)
                search_params = dict(checks=50)  # or pass empty dictionary
                # create FLANN object with params
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        else:
            # initialize brute-force matching object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # match using brute force matcher
            matches = bf.match(descriptors1, descriptors2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
        if self.ratio_ver == 1:
            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            return matches, good
        else:
            return matches
            # match_ver = self.match_ver

    def detect_and_compute(self, img, method=PREDEFINED):
        # convert image to greyscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if method != self.PREDEFINED:
            # ORB is only other method that offers detect and compute so no need to check method for m
            orb = cv2.ORB_create()
            key_points = orb.detect(img, None)
            key_points, descriptors = orb.compute(img, key_points)
        else:
            # create SIFT object
            sift = cv2.SIFT_create()
            # detect key points and compute descriptors
            key_points, descriptors = sift.detectAndCompute(grey, None)
        return key_points, descriptors

    def plot_matches(self, img1, img2, matches, kp1, kp2):
        # Draw matches
        img3 = cv2.drawMatches (img1, kp1, img2, kp2, matches[:25], None)
        cv2.imshow("matches", img3)

        return 0



# I. Background
Using the trackbars provided when running featureDetection.py, we may select between 4 types of corner detection algorithms (Harris, FAST, ORB, SIFT), 3 types of feature description algorithms (BRIEF, ORB, SIFT), and 3 types of matching algorithms (brute force with Hamming distance or L2SQR, and FLANN), along with applying the ratio test or not.

# II. Experimenting with Configurations I and Discussion
Although up to 72 configurations exist by manipulating the trackbars, I tested only 4 for convenience on 6 pairs of images (listed at the end of this document). Each of the configurations are listed below and will hereby referred to as Configuration 1-4. I attempted to determine the number of true positive and false positive matches between features in the training and test image pairs for each configuration. Table 1 presents the number of true positives for each configuration while Table 2 presents the number of false positives for each configuration.
  
  a. Configuration 1: Harris corner detection + BRIEF computation + brute force matching (Hamming distance) - ratio test.
  b. Configuration 2: FAST detection + BRIEF computation + FLANN matching - ratio test
  c. Configuration 3: ORB detection + ORB computation + brute force matching (L2SQR) + ratio test
  d. Configuration 4: SIFT detection + SIFT computation + FLANN matching + ratio test

        Image Pair        Configuration 1   Configuration 2   Configuration 3   Configuration 4
    cv_cover, cv_desk         10                15                14                17
    hp_cover, hp_desk         9                 10                17                18  
    pano_left, pano_right     21                21                NA                N/A
    cv_cover, hp_desk         0                 0                 0                 0
    nasa_logo, k_s_c          15                16                20                18
    pano_left, k_s_c          0                 0                 0                 0
 
**Table 1**: The number of true positives for each image pair passed to each configuration. FAST detection with FLANN matching (Configuration 2) appeared to roduce slightly improved matches compared to Harris detection with brute force matching (Configuration 1) despite not using the ratio test. SIFT detection and omputation with FLANN matching (Configuration 3) appeared to produce slightly improved matches to ORB detection and computation with brute force matching (Configuration 4) despite both using the ratio test.
       
       Image Pair         Configuration 1   Configuration 2   Configuration 3   Configuration 4
    cv_cover, cv_desk         15                10                11                8
    hp_cover, hp_desk         16                15                8                 7
    pano_left, pano_right     4                 4                 N/A               N/A
    cv_cover, hp_desk         25                25                11                14
    nasa_logo, k_s_c          10                9                 5                 7
    pano_left, k_s_c          15                25                25                25

**Table 2**: The number of false positives for each image pair passed to each configuration. Configuration 1 appeared to produce slightly higher false positives compared to Configuration 2, which we expect due to it producing lower true positives (Table 1). Configuration 4 also appeared to produce slightly lower false positives than Configuration 3 for the same reason. However, all 4 configurations produced high false positive rates for the final image pair passed despite Configuration 2 and 4 tending to produce higher true positives than Configurations 1 and 3. The use of the ratio test did however likely allow Configurations 3 and 4 to produce lower false positives than Configurations 1 and 3 for the 4th image pair.
       
To determine the difference between false positives and true positives required careful definitions. Initially, I defined a false positive as any keypoint on the training image that the algorithm mapped to the wrong location on the test image. This definition worked well for image pairs 1, 2, 3, and 5 because the test image contained part of the training image within it. However for image pairs 4 and 6 because the training and test images did not contain any elements of each other. These were intentionally incorrect pairs in order to test the validity of the feature detector. Hence, every match between the training and test image was technically a false positive. Therefore, I attempted to modify my definition of a false positive to where the algorithm attempted to map a keypoint from the training image to the wrong location on the test image. While this did not improve the false positive counts for image pair 6, it did help for image pair 4. Based on the results in Table 1 and Table 2, Configuration 4 appeared to work the best.

# III. Experimenting with Configurations II and Discussion
I now attempted to use the feature detector with a new set of 6 configurations listed below, but this time ran all of the configurations on only 1 image pair (nasa_logo.png, kennedy_space_center.jpg). The results are provided in Table 3.
 a. Configuration 1: SIFT detection + BRIEF computation + FLANN matching
 b. Configuration 2: FAST detection + ORB computation + FLANN matching
 c. Configuration 3: Harris detection + SIFT computation + FLANN matching
 d. Configuration 4: ORB detection + SIFT computation + FLANN matching
 e. Configuration 5: SIFT detection + BRIEF computation + FLANN matching
     
     Configuration    True Positives    False Positives
    Configuration 1       16                  9
    Configuration 2       15                  10
    Configuration 3       18                  7
    Configuration 4       17                  8
    Configuration 5       18                  7

**Table 3**: All configurations 1-5 appear to work with similar accuracy for the given image pair. Configurations 3 and 5 provide the highest true positive and lowest false positive rate.
      

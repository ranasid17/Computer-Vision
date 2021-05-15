# Computer-Vision

## 1) Retina Scanner 

    a) work in progress

## 2) Detect Screens
    
    a) This program accepts 1 input image (as a path to the directory) and returns the laptop screen contained within that image in red. The input image must      contain a laptop or computer screen for this program to work correctly. It does this by first applying adaptive thresholding to the (converted) greyscale image to filter out background. Morphological opening (iterations=1) and closing (iterations=2) is then applied to denoise the filtered image. I then run the connected components (with stats) method from Open CV2 to extract the connected compnoents and their information from the image. Finally to identify the laptop screen I draw contours around each component and filter out any non-rectangular copmonents based on the aspect ratio and extent of each contour. At this point only rectangular contours with the type of a laptop screen remain, and I select for the laptop screen by returning the largest contour (filled in red). 
    
    b) identifyComputerScreen.pdf provides a lengthier discussion on this program. It provides information on how I determined the parameters for each threshold and morphological filtering method, how I optimized this program to run on multiple types of images, and the possible limitations of my program. 
    
    c) 3 images to test this program are provided. A-orig_img.jpg, is the type of image this program was designed to consistently identify laptop screens. The ideal image contains only 1 laptop screen within the image and does not take up the majority of the image. 5_screen_in_bkgd.png contains a laptop screen and a desktop in the background. This is a variation on A-orig_img.jpg. The program can correctly identify the primary laptop screen but will not identify the desktop in the background due to the selection algorithm of the returned bounding box and because the partially covered screen does not form a true rectangle. 1_ms_surface.jpg is a type of image this program will fail upon because the laptop screen takes up the majority of the image field of view 
    

## 3) hybridImages.py 

    a) This program accepts 2 input images (as a path to the directory) and returns the hybrid of both. 

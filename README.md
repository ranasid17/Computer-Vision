# Computer-Vision

1) identifyComputerScreen.py
    
    a) This program accepts 1 input image (as a path to the directory) and returns the laptop screen contained within that image in red. The input image must      contain a laptop or computer screen for this program to work correctly. It does this by first applying adaptive thresholding to the (converted) greyscale image to filter out background. Morphological opening (iterations = 1) and closing (iterations = 2) is then applied to denoise the filtered image. I then run the connected components (with stats) method from Open CV2 to extract the connected compnoents and their information from the image. Finally to identify the laptop screen I draw contours around each component and filter out any non-rectangular copmonents based on the aspect ratio and extent of each contour. At this point only rectangular contours with the type of a laptop screen remain, and I select for the laptop screen by returning the largest contour (filled in red). 
    
    b) identifyComputerScreen_README provides a lengthier discussion on this program. It provides information on how I determined the parameters for each threshold and morphological filtering method, how I optimized this program to run on multiple types of images, and the possible limitations of my program. 


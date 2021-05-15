import sys
import os
import cv2 as cv
from CannyInteractive import CannyInteractive
import numpy as np


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


def get_slope_of_line(line):
    xDis = line[0][2] - line[0][0]
    if xDis == 0:
        return None
    return (line[0][3] - line[0][1]) / xDis


def calculate_distance(lines):
    distances = []
    for line in lines:
        n = line.shape[0]
        for i in range(n):
            x1 = line[i][0][0]
            y1 = line[i][0][1]
            x2 = line[i][0][2]
            y2 = line[i][0][3]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append(distance)
    return distances


def get_slope_line(line):
    x_dist = line[0][2] - line[0][0]
    if x_dist == 0:
        return None
    return (line[0][3] - line[0][1]) / x_dist


def hough(image):
    # run probabilistic Hough lines transform function
    lines = cv.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=100)
    # extract start and stop (x,y) cds of each calculated line
    n = lines.shape[0]
    for i in range(n):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    parallel_lines = []
    for a in lines:
        for b in lines:
            if a is not b:
                slope_a = get_slope_line(a)
                slope_b = get_slope_line(b)
                if slope_a is not None and slope_b is not None:
                    if 0 <= abs(slope_a - slope_b) <= 0.6:
                        parallel_lines.append({'lineA': a, 'lineB': b})
    for pairs in parallel_lines:
        line_a = pairs['lineA']
        line_b = pairs['lineB']
        leftx, boty, rightx, topy = line_a[0]
        cv.line(image, (leftx, boty), (rightx, topy), (0, 0, 255), 2)

    return image, lines


def find_skeleton(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    skel = np.zeros(img.shape, np.uint8)
    eroded = np.zeros(img.shape, np.uint8)
    temp = np.zeros(img.shape, np.uint8)

    _, thresh = cv.threshold(img, 127, 255, 0)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    iters = 0
    while(True):
        cv.erode(thresh, kernel, eroded)
        cv.dilate(eroded, kernel, temp)
        cv.subtract(thresh, temp, temp)
        cv.bitwise_or(skel, temp, skel)
        thresh, eroded = eroded, thresh  # Swap instead of copy

        iters += 1
        if cv.countNonZero(thresh) == 0:
            return skel, iters


def pruning(img):

    morph_kernel = np.array((
        [1, 1, 1],
        [1, -1, 1],
        [0, 1, 0]), dtype="int")
    pruned = cv.morphologyEx(filtered_img, cv.MORPH_HITMISS, morph_kernel)
    # Commented out full pruning algo (as implemented in morphology lecture for runtime purposes)
    # rate = 50
    # morph_kernel = (morph_kernel + 1) * 127
    # morph_kernel = np.uint8(morph_kernel)
    # morph_kernel = cv.resize (morph_kernel, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    # cv.imshow("kernel", kernel)
    # cv.moveWindow ("kernel", 0, 0)
    # input_image = cv.resize(img, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    # cv.imshow("Original", input_image)
    # cv.moveWindow ("Original", 0, 200)
    # pruned = cv.resize (pruned, None, fx=rate, fy=rate, interpolation=cv.INTER_NEAREST)
    # cv.imshow("Hit or Miss", pruned)
    # cv.moveWindow("Hit or Miss", 500, 200)
    return pruned


def connected_pixel_count(img):
    # initialize variable to store number of connected pixels in each direction
    parallel = 0
    skeleton_pixel_count = 1
    for i in range(1, len(filtered_img)-1):  # start from second row, end in second to last to prevent domain error
        for j in range(1, len(filtered_img)-1):  # start from second col, end in second to last to prevent domain error
            # store current pixel
            curr_pixel = filtered_img[i, j]
            # store north, east, south, and west pixels
            prev_pixel = filtered_img[i, j-1]
            next_pixel = filtered_img[i, j+1]
            up_pixel = filtered_img[i-1, j]
            down_pixel = filtered_img[i+1, j]
            # paralellism pixels
            pixel_135 = filtered_img[i-1, j-1]  # northwest pixel
            pixel_45 = filtered_img[i-1, j+1]  # northeast pixel

            # identify a pixel that is part of skeleton
            if curr_pixel != 0:
                # check if its neighbors are part of the skeleton
                if (prev_pixel != 0) & (next_pixel != 0) & (up_pixel !=0) & (down_pixel != 0):
                    # add one to count if so
                    skeleton_pixel_count = skeleton_pixel_count + 1
                    # calculate parallelism as defined by Uji et al (2014) for current pixel
                    parallel = parallel + (np.abs(curr_pixel - up_pixel) + np.abs(pixel_45 - pixel_135) / np.abs(curr_pixel + up_pixel + pixel_45 + pixel_135))
    return skeleton_pixel_count, parallel


if __name__ == "__main__":
    # import file and read as image
    filename = get_input_args()
    src = cv.imread(filename)
    # convert to greyscale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # initialize CI class
    CI = CannyInteractive(src)
    # hold window open until done filtering image
    cv.waitKey()
    cv.destroyAllWindows()

    # pull filtered image
    filtered_img = CI.get_current_img()
    # find skeleton in filtered image
    skeleton_img, iterations = find_skeleton(filtered_img)
    # run hough lines transform and display image
    filtered_img, filtered_lines = hough(skeleton_img)
    cv.imshow("Hough transform", filtered_img)
    # count connectivity of skeleton
    num_conn_pixels, parallelism = connected_pixel_count(filtered_img)

    print("Number of pixels in skeleton:", num_conn_pixels)
    print("Parallelism:", parallelism)

    cv.waitKey()
    cv.destroyAllWindows()


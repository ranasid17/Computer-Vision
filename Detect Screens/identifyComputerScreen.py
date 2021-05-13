import cv2
import numpy as np
import os
import sys
import random as rnd


def get_input_args():
    if len(sys.argv) != 2:
        print(f'\nFormat:\n    {sys.argv[0]}  {"{image path/filename}"}\n')
        exit()

    if not os.path.isfile(sys.argv[1]):
        print(f'\nInvalid file:  {sys.argv[1]}\n')
        exit()

    return sys.argv[1]


def filter_components(num_labels, img_labels):
    # create empty mask
    mask = np.zeros(grey.shape, dtype="uint8")
    for i in range(1, num_labels):  # iter through num_labels
        # store height + width for each component
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # area calculation
        area = w * h
        desired_ratio = w/h  # calculate aspect ratio of each component
        if (desired_ratio >= 1.5) and (desired_ratio <= 2) and (area > 10000):
            component_mask = (img_labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, component_mask)

    return mask


def draw_contours(thresh, pix_labels):
    num_labels = np.max(pix_labels) + 1

    boxed_comps_img = np.zeros([pix_labels.shape[0], pix_labels.shape[1], 3], dtype=np.uint8)
    boxed_comps_img[:, :, :] = 0
    # find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    # draw bounding rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    # calculate aspect ratio, extent, and contour of maximal contour
    aspect_ratio = float(w)/h
    rect_area = w * h
    # calculate area of contour
    area = cv2.contourArea(cnt)
    extent = float(area) / rect_area

    cpy_img = np.zeros(labels.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        if (1.5 <= aspect_ratio <= 3) and (extent > 1.0):
            cpy_img[labels == i] = 177
        else:
            labels[labels == 1] = 0

    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    bound_rect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        bound_rect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)

    # fill target region as red
    for i, cnt in enumerate(contours):
        one_pix_hsv = np.zeros([1, 1, 3], dtype=np.uint8)
        one_pix_hsv[0, 0, :] = [175, 175, 175]
        bgr_color = cv2.cvtColor(one_pix_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        mask = np.zeros(thresh.shape, np.uint8)
        cv2.drawContours(drawing, contours, i, bgr_color, -1)

    return drawing


if __name__ == "__main__":
    filename = get_input_args()  # import file
    # import img + convert to greyscale
    img = cv2.imread(filename)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply thresholding
    th2 = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)  # adaptive threshold
    # morphological filtering
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    # run connected components + return [num unique labels, mask, stats, centroids]
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(closing)
    # filter components
    maskedImg = filter_components(numLabels, labels)
    # contouring
    draw = draw_contours(maskedImg, labels)

    cv2.imshow('original', img)
    cv2.imshow('greyscale', grey)
    cv2.imshow('Adaptive Threshold', th2)
    cv2.imshow('Morphological Opening', opening)
    cv2.imshow('Morphological Opening + Closing', closing)
    cv2.imshow("Components", maskedImg)
    cv2.imshow("Colored Contour", draw)

    cv2.waitKey(0)

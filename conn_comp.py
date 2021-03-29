
import cv2
import numpy as np
import os
import sys
import random as rnd

def getInputArgs ():
    if len(sys.argv) != 2:
        print (f'\nFormat:\n    {sys.argv[0]}  {"{image path/filename}"}\n')
        exit()

    if not os.path.isfile(sys.argv[1]):
        print (f'\nInvalid file:  {sys.argv[1]}\n')
        exit()

    return sys.argv[1]


def draw_contours (pix_labels, thresh):
    num_labels = np.max(pix_labels) + 1

    boxed_comps_img = np.zeros([pix_labels.shape[0], pix_labels.shape[1], 3], dtype=np.uint8)
    boxed_comps_img[:,:,:] = 0

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    rnd.seed()

    # fill regions with random colors
    for i,cnt in enumerate(contours):
        one_pix_hsv = np.zeros([1,1,3],dtype=np.uint8)
        one_pix_hsv[0,0,:] = [ rnd.randint(0,255), rnd.randint(150,255), rnd.randint(200,255) ]
        bgr_color = cv2.cvtColor (one_pix_hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
        mask = np.zeros(thresh.shape,np.uint8)
        cv2.drawContours(boxed_comps_img,contours,i,bgr_color,-1)

    return boxed_comps_img


if __name__ == "__main__":
    filename = getInputArgs()

    img = cv2.imread (filename)
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold (gray, 200, 255, cv2.THRESH_BINARY)[1]  		# ensure binary

    num_labels, pix_labels = cv2.connectedComponents (thresh)
    boxed_conn_comps = draw_contours (pix_labels, thresh)

    cv2.imshow('original', img)
    cv2.imshow('thresholded', thresh)
    cv2.imshow('boxed_conn_comps', boxed_conn_comps)

    cv2.waitKey()

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import os


def getInputArgs():
    """
    :input: min 3 args: script name, image1 location, image2 location (kernel size)
    :return: img1, img2, kernel size
    """
    # modified condition to throw error if <3 args (program, file1, file2) or >4 (program, file1, file2, ksize)
    if len(sys.argv) < 3 or len(sys.argv) >= 5:
        print(f'\nFormat:\n {sys.argv[0]} {"{image path/filename}"} {"{image path/filename}"}\n')
        print('                   OR')
        print(
            f'\n {sys.argv[0]} {"{image path/filename}"} {"{image path/filename}"} {"{height/width of filter mask}"}\n')
        exit()
    if not os.path.isfile(sys.argv[1]):
        print(f'\nInvalid file: {sys.argv[1]}\n')
        exit()
    if not os.path.isfile(sys.argv[2]):  # add condition to throw error if file2 does not exist
        print(f'\nInvalid file: {sys.argv[2]}\n')
        exit()
    if len(sys.argv) == 4:  # change ksize variable initialize to 4th input arg (originally 3)
        ksize = int(sys.argv[3])
    else:
        ksize = 5

    return sys.argv[1], sys.argv[2], ksize  # add return statement for sys.argv[2] AKA: second image


def getLaplaceofGauss(image, ddepth, ksize):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert input image to greyscale
    gauss_blur = cv2.GaussianBlur(grey, (ksize, ksize), 0)  # apply gaussian filter
    laplace_gauss = cv2.Laplacian(gauss_blur, ddepth, ksize, scale=4.5)  # apply Laplacian to gaussian filtered img
    gauss_laplace = cv2.GaussianBlur(laplace_gauss, (ksize, ksize), 0)  # apply second gauss filter to same image
    return gauss_laplace

def convertToFloat32(image):
#    if image != np.float32(image):  # check if image already 32 bit
#        return image  # return unaltered image

    image = cv2.divide(image, 255) # scale image
    image_32bit = np.float32(image)  # change numpy dtype

    return image_32bit

def convertToInt8(image):
#    if image == np.uint8(image): # check if image already 8 bit
#        return image  # return unaltered image

    image_8bit = np.uint8(image)  # change numpy dtype
    image_8bit = cv2.multiply(image_8bit, 255)  # multiply image by 256 to scale

    return image_8bit

def getSobelEdgesFloat32(image, ksize):
    sobel_ksize = min(ksize,7)  # calculate kernel for sobel filter
    sobel_x_flt = cv2.Sobel(image, -1, 1, 0, sobel_ksize)  # apply in x direction
    sobel_y_flt = cv2.Sobel(image, -1, 0, 1, sobel_ksize)  # apply in y direction
    squared_sobelX = cv2.multiply(sobel_x_flt, sobel_x_flt)  # square x image
    squared_sobelY = cv2.multiply(sobel_y_flt, sobel_y_flt)  # square y image
    summed_squares = cv2.add(squared_sobelX, squared_sobelY)  # sum each squared image
    sqrt_summed_squares = cv2.sqrt(summed_squares)  # sqrt sum of squares

    return sqrt_summed_squares

if __name__ == "__main__":
    filename, filename2, ksize = getInputArgs()  # include img2 from getInputArgs

    # Load both images
    img = cv2.imread(filename)
    img2 = cv2.imread(filename2)

#    img = convertToFloat32(img)
#    img2 = convertToInt8(img2)

    img_gauss = cv2.GaussianBlur(img, (ksize, ksize), 2)  # apply Gaussian blurring to first image
    img2_LoG = getLaplaceofGauss(img2, -1, ksize)  # call getLoG function to apply convolved filter to second image

    grey_LoG = cv2.merge([img2_LoG, img2_LoG, img2_LoG])  # create 3 channel greyscale img of LoG
#    im2edges = cv2.bitwise_and(img_gauss, grey3)  # combine 3 ch greyscale img with color
    im2edges_better = cv2.addWeighted(img_gauss, 0.4, grey_LoG, 0.6, 0)

#    img2_sobel = getSobelEdgesFloat32(img2, ksize)
#    grey_sobel = cv2.merge([img2_LoG, img2_LoG, img2_LoG])
#    im2edges_sobel = cv2.addWeighted(img_gauss, 0.5, grey_sobel, 0.50, 0)

    cv2.imshow('image', img)
    cv2.imshow(f'Gaussian filtered by {ksize}x{ksize}', img_gauss)
    cv2.imshow('image2', img2)  # show second image
    cv2.imshow(f'Laplacian filtered by {ksize}x{ksize}, preceded by 3x3 Gaussian filter', grey_LoG)
    cv2.imshow(f'Combined 3 channel grey and LoG', im2edges_better)

    while True:
        k = cv2.waitKey(50) & 0xFF  # 0xFF? To get the lowest byte.
        if k == 27: break  # Code for the <ESC> key
        if k == 32: break  # Code for the <spacebar> key

    cv2.destroyAllWindows()

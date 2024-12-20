"""
Demonstration of how otsu binarisation works
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class HistogramCreation():
    """
    Creates a histogram of grey values from an inputted sheet
    """

    def __init__(self, file):
        """
        Only need to include file path
        """
        self.file = file

    def histogram(self, show_images: bool = False):
        """
        Create histogram from file
        Note 255 represents white so the larger the threshold, the more 
        text is recognised
        """
        image = np.array(cv.imread(self.file, ))
        # print(image)
        image_copy = image.copy()
        # greyscale image
        image_grey = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
        # blurred image
        image_blur = cv.GaussianBlur(
            image_grey, ksize=(15, 15), sigmaX=10, sigmaY=10)
        im_arr = image.flatten()
        blur_arr = image_blur.flatten()
        # threshold value chosen by otsu
        no_bl, image_thresh_no_blur = cv.threshold(
            image_grey, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        image_thresh_temp = cv.threshold(
            image_blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[0]
        image_thresh = cv.threshold(
            image_grey, image_thresh_temp, 255, cv.THRESH_BINARY)[1]
        print(f"No blur Threshold: {no_bl}")
        print(f"Burred Threshold: {image_thresh_temp}")
        # im_hist = np.histogram(im_arr, bins=255)
        # print(im_arr)
        # blur_hist = np.histogram(blur_arr, bins=255)
        plt.hist(im_arr, bins=255)
        plt.axvline(x=no_bl, color="red")
        plt.title("Pixel Frequency Without Blurring")
        plt.show()
        plt.hist(blur_arr, bins=255)
        plt.axvline(x=image_thresh_temp, color="red")
        plt.title("Pixel Frequency With Blurring")
        plt.show()

        # cv.imwrite("OtsuOriginal2.png", image_grey)
        # cv.imwrite("OtsuNoBlur2.png", image_thresh_no_blur)
        # cv.imwrite("OtsuBlur2.png", image_thresh)
        if show_images:
            cv.namedWindow("Original Image", cv.WINDOW_NORMAL)
            cv.namedWindow("No Blur Threshold", cv.WINDOW_NORMAL)
            cv.namedWindow("Blurred Threshold", cv.WINDOW_NORMAL)
            cv.imshow("Original Image", image)
            cv.waitKey()
            cv.imshow("No Blur Threshold", image_thresh_no_blur)
            cv.waitKey()
            cv.imshow("Blurred Threshold", image_thresh)
            cv.waitKey()
            cv.waitKey(0)
            cv.destroyAllWindows()
            cv.waitKey(1)


# FILE = "./PNGFolder/Sheet14.png"
FILE = "./RowTestYear_1959/Sheet0/ColumnFolder1/row30.png"
test = HistogramCreation(FILE)
test.histogram(show_images=True)
# Combine images
# img1 = cv.imread("OtsuOriginal1.png")
# img2 = cv.imread("OtsuNoblur1.png")
# img3 = cv.imread("OtsuBlur1.png")
# img4 = cv.imread("OtsuOriginal2.png")
# img5 = cv.imread("OtsuNoblur2.png")
# img6 = cv.imread("OtsuBlur2.png")

# row1 = np.concatenate((img1, img2, img3), axis=1)
# row2 = np.concatenate((img4, img5, img6), axis=1)
# tab = np.concatenate((row1, row2), axis=0)
# cv.imwrite("OtsuExample.png", tab)
# cv.imshow("Table", tab)
# cv.waitKey(0)
# cv.destroyAllWindows()

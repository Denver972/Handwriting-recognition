"""
Demonstration of Gaussian Blurring
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class BlurDemo():
    """
    Creates images with different blur levels
    """

    def __init__(self, file):
        """
        Only need to include file path
        """
        self.file = file

    def blurred_images(self, show_images: bool = False):
        """
        Create images with a different blur parameters
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
            image_grey, ksize=(15, 15), sigmaX=2, sigmaY=2)
        image_blur_two = cv.GaussianBlur(
            image_grey, ksize=(15, 15), sigmaX=30, sigmaY=30)
        image_blur_three = cv.GaussianBlur(
            image_grey, ksize=(51, 51), sigmaX=2, sigmaY=2)

        # cv.imwrite("BlurOriginal.png", image)
        cv.imwrite("BlurStandard2.png", image_blur)
        cv.imwrite("BlurStrong2.png", image_blur_two)
        cv.imwrite("BlurLarge2.png", image_blur_three)

        if show_images:
            cv.namedWindow("Original Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Standard Blur", cv.WINDOW_NORMAL)
            cv.namedWindow("Strong Blur", cv.WINDOW_NORMAL)
            cv.namedWindow("Large Blur", cv.WINDOW_NORMAL)
            cv.imshow("Original Image", image)
            cv.waitKey()
            cv.imshow("Standard Blur", image_blur)
            cv.waitKey()
            cv.imshow("Strong Blur", image_blur_two)
            cv.waitKey()
            cv.imshow("Large Blur", image_blur_three)
            cv.waitKey(0)
            cv.destroyAllWindows()
            cv.waitKey(1)


# FILE = "./PNGFolder/Sheet14.png"
# FILE = "./RowTestYear_1959/Sheet0/ColumnFolder1/row15.png"
# test = BlurDemo(FILE)
# test.blurred_images(show_images=True)
#### Combine images#####
img1 = cv.imread("OtsuOriginal1.png")
img2 = cv.imread("BlurStandard2.png")
img3 = cv.imread("BlurStrong2.png")
img4 = cv.imread("BlurLarge2.png")
img5 = cv.imread("OtsuOriginal2.png")
img6 = cv.imread("BlurStandard.png")
img7 = cv.imread("BlurStrong.png")
img8 = cv.imread("BlurLarge.png")

row1 = np.concatenate((img1, img2, img3, img4), axis=1)
row2 = np.concatenate((img5, img6, img7, img8), axis=1)
tab = np.concatenate((row1, row2), axis=0)
cv.imwrite("BlurExample.png", tab)
# cv.imshow("Table", tab)
# cv.waitKey(0)
# cv.destroyAllWindows()

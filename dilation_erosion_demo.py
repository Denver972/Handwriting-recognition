"""
Demo of dilation and erosion
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class DilationErosionDemo():
    """
    Creates images with different dilation and erosion applied
    """

    def __init__(self, file):
        """
        Only need to include file path
        """
        self.file = file
        self.dil_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        self.ero_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    def dil_ero_images(self, show_images: bool = False):
        """
        Input: file path
        Output
        """
        image = np.array(cv.imread(self.file, ))
        # print(image)
        image_copy = image.copy()
        # greyscale image
        image_grey = cv.cvtColor(
            image_copy, cv.COLOR_BGR2GRAY+cv.THRESH_BINARY_INV)
        image_bin = cv.threshold(
            image_grey, 128, 255, cv.THRESH_BINARY_INV)[1]
        # blurred image
        image_dil = cv.dilate(
            image_bin, self.dil_kernel, iterations=1)
        image_dil_two = cv.dilate(
            image_bin, self.dil_kernel, iterations=2)
        image_dil_three = cv.dilate(
            image_bin, self.dil_kernel, iterations=3)
        image_ero = cv.erode(
            image_bin, self.ero_kernel, iterations=1)
        image_ero_two = cv.erode(
            image_bin, self.ero_kernel, iterations=2)
        image_ero_three = cv.erode(
            image_bin, self.ero_kernel, iterations=3)

        cv.imwrite("InverseBinary.png", image_bin)
        cv.imwrite("Dilation1.png", image_dil)
        cv.imwrite("Dilation2.png", image_dil_two)
        cv.imwrite("Dilation3.png", image_dil_three)
        cv.imwrite("Erosion1.png", image_ero)
        cv.imwrite("Erosion2.png", image_ero_two)
        cv.imwrite("Erosion3.png", image_ero_three)

        if show_images:
            cv.namedWindow("Original Image", cv.WINDOW_NORMAL)
            cv.namedWindow("One Iteration", cv.WINDOW_NORMAL)
            cv.namedWindow("Two Iterations", cv.WINDOW_NORMAL)
            cv.namedWindow("Three Iterations", cv.WINDOW_NORMAL)
            cv.namedWindow("One Erosion Iteration", cv.WINDOW_NORMAL)
            cv.namedWindow("Two Erosion Iterations", cv.WINDOW_NORMAL)
            cv.namedWindow("Three Erosion Iterations", cv.WINDOW_NORMAL)
            cv.imshow("Original Image", image_bin)
            cv.waitKey()
            cv.imshow("One Iteration", image_dil)
            cv.waitKey()
            cv.imshow("Two Iterations", image_dil_two)
            cv.waitKey()
            cv.imshow("Three Iterations", image_dil_three)
            cv.waitKey()
            cv.imshow("One Erosion Iteration", image_ero)
            cv.waitKey()
            cv.imshow("Two Erosion Iterations", image_ero_two)
            cv.waitKey()
            cv.imshow("Three Erosion Iterations", image_ero_three)
            cv.waitKey(0)
            cv.destroyAllWindows()
            cv.waitKey(1)


# FILE = "OtsuBlur1.png"
# test = DilationErosionDemo(FILE)
# test.dil_ero_images(show_images=False)
# #### Combine images#####
# img1 = cv.imread("InverseBinary.png")
# img2 = cv.imread("DilationOne.png")
# img3 = cv.imread("Dilation2.png")
# img4 = cv.imread("Dilation3.png")
# img5 = cv.imread("InverseBinary.png")
# img6 = cv.imread("Erosion1.png")
# img7 = cv.imread("Erosion2.png")
# img8 = cv.imread("Erosion3.png")

# row1 = np.concatenate((img1, img2, img3, img4), axis=1)
# row2 = np.concatenate((img5, img6, img7, img8), axis=1)
# tab = np.concatenate((row1, row2), axis=0)
# cv.imwrite("DilEroExample5x5.png", tab)
# cv.imshow("Table", tab)
# cv.waitKey(0)
# cv.destroyAllWindows()
# combine two different kernel sizes
row3x3 = cv.imread("DilEroExample3x3.png")
row5x5 = cv.imread("DilEroExample5x5.png")
tab2 = np.concatenate((row3x3, row5x5), axis=0)
cv.imwrite("DilEroExample.png", tab2)
cv.imshow("Table", tab2)
cv.waitKey(0)
cv.destroyAllWindows()

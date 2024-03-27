# Image preprocessing. The goal is to first tilt the image so text runs
# horizontally. Then blur the text to merge the text in each column
# together create bounding boxes around each column. Then potentially
# sharpen the image and within each bounding box, put bounding boxes
# around each character. These characters will be indexed with the
# column number, line they occupy, and the column they occupy. This
# should allow for the format to be reconstructed after handwriting
# recognition has been performed.

"""
Need cv2 for image manipulation and numpy for the arrays
"""
import cv2
import numpy as np


class PreProcess:
    """
    This class will include all the methods required to pre process an 
    image for handwriting recognition. Each method in the class should 
    be called in order to identify where the characters are.
    """

    def __init__(self, file_path):
        """
        remove error
        """

    def grey_image(self):
        """
        remove error
        """

    def blur_text(self):
        """
        remove error
        """

    def rotate_image(self):
        """
        Takes the blurred image and will compute the angle to rotate the
        image by and then rotate the image outputting a new image that 
        will be orriented correctly.
        INPUT: Greyed, blurrred image
        OUTPUT: orriginal image orriented correctly
        PATH 
        """


class ImageRotation():
    """
    Method to rotate the image 
    """

    def __init__(self):
        # self.angle = None
        self.file = None

    def skew_angle(self, file, show_images: bool = False):
        """
        Returns the angle with which to rotate the image
        """
        image = np.array(cv2.imread(file, 1))

        # Make copy of orriginal image
        image_copy = image.copy()

        # Convert image to greyscale
        image_grey = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

        # Blur image
        image_blur = cv2.GaussianBlur(image_grey, (9, 9), 0)

        # Threshold
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # dilate to merge lines or columns
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        image_dilated = cv2.dilate(image_thresh, kernel, iterations=5)

        # detect contours
        contours, hierarchy = cv2.findContours(
            image_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # find largest contour
        largest_contour = contours[0]
        min_area_rect = cv2.minAreaRect(largest_contour)

        # detect angle
        angle = min_area_rect[-1]

        if angle < -45:
            angle = 90 + angle
            return -1.0 * angle
        elif angle > 45:
            angle = 90 - angle
            return angle

        if show_images:
            min_rect_contour = np.int0(cv2.boxPoints(min_area_rect))
            temp1 = cv2.drawContours(
                image_copy.copy(), contours, -1, (255, 0, 0), 2)
            temp2 = cv2.drawContours(
                image_copy.copy(), [min_rect_contour], -1, (255, 0, 0), 2)
            cv2.namedWindow("Greyed imagage", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Blurred image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Threeshold image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Dilated image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("All Contours", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Largest Contour", cv2.WINDOW_NORMAL)
            cv2.imshow("Greyed imagage", image_grey)
            cv2.waitKey()
            cv2.imshow("Blurred image", image_blur)
            cv2.waitKey()
            cv2.imshow("Threeshold image", image_thresh)
            cv2.waitKey()
            cv2.imshow("Dilated image", image_dilated)
            cv2.waitKey()
            cv2.imshow("All Contours", temp1)
            cv2.waitKey()
            cv2.imshow("Largest Contour", temp2)
            cv2.waitKey()
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        return angle

    def rotate_image(self):
        """
        Returns the rotated image
        """

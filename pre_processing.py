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
import os
import cv2
import numpy as np
import fitz_old
from pypdf import PdfReader, PdfWriter


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
        print(image.shape)
        # Make copy of orriginal image
        image_copy = image.copy()
        print(image_copy.shape)
        # Convert image to greyscale
        image_grey = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        print(image_grey.shape)
        # Blur image
        image_blur = cv2.GaussianBlur(
            image_grey, ksize=(15, 15), sigmaX=10, sigmaY=10)
        print(image_blur.shape)
        # Threshold
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        print(image_thresh.shape)
        # dilate to merge lines or columns
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        image_dilated = cv2.dilate(image_thresh, kernel, iterations=2)
        print(image_dilated.shape)
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
            cv2.namedWindow("Threshold image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Dilated image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("All Contours", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Largest Contour", cv2.WINDOW_NORMAL)
            cv2.imshow("Greyed imagage", image_grey)
            cv2.waitKey()
            cv2.imshow("Blurred image", image_blur)
            cv2.waitKey()
            cv2.imshow("Threshold image", image_thresh)
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

    def rotate_image(self, file):
        """
        Returns the rotated image
        """
        angle = self.skew_angle(file)
        image_new = np.array(cv2.imread(file, 1))
        image_grey = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)
        (h, w) = image_grey.shape[:2]
        centre = (w//2, h//2)
        M = cv2.getRotationMatrix2D(centre, angle, 1.0)
        image_new = cv2.warpAffine(image_grey, M, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        image_save = cv2.imwrite(file, image_new)
        return image_new


class FileSeparation():
    """
    This class will separate a multi page PDF into individual PNG files 
    to give to the PreProccess class. This should rename the files 
    appropriately to perserve the order and structure. Additionally two
    directories will be created to store all the individual files

    INPUT: PDF containing multiple pages
    OUTPUT: Individual PNG files
    """

    def __init__(self):
        self.file = None

    def folder_creation(self):
        """
        Create directories for both pdf file and png file
        """
        pdf_path = "./PDFFolder"
        png_path = "./PNGFolder"

        os.mkdir(pdf_path)
        os.mkdir(png_path)

    def file_split(self, file):
        """
        Splits the PDFs into single page PDFs
        Currently unnessecary as pdf_to_png will split the pdf into 
        individual pages
        """
        input_pdf = PdfReader(open(file, "rb"))
        page = 0
        total_pages = len(input_pdf.pages)
        for pg in range(page, total_pages):
            print(pg)
            output_pdf = PdfWriter()
            output_pdf.add_page(input_pdf.pages[pg])
            output_file = f"./PDFFolder/test{pg}.pdf"
            with open(output_file, "wb") as output:
                output_pdf.write(output)

# TODO: fix fitz_old
    def pdf_to_png(self, file):
        """
        Convert pdf to multiple images
        RGB format
        """
        input_pdf = fitz_old.open(file)
        for page in input_pdf:
            print(page)
            pix = page.get_pixmap(dpi=300)
            pix.save(f"./PNGFolder/test{page.number}.png")

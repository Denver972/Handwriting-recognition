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
        return image_save


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


class TableDetect():
    """
    Determine the number of columns and rows in the image
    Input: rotated, greyed image
    Output: number of rows
            number of columns
    """

    def __init__(self):
        self.file = None

    def remove_lines(self, file, fileOut, show_images: bool = False):
        """
        Input: Grey image
        Output: Image with table lines removed
        """
        image = np.array(cv2.imread(file, 0))
        image_copy = image.copy()
        image_blur = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=10,
                                      sigmaY=10)
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
        remove_horizontal = cv2.morphologyEx(
            image_thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        cnts = cv2.findContours(
            remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image_copy, [c], -1, (255, 255, 255), 5)
        # Remove vertical lines
        # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
        # remove_vertical = cv2.morphologyEx(
        #     image_thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        # cnts = cv2.findContours(
        #     remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # for c in cnts:
        #     cv2.drawContours(image_copy, [c], -1, (255, 255, 255), 5)

        cv2.imwrite(fileOut, image_copy)

        if show_images:
            cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            cv2.imshow('Threshold', image_thresh)
            cv2.imshow('Result', image_copy)
            cv2.waitKey()

    def rows(self, file, show_images: bool = False):
        """
        Extract the rows from the table. Idea is to elongate the text in
        x direction to make row a continuous block and then use contours
        to count how many blocks there are
        """
        image = np.array(cv2.imread(file, 0))
        image_thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # horizontal_blur = cv2.GaussianBlur(image, ksize=(15, 15), sigmaX=10,
        #                                    sigmaY=10)
        # instead want to dilate
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        image_dilated = cv2.dilate(
            image_thresh, horizontal_kernel, iterations=4)
        # make contours

        if show_images:
            cv2.namedWindow("Horizontal Blur", cv2.WINDOW_NORMAL)
            cv2.imshow("Horizontal Blur", image_dilated)
            cv2.waitKey()
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        # return nrows

    def columns(self):
        """
        Same concept as above except stretch in the y dimension
        """
        # return ncols

    def dimensions(self):
        """
        Combine the rows and columns to give the dimensions of the table
        This will then be used when indexing the cells
        """
        dims = np.array([self.rows, self.columns], dtype=int)

        return dims


class RowExtraction():
    """
    Extract the lines of the text
    INPUT: Extracted columns
    Output: Images of the column rows
    """

    def __init__(self):
        self.file = None

    def row_locate(self, file, show_images: bool = False):
        """
        Find the contours to extract the rows of the columns
        Input: Column image
        OUTPUT: Images of rows in the column
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv2.imread(file, 0))
        image_copy = image.copy()
        image_colour = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
        image_blur = cv2.GaussianBlur(image, ksize=(9, 9), sigmaX=50,
                                      sigmaY=50)
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # step 2 is to dilate the text to merge characters together
        # want a balance between under dilation and over dilation
        # This is for no lines and 6 iterations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        image_dilate = cv2.dilate(image_thresh, kernel, iterations=10)

        # step 3 is to find contours
        contours, hierarchy = cv2.findContours(
            image_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Assume there is no rotation in the image as it is already corrected
        # lower left coordinates of bounding rectangle
        x = np.zeros((len(contours)), dtype=int)
        y = np.zeros((len(contours)), dtype=int)
        # width and height of the bounding rectangle
        w = np.zeros((len(contours)), dtype=int)
        h = np.zeros((len(contours)), dtype=int)
        print(len(contours))
        # print(x.shape)
        # Loop through all contours to get the bounding boxes
        for ix, cont in enumerate(contours):
            x[ix], y[ix], w[ix], h[ix] = cv2.boundingRect(cont)
        boundary = np.transpose(np.array([x, y, h, w], dtype=int))

        if show_images:
            # min_rect_contour = np.int0(cv2.boxPoints(min_area_rect))
            # temp1 = cv2.drawContours(
            #    image_colour, contours, -1, (0, 255, 0), 2)
            for ix, cont in enumerate(contours):
                if cv2.contourArea(cont) > 500:
                    temp1 = cv2.rectangle(image_colour, (x[ix], y[ix]),
                                          (x[ix]+w[ix], y[ix]+h[ix]), (0, 255, 0), 2)
            cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Blurred Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Threshold Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Dilated Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
            cv2.imshow("Input Image", image)
            cv2.imshow("Blurred Image", image_blur)
            cv2.imshow("Threshold Image", image_thresh)
            cv2.imshow("Dilated Image", image_dilate)
            cv2.imshow("Contours", temp1)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return boundary

    def extraction(self, fileIn, fileOut):
        """
        INPUT: Rectangle dims [lower x, lower y, height, width]
               Image with no lines removed
        OUTPUT: Individual images of the detected words
        """
        image = np.array(cv2.imread(fileIn, 0))
        boundary = self.row_locate(fileIn)
        # sort the boundaries by row
        # boundary = sorted(boundary, key=lambda x: x[1])
        # boundary.sort()
        # sort boundary by distance from top
        # print(boundary.shape)
        boundary = np.stack(boundary)
        # print(boundary.shape)
        boundary = sorted(
            boundary, key=lambda x: x[1])

        # print(boundary)
        for ix, bound in enumerate(boundary):
            rect = image[bound[1]:bound[1] +
                         bound[2], bound[0]:bound[0]+bound[3]]
            place = os.path.join(fileOut, f"row{ix}.png")
            cv2.imwrite(place, rect)


class ColumnExtraction():
    """
    Extract the columns of the tables. Use the Time column as a guide for
    tables vertical size, then use headers as horizontal boundary locaters
    """

    def __init__(self):
        self.file = None

    def col_locate(self, file, show_images: bool = False):
        """
        Find the contours for columns to prepare them for extraction
        INPUT: Pre-processed image
        OUTPUT: Contours off the columns sorted left to right
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv2.imread(file, 0))
        image_copy = image.copy()
        image_colour = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
        image_blur = cv2.GaussianBlur(image, ksize=(9, 9), sigmaX=50,
                                      sigmaY=50)
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # step 2 is to dilate the text to merge characters together
        # want a balance between under dilation and over dilation
        # This is for no lines and 6 iterations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        image_dilate = cv2.dilate(image_thresh, kernel, iterations=5)

        # step 3 is to find contours
        contours, hierarchy = cv2.findContours(
            image_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Assume there is no rotation in the image as it is already corrected
        # lower left coordinates of bounding rectangle
        x = np.zeros((len(contours)), dtype=int)
        y = np.zeros((len(contours)), dtype=int)
        # width and height of the bounding rectangle
        w = np.zeros((len(contours)), dtype=int)
        h = np.zeros((len(contours)), dtype=int)
        print(len(contours))
        # print(x.shape)
        # Loop through all contours to get the bounding boxes
        for ix, cont in enumerate(contours):
            x[ix], y[ix], w[ix], h[ix] = cv2.boundingRect(cont)
        boundary = np.transpose(np.array([x, y, h, w], dtype=int))

        if show_images:
            # min_rect_contour = np.int0(cv2.boxPoints(min_area_rect))
            # temp1 = cv2.drawContours(
            #    image_colour, contours, -1, (0, 255, 0), 2)
            for ix, cont in enumerate(contours):
                if cv2.contourArea(cont) > 500:
                    temp1 = cv2.rectangle(image_colour, (x[ix], y[ix]),
                                          (x[ix]+w[ix], y[ix]+h[ix]), (0, 255, 0), 2)
            cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Blurred Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Threshold Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Dilated Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
            cv2.imshow("Input Image", image)
            cv2.imshow("Blurred Image", image_blur)
            cv2.imshow("Threshold Image", image_thresh)
            cv2.imshow("Dilated Image", image_dilate)
            cv2.imshow("Contours", temp1)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return boundary

    def extraction(self, fileIn, fileOut):
        """
        INPUT: Rectangle dims [lower x, lower y, height, width]
               Image with no lines removed
        OUTPUT: Individual images of the detected words
        """
        image = np.array(cv2.imread(fileOut, 0))
        boundary = self.col_locate(fileIn)
        # sort the boundaries by row
        # boundary = sorted(boundary, key=lambda x: x[1])
        # boundary.sort()
        # sort boundary by distance from the origin
        # print(boundary.shape)
        boundary = np.stack(boundary)
        # print(boundary.shape)
        boundary = sorted(
            boundary, key=lambda x: x[0])

        # print(boundary)
        for ix, bound in enumerate(boundary):
            rect = image[bound[1]:bound[1] +
                         bound[2], bound[0]:bound[0]+bound[3]]
            cv2.imwrite(f"test_cont_extract/column{ix}.png", rect)


class WordExtraction():
    """
    Extract the whole "cell" or words within the cell
    INPUT: Pre-processed image
    """

    def __init__(self):
        self.file = None

    def cell_locate(self, file, show_images: bool = False):
        """
        Finds the contours around each cell
        INPUT: Pre-processed image
        OUTPUT: Countours of the cells
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv2.imread(file, 0))
        image_copy = image.copy()
        image_colour = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
        image_blur = cv2.GaussianBlur(image, ksize=(9, 9), sigmaX=50,
                                      sigmaY=50)
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # step 2 is to dilate the text to merge characters together
        # want a balance between under dilation and over dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
        image_dilate = cv2.dilate(image_thresh, kernel, iterations=2)

        # step 3 is to find contours
        contours, hierarchy = cv2.findContours(
            image_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Assume there is no rotation in the image as it is already corrected
        # lower left coordinates of bounding rectangle
        x = np.zeros((len(contours)), dtype=int)
        y = np.zeros((len(contours)), dtype=int)
        # width and height of the bounding rectangle
        w = np.zeros((len(contours)), dtype=int)
        h = np.zeros((len(contours)), dtype=int)
        print(len(contours))
        print(x.shape)
        # Loop through all contours to get the bounding boxes
        for ix, cont in enumerate(contours):
            x[ix], y[ix], w[ix], h[ix] = cv2.boundingRect(cont)
        boundary = np.transpose(np.array([x, y, h, w], dtype=int))
        # print(contours[1])
        # largest_contour = contours[0]
        # rect_cont = np.empty(shape=(len(contours), 4), dtype=int)
        # min_rect_contour = np.empty(shape=(len(contours), 4), dtype=int)
        # for ix, cont in enumerate(contours):
        #     # rect_cont[ix] = cv2.minAreaRect(cont)
        #     min_rect_contour[ix] = np.int0(cv2.boxPoints(cont))

        if show_images:
            # min_rect_contour = np.int0(cv2.boxPoints(min_area_rect))
            # temp1 = cv2.drawContours(
            #    image_colour, contours, -1, (0, 255, 0), 2)
            for ix, cont in enumerate(contours):
                if cv2.contourArea(cont) > 500:
                    temp1 = cv2.rectangle(image_colour, (x[ix], y[ix]),
                                          (x[ix]+w[ix], y[ix]+h[ix]), (0, 255, 0), 2)
            cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Blurred Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Threshold Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Dilated Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
            cv2.imshow("Input Image", image)
            cv2.imshow("Blurred Image", image_blur)
            cv2.imshow("Threshold Image", image_thresh)
            cv2.imshow("Dilated Image", image_dilate)
            cv2.imshow("Contours", temp1)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return boundary

    def extraction(self, file):
        """
        INPUT: Rectangle dims [lower x, lower y, height, width]
        OUTPUT: Individual images of the detected words
        """
        image = np.array(cv2.imread(file, 0))
        boundary = self.cell_locate(file)
        # sort the boundaries by row
        # boundary = sorted(boundary, key=lambda x: x[1])
        # boundary.sort()
        # sort boundary by distance from the origin
        print(boundary.shape)
        boundary = np.stack(boundary)
        print(boundary.shape)
        boundary = sorted(
            boundary, key=lambda x: np.sqrt(x[0]*x[0] + x[1]*x[1]))

        print(boundary)
        for ix, bound in enumerate(boundary):
            rect = image[bound[1]:bound[1] +
                         bound[2], bound[0]:bound[0]+bound[3]]
            cv2.imwrite(f"test_cont_extract/word{ix}.png", rect)


class CharacterExtraction():
    """
    Extract the characters from the image, resize them to make them
    uniform, save them as individual images
    INPUT: Pre-processed image
    """

    def __init__(self):
        self.file = None


class TableExtraction():
    """
    Extract the columns of the tables. Use the Time column as a guide for
    tables vertical size, then use headers as horizontal boundary locaters
    """

    def __init__(self):
        self.file = None

    def table_locate(self, file, show_images: bool = False):
        """
        Find the contours for columns to prepare them for extraction
        WILL NOT WORK FOR TYPED DATA
        INPUT: Pre-processed image
        OUTPUT: Contours off the columns sorted left to right
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv2.imread(file, 0))
        image_copy = image.copy()
        image_colour = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
        image_blur = cv2.GaussianBlur(image, ksize=(9, 9), sigmaX=50,
                                      sigmaY=50)
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # step 2 is to dilate the text to merge characters together
        # want a balance between under dilation and over dilation
        # This is for no lines and 6 iterations
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 10))
        # with lines in the table
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        image_dilate = cv2.dilate(image_thresh, kernel, iterations=1)

        # step 3 is to find contours
        contours, hierarchy = cv2.findContours(
            image_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Assume there is no rotation in the image as it is already corrected
        # lower left coordinates of bounding rectangle
        x = np.zeros((len(contours)), dtype=int)
        y = np.zeros((len(contours)), dtype=int)
        # width and height of the bounding rectangle
        w = np.zeros((len(contours)), dtype=int)
        h = np.zeros((len(contours)), dtype=int)
        print(len(contours))
        print(x.shape)
        # Loop through all contours to get the bounding boxes
        for ix, cont in enumerate(contours):
            x[ix], y[ix], w[ix], h[ix] = cv2.boundingRect(cont)
        boundary = np.transpose(np.array([x, y, h, w], dtype=int))

        if show_images:
            # min_rect_contour = np.int0(cv2.boxPoints(min_area_rect))
            # temp1 = cv2.drawContours(
            #    image_colour, contours, -1, (0, 255, 0), 2)
            for ix, cont in enumerate(contours):
                if cv2.contourArea(cont) > 500:
                    temp1 = cv2.rectangle(image_colour, (x[ix], y[ix]),
                                          (x[ix]+w[ix], y[ix]+h[ix]), (0, 255, 0), 2)
            cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Blurred Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Threshold Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Dilated Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
            cv2.imshow("Input Image", image)
            cv2.imshow("Blurred Image", image_blur)
            cv2.imshow("Threshold Image", image_thresh)
            cv2.imshow("Dilated Image", image_dilate)
            cv2.imshow("Contours", temp1)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return boundary

    def extraction(self, file):
        """
        INPUT: Rectangle dims [lower x, lower y, height, width]
        OUTPUT: Table area (Largest found contour)
        """
        image = np.array(cv2.imread(file, 0))
        boundary = self.table_locate(file)
        # sort the boundaries by row
        # boundary = sorted(boundary, key=lambda x: x[1])
        # boundary.sort()
        # sort boundary by distance from the origin
        # print(boundary.shape)
        boundary = np.stack(boundary)
        # print(boundary.shape)
        # Sort by area and reverse to get the largest indexed first
        boundary = sorted(
            boundary, key=lambda x: x[3]*x[2], reverse=True)
        # only want largest boundary as this should be the table
        # print(boundary)
        table_boundary = boundary[0]
        rectangle = image[table_boundary[1]:table_boundary[1]+table_boundary[2],
                          table_boundary[0]:table_boundary[0]+table_boundary[3]]
        cv2.imwrite("test_cont_extract/Table.png", rectangle)

        # for ix, bound in enumerate(boundary):
        #     rect = image[bound[1]:bound[1] +
        #                  bound[2], bound[0]:bound[0]+bound[3]]
        #     cv2.imwrite(f"test_cont_extract/word{ix}.png", rect)

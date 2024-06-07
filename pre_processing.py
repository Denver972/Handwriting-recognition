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
import math
import cv2
import numpy as np
import pandas as pd
import fitz_old
from pypdf import PdfReader, PdfWriter
from PIL import Image


class PreProcess():
    """
    This class will include all the methods required to pre process an
    image for handwriting recognition. Each method in the class should
    be called in order to identify where the characters are.
    """

    def __init__(self, file, year):
        """
        Chosen kernels for dilation/erosion for spliting the image for
        character recognition
        file: Multipage PDF
        year: year of data
        """
        self.file = file
        self.year = year
        self.rotate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        self.table_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        self.horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (80, 1))
        self.vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, 100))
        self.col_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        self.row_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        self.char_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def construct(self):
        """
        Goal is to create folder structure that is easily loopable through
        with structure page->column->row->order. this will need to detect the
        pages in the pdf and then for now, the number of columns, rows and
        characters will be detected automatically in further parts
        INPUT: Multi page tabular pdf
        OUTPUT: Folder structure and
        """
        # Step 1: Create the higher level folders
        png_path = "./PNGFolder"
        year_path = f"./Year_{self.year}"
        os.mkdir(png_path)
        os.mkdir(year_path)
        input_pdf = fitz_old.open(self.file)
        # create blank list to store the file names
        file_names = []
        # Step 2: Split the pdf into individual pages and apply the
        #         pre-processing steps to each page at spliting time
        for page in input_pdf:
            print(page)
            pix = page.get_pixmap(dpi=300)
            pix.save(f"./PNGFolder/Sheet{page.number}.png")
            page_path = f"./Year_{self.year}/Sheet{page.number}"
            os.mkdir(page_path)
            # Save the rotated image in the page path with name Rotated_sheet
            rotated_path = os.path.join(page_path, "Rotated_sheet.png")
            cv2.imwrite(rotated_path,
                        self.rotate_image(path=f"./PNGFolder/Sheet{page.number}.png", show_images=False))
            # Extract and save the table part
            table_path = os.path.join(page_path, "Table.png")
            cv2.imwrite(table_path, self.table_locate(path=rotated_path))
            # Save an image with no vertical lines, no horizontal lines and
            # neither lines (three images in total)
            no_hori_path = os.path.join(page_path, "No_horizontal_lines.png")
            no_vert_path = os.path.join(page_path, "No_vertical_lines.png")
            no_line_path = os.path.join(page_path, "No_lines.png")

            cv2.imwrite(no_hori_path, self.remove_hori_lines(table_path))
            cv2.imwrite(no_vert_path, self.remove_vert_lines(table_path))
            cv2.imwrite(no_line_path, self.remove_lines(table_path))

            # Extract columns, use the image with no vertical lines as input
            # and extract the found contours on the image that contains no
            # horizontal lines and no lines
            column_path = os.path.join(page_path, "Work")
            os.mkdir(column_path)
            col_no_lines_image = np.array(cv2.imread(no_line_path, 0))
            col_extract_image = np.array(cv2.imread(no_hori_path, 0))
            for ix, bound in enumerate(self.col_extract(path=no_vert_path)):
                rect = col_extract_image[bound[1]:bound[1] +
                                         bound[2], bound[0]:bound[0]+bound[3]]

                rect_two = col_no_lines_image[bound[1]:bound[1] +
                                              bound[2], bound[0]:bound[0]+bound[3]]
                out_file = f"column{ix}.png"
                col_no_line = f"col_no_line{ix}.png"
                cv2.imwrite(os.path.join(column_path, out_file), rect)
                cv2.imwrite(os.path.join(column_path, col_no_line), rect_two)

                row_extract_input = os.path.join(column_path, out_file)
                row_extract_image = np.array(cv2.imread(
                    os.path.join(column_path, col_no_line), 0))
                new_folder = f"ColumnFolder{ix}"
                col_folder = os.path.join(page_path, new_folder)
                os.mkdir(col_folder)
                for jx, perim in enumerate(self.row_extract(path=row_extract_input)):
                    cell = row_extract_image[perim[1]:perim[1] +
                                             perim[2], perim[0]:perim[0]+perim[3]]
                    result_file = f"row{jx}.png"
                    row_result = os.path.join(col_folder, result_file)
                    cv2.imwrite(row_result, cell)
                    # set up character folder
                    r_folder = f"RowFolder{jx}"
                    row_folder = os.path.join(col_folder, r_folder)
                    row_result_image = np.array(cv2.imread(row_result, 0))
                    os.mkdir(row_folder)
                    for kx, outside in enumerate(self.char_extract(path=row_result)):
                        char = row_result_image[outside[1]:outside[1] +
                                                outside[2], outside[0]:outside[0]+outside[3]]
                        char_file = f"char{kx}.png"
                        char_result = os.path.join(row_folder, char_file)
                        cv2.imwrite(char_result, char)

                        # Create resized characters: Goal dimensions 30x30
                        resize_file = f"resized{kx}.png"
                        resize_result = os.path.join(row_folder, resize_file)
                        image = Image.open(char_result)
                        width, height = image.size
                        new_width = 30
                        new_height = 30
                        if width <= new_width and height <= new_height:
                            horizontal_adjustment = new_width - width
                            vertical_adjustment = new_height - height
                            # right = horizontal_adjustment/2
                            left = math.floor(horizontal_adjustment/2)
                            top = math.floor(vertical_adjustment/2)
                            # bottom = vertical_adjustment/2
                            result = Image.new(
                                image.mode, (new_width, new_height), 255)

                            result.paste(image, (left, top))
                            result.save(resize_result)
                        elif width < new_width:
                            horizontal_adjustment = new_width - width
                            left = math.floor(horizontal_adjustment/2)
                            widened_image = Image.new(
                                image.mode, (new_width, height), 255)
                            widened_image.paste(image, (left, 0))
                            widened_result = widened_image.resize(
                                (new_width, new_height))
                            widened_result.save(resize_result)
                        elif height < new_height:
                            vertical_adjustment = new_height - height
                            top = math.floor(vertical_adjustment/2)
                            heightened_image = Image.new(
                                image.mode, (width, new_height), 255)
                            heightened_image.paste(image, (0, top))
                            heightened_result = heightened_image.resize(
                                (new_width, new_height))
                            heightened_result.save(resize_result)
                        else:
                            result = image.resize((new_width, new_height))
                            result.save(resize_result)

                        binary_file = f"binary{kx}.png"
                        binary_result = os.path.join(row_folder, binary_file)
                        binary = self.convert_to_binary(path=resize_result)
                        cv2.imwrite(binary_result, binary)
                        skeleton_file = f"skeleton{kx}.png"
                        skeleton_result = os.path.join(
                            row_folder, skeleton_file)
                        skeleton = self.skeletonize(path=binary_result)
                        cv2.imwrite(skeleton_result, skeleton)

                        file_names.append(skeleton_result)
        # Create the csv/dataframe to hold the character paths
        dict = {"Character Path": file_names}
        df = pd.DataFrame(dict)
        df.to_csv(f"{self.year}Characters.csv", index=False)

    def resize_char(self):
        pass

    def dataset_creation(self):
        # want to create a list that contains the names of all the images and
        # their paths. Then they will be converted to a dataframe and saved as
        # a csv file
        # file_names = []
        # for pg in year_folder:
        #     for
        pass

    def rotate_image(self, path, show_images: bool = False):
        """
        Takes the image and will compute the angle to rotate the
        image by and then rotate the image outputting a new image that
        will be orriented correctly.
        INPUT: image path
        OUTPUT:image orriented correctly
        """
        # read in the image, accepts rgb images
        image = np.array(cv2.imread(path, 1))
        # print(image.shape)
        # Make copy of orriginal image
        image_copy = image.copy()
        # print(image_copy.shape)
        # Convert image to greyscale
        image_grey = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        # print(image_grey.shape)
        # Blur image
        image_blur = cv2.GaussianBlur(
            image_grey, ksize=(15, 15), sigmaX=10, sigmaY=10)
        # print(image_blur.shape)
        # Threshold
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # print(image_thresh.shape)
        # dilate to merge lines or columns
        image_dilated = cv2.dilate(
            image_thresh, self.rotate_kernel, iterations=2)
        # print(image_dilated.shape)
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

        (h, w) = image_grey.shape[:2]
        centre = (w//2, h//2)
        rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        image_new = cv2.warpAffine(image_grey, rotation_matrix, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        # save image
        # cv2.imwrite("Rotated_sheet.png", image_new)

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

        return image_new

    def table_locate(self, path, show_images: bool = False):
        """
        Find the contours for columns to prepare them for extraction
        WILL NOT WORK FOR TYPED DATA
        INPUT: Rotated image
        OUTPUT: Rectangular contour containing the table
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv2.imread(path, 0))
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
        image_dilate = cv2.dilate(
            image_thresh, self.table_kernel, iterations=1)

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
        # Loop through all contours to get the bounding boxes
        for ix, cont in enumerate(contours):
            x[ix], y[ix], w[ix], h[ix] = cv2.boundingRect(cont)

        boundary = np.transpose(np.array([x, y, h, w], dtype=int))
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
        # cv2.imwrite("test_cont_extract/Table.png", rectangle)

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
        return rectangle

    def remove_vert_lines(self, path):
        """
        Removes vertical lines from the table providing distinct columns
        separated by whitespace. Identifies objects in the image that have
        the structure of a vertical line and replaces the pixel value with
        that of a white pixel
        Input: Table image
        Oput: Table image with the vertical lines removed
        """
        image = np.array(cv2.imread(path, 0))
        image_copy = image.copy()
        image_blur = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=10,
                                      sigmaY=10)
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Detect vertical lines
        remove_vertical = cv2.morphologyEx(
            image_thresh, cv2.MORPH_OPEN, self.vertical_kernel, iterations=1)
        cnts = cv2.findContours(
            remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # Remove vertical lines
        for c in cnts:
            cv2.drawContours(image_copy, [c], -1, (255, 255, 255), 5)

        return image_copy

    def remove_hori_lines(self, path):
        """
        Remove horizontal lines from the table
        Input: Table image
        Output: Image without horizontal lines
        """
        image = np.array(cv2.imread(path, 0))
        image_copy = image.copy()
        image_blur = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=10,
                                      sigmaY=10)
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Identifyhorizontal lines
        remove_horizontal = cv2.morphologyEx(
            image_thresh, cv2.MORPH_OPEN, self.horizontal_kernel, iterations=1)
        cnts = cv2.findContours(
            remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # remove contours
        for c in cnts:
            cv2.drawContours(image_copy, [c], -1, (255, 255, 255), 5)

        return image_copy

    def remove_lines(self, path):
        """
        Input: Grey image
        Output: Image with table lines removed
        """
        image = np.array(cv2.imread(path, 0))
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
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
        remove_vertical = cv2.morphologyEx(
            image_thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        cnts = cv2.findContours(
            remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image_copy, [c], -1, (255, 255, 255), 5)

        return image_copy

    def col_extract(self, path, show_images: bool = False):
        """
        Find the contours for columns to prepare them for extraction
        INPUT: Pre-processed image
        OUTPUT: Contours off the columns sorted left to right
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv2.imread(path, 0))
        # colour image only to show green contours
        image_copy = image.copy()
        image_colour = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
        image_blur = cv2.GaussianBlur(image, ksize=(9, 9), sigmaX=50,
                                      sigmaY=50)
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # step 2 is to dilate the text to merge characters together
        # want a balance between under dilation and over dilation
        # This is for no lines and 6 iterations
        image_dilate = cv2.dilate(image_thresh, self.col_kernel, iterations=5)

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
        # print(len(contours))
        # print(x.shape)
        # Loop through all contours to get the bounding boxes
        for ix, cont in enumerate(contours):
            x[ix], y[ix], w[ix], h[ix] = cv2.boundingRect(cont)
        boundary = np.transpose(np.array([x, y, h, w], dtype=int))

        boundary = np.stack(boundary)

        boundary = sorted(boundary, key=lambda x: x[0])

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

    def row_extract(self, path, show_images: bool = False):
        """
        Finds the contours around each cell
        INPUT: Pre-processed image
        OUTPUT: Countours of the cells
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv2.imread(path, 0))
        image_copy = image.copy()
        image_colour = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
        image_blur = cv2.GaussianBlur(image, ksize=(9, 9), sigmaX=50,
                                      sigmaY=50)
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # step 2 is to dilate the text to merge characters together
        # want a balance between under dilation and over dilation
        image_dilate = cv2.dilate(image_thresh, self.row_kernel, iterations=10)

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
        # print(len(contours))
        # print(x.shape)
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

        boundary = np.stack(boundary)
        # Curently sorted by distance from the origin (upper left corner)
        # Distance from top may be better
        boundary = sorted(
            boundary, key=lambda x: np.sqrt(x[0]*x[0] + x[1]*x[1]))
        # Want contours only larger than a specific size/ want to only extraxt
        # the cells and nothing between on accident
        # new_bound = boundary
        # for ix, cont in enumerate(cont):
        #     if cv2.contourArea(cont) > 500:
        #         x[ix], y[ix], w[ix], h[ix] = cv2.boundingRect(cont)

        # reduced_bounds = []
        # index = []

        # for i, bound in enumerate(boundary):
        #     if bound[2] > 30:
        #         index[i] = i
        # print(reduced_bounds.shape)
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

    def char_extract(self, path, show_images: bool = False):
        """
        Finds the contours around each cell
        INPUT: Pre-processed image
        OUTPUT: Countours of the cells
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv2.imread(path, 0))
        image_copy = image.copy()
        image_colour = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
        image_blur = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=5,
                                      sigmaY=5)
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # step 2 is to dilate the text to merge characters together
        # want a balance between under dilation and over dilation
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        # image_dilate = cv2.erode(image_thresh, kernel, iterations=1)

        # step 3 is to find contours
        contours, hierarchy = cv2.findContours(
            image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Assume there is no rotation in the image as it is already corrected
        # lower left coordinates of bounding rectangle
        x = np.zeros((len(contours)), dtype=int)
        y = np.zeros((len(contours)), dtype=int)
        # width and height of the bounding rectangle
        w = np.zeros((len(contours)), dtype=int)
        h = np.zeros((len(contours)), dtype=int)
        # print(len(contours))
        # print(x.shape)
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
        # sort characters left to right
        # boundary = np.stack(boundary)
        boundary = sorted(boundary, key=lambda x: x[0])
        if show_images:
            # min_rect_contour = np.int0(cv2.boxPoints(min_area_rect))
            # temp1 = cv2.drawContours(
            #    image_colour, contours, -1, (0, 255, 0), 2)
            for ix, cont in enumerate(contours):
                temp1 = cv2.rectangle(image_colour, (x[ix], y[ix]),
                                      (x[ix]+w[ix], y[ix]+h[ix]), (0, 255, 0), 2)
            cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Blurred Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Threshold Image", cv2.WINDOW_NORMAL)
            # cv2.namedWindow("Dilated Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
            cv2.imshow("Input Image", image)
            cv2.imshow("Blurred Image", image_blur)
            cv2.imshow("Threshold Image", image_thresh)
            # cv2.imshow("Dilated Image", image_dilate)
            cv2.imshow("Contours", temp1)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        return boundary

    def convert_to_binary(self, path, show_images: bool = False):
        """
        Convert to binary and invert the color so text will be white (1)
        and background will be black (0) use otsu method as it should yield
        good performance when the character is not a single shade
        """
        image = np.array(cv2.imread(path, 0))
        image_copy = image.copy()
        image_blur = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=15,
                                      sigmaY=15)
        thresh, image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        image_out = cv2.threshold(
            image_copy, thresh, 255, cv2.THRESH_BINARY_INV)[1]
        if show_images:
            cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Threshold Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Input Image", image)
            cv2.imshow("Threshold Image", image_thresh)
            cv2.imshow("Output Image", image_out)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return image_out

    def skeletonize(self, path, show_images: bool = False):
        """
        Extract the skeletons of the characters in order to make them more 
        uniform. Need to dialate first as some characters have gaps in their 
        lines so erosion just makes those larger
        INPUT: Binary Image
        OUTPUT: Skeletonized Image
        """
        image = np.array(cv2.imread(path, 0))
        image_copy = image.copy()
        # image_blur = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=15,
        #                              sigmaY=15)
        image_dilate = cv2.dilate(
            image_copy, kernel=self.char_kernel, iterations=1)
        image_out = cv2.erode(
            image_dilate, kernel=self.erosion_kernel, iterations=1)
        if show_images:
            cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Dilated Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Input Image", image)
            cv2.imshow("Dilated Image", image_dilate)
            cv2.imshow("Output Image", image_out)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return image_out


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
        DEPRECIATED
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
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
        remove_vertical = cv2.morphologyEx(
            image_thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        cnts = cv2.findContours(
            remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image_copy, [c], -1, (255, 255, 255), 5)

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

    def char_locate(self, file, show_images: bool = False):
        """
        Finds the contours around each cell
        INPUT: Pre-processed image
        OUTPUT: Countours of the cells
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv2.imread(file, 0))
        image_copy = image.copy()
        image_colour = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
        image_blur = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=5,
                                      sigmaY=5)
        image_thresh = cv2.threshold(
            image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # step 2 is to dilate the text to merge characters together
        # want a balance between under dilation and over dilation
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        # image_dilate = cv2.erode(image_thresh, kernel, iterations=1)

        # step 3 is to find contours
        contours, hierarchy = cv2.findContours(
            image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                temp1 = cv2.rectangle(image_colour, (x[ix], y[ix]),
                                      (x[ix]+w[ix], y[ix]+h[ix]), (0, 255, 0), 2)
            cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Blurred Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Threshold Image", cv2.WINDOW_NORMAL)
            # cv2.namedWindow("Dilated Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
            cv2.imshow("Input Image", image)
            cv2.imshow("Blurred Image", image_blur)
            cv2.imshow("Threshold Image", image_thresh)
            # cv2.imshow("Dilated Image", image_dilate)
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
        boundary = self.char_locate(file)
        # sort the boundaries by row
        # boundary = sorted(boundary, key=lambda x: x[1])
        # boundary.sort()
        # sort boundary by distance from the origin
        print(boundary.shape)
        boundary = np.stack(boundary)
        print(boundary.shape)
        boundary = sorted(
            boundary, key=lambda x: x[0])

        print(boundary)
        for ix, bound in enumerate(boundary):
            rect = image[bound[1]:bound[1] +
                         bound[2], bound[0]:bound[0]+bound[3]]
            cv2.imwrite(f"test_cont_extract/word{ix}.png", rect)


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


class FileConstructor():
    """
    Create the folder structure to store the images and split the pdf 
    into individual images for each page
    """

    def __init__(self, file, year):
        self.file = file
        self.year = year

    def construct(self):
        """
        Goal is to create folder structure that is easily loopable through
        with structure page->column->row->order. this will need to detect the 
        pages in the pdf and then for now, the number of columns, rows and 
        characters will be detected automatically in further parts
        INPUT: Multi page tabular pdf
        OUTPUT: Folder structure and
        """
        png_path = "./PNGFolder"
        pg_path = f"./Y{self.year}"
        os.mkdir(png_path)
        os.mkdir(pg_path)
        input_pdf = fitz_old.open(self.file)
        for page in input_pdf:
            print(page)
            pix = page.get_pixmap(dpi=300)
            pix.save(f"./PNGFolder/Sheet{page.number}.png")
            page_path = f"./Y{self.year}/P{page.number}"
            os.mkdir(page_path)

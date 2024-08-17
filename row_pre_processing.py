# Pre-processing of the documents to split up the pdf and create a csv of
# the file path to the cell images. Includes converting the images to binary
# and potentially other preprocessing steps
"""
Get rid of the error
"""

import os
import math
import cv2 as cv
import numpy as np
import pandas as pd
import fitz_old
from PIL import Image


class PreProcess():
    """
    INPUT: PDF of scanned documents/datesets
    OUTPUT: CSV of image paths of the cells/"words"
    """

    def __init__(self, file, year):
        """
        Defines the kernels that will be used
        """
        self.file = file
        self.year = year
        self.rotate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
        self.table_kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
        self.horizontal_kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (80, 1))
        self.vertical_kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (1, 100))
        self.col_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 50))
        self.row_kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 1))
        self.word_kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 5))
        # self.char_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        self.erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

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
        png_path = "./PNGFolder2"
        year_path = f"./RowTestYear_{self.year}"
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
            pix.save(f"./PNGFolder2/Sheet{page.number}.png")
            page_path = f"./RowTest3Year_{self.year}/Sheet{page.number}"
            os.mkdir(page_path)
            # Save the rotated image in the page path with name Rotated_sheet
            rotated_path = os.path.join(page_path, "Rotated_sheet.png")
            cv.imwrite(rotated_path,
                       self.rotate_image(path=f"./PNGFolder2/Sheet{page.number}.png", show_images=False))
            # Extract and save the table part
            table_path = os.path.join(page_path, "Table.png")
            crop_path = os.path.join(page_path, "CropTable.png")
            table_extract, threshold = self.table_locate(path=rotated_path)
            cv.imwrite(table_path, table_extract)
            cv.imwrite(crop_path, self.crop_table(path=table_path))
            # Save an image with no vertical lines, no horizontal lines and
            # neither lines (three images in total)
            no_hori_path = os.path.join(page_path, "No_horizontal_lines.png")
            no_vert_path = os.path.join(page_path, "No_vertical_lines.png")
            no_line_path = os.path.join(page_path, "No_lines.png")

            cv.imwrite(no_hori_path, self.remove_hori_lines(crop_path))
            cv.imwrite(no_vert_path, self.remove_vert_lines(crop_path))
            cv.imwrite(no_line_path, self.remove_lines(crop_path))

            # Extract columns, use the image with no vertical lines as input
            # and extract the found contours on the image that contains no
            # horizontal lines and no lines
            column_path = os.path.join(page_path, "Work")
            os.mkdir(column_path)
            col_no_lines_image = np.array(cv.imread(no_line_path, 0))
            col_extract_image = np.array(cv.imread(no_hori_path, 0))
            for ix, bound in enumerate(self.col_extract(path=no_vert_path)):
                rect = col_extract_image[bound[1]:bound[1] +
                                         bound[2], bound[0]:bound[0]+bound[3]]

                rect_two = col_no_lines_image[bound[1]:bound[1] +
                                              bound[2], bound[0]:bound[0]+bound[3]]
                out_file = f"column{ix}.png"
                col_no_line = f"col_no_line{ix}.png"
                cv.imwrite(os.path.join(column_path, out_file), rect)
                cv.imwrite(os.path.join(column_path, col_no_line), rect_two)

                row_extract_input = os.path.join(column_path, out_file)
                row_extract_image = np.array(cv.imread(
                    os.path.join(column_path, col_no_line), 0))
                new_folder = f"ColumnFolder{ix}"
                col_folder = os.path.join(page_path, new_folder)
                os.mkdir(col_folder)
                for jx, perim in enumerate(self.row_extract(path=row_extract_input)):
                    cell = row_extract_image[perim[1]:perim[1] +
                                             perim[2], perim[0]:perim[0]+perim[3]]
                    result_file = f"row{jx}.png"
                    row_result = os.path.join(col_folder, result_file)
                    cv.imwrite(row_result, cell)
                    # row_result_image = np.array(cv.imread(row_result, 0))
                    # convert to binary
                    binary_file = f"binary{jx}.png"
                    # binary_folder = "BinaryFolder"
                    # binary_path = os.path.join(col_folder, binary_folder)
                    # binary_result = os.path.join(binary_path, binary_file)
                    binary_result = os.path.join(col_folder, binary_file)
                    binary = self.convert_to_binary(
                        path=row_result, threshold=threshold)
                    cv.imwrite(binary_result, binary)

                    word_file = f"word{jx}.png"
                    # word_folder = "WordFolder"
                    # word_interim = os.path.join(col_folder, word_folder)
                    word_result = os.path.join(col_folder, word_file)

                    word = self.word_extract(binary_result)
                    cv.imwrite(word_result, word)
                    resized_file = f"resized{jx}.png"
                    resized_result = os.path.join(col_folder, resized_file)
                    # resized_folder = "ResizedFolder"
                    # resized_path = os.path.join(col_folder, resized_folder)
                    # resized_result = os.path.join(resized_path, resized_file)
                    resized = self.resize_image(path=word_result)
                    resized.save(resized_result)

                    file_names.append(resized_result)

        # Create the csv/dataframe to hold the character paths
        file_path_dict = {"CellPath": file_names}
        df = pd.DataFrame(file_path_dict)
        df["Label"] = 0  # word that displayed in the image
        df["Class"] = 0  # numerical label that coresponds to the word
        # df["PredClass"] = 0  # Predicted class by the model
        df.to_csv(f"{self.year}RowTest3.csv", index=False)

    def rotate_image(self, path, show_images: bool = False):
        """
        Takes the image and will compute the angle to rotate the
        image by and then rotate the image outputting a new image that
        will be orriented correctly.
        INPUT: image path
        OUTPUT:image orriented correctly
        """
        # read in the image, accepts rgb images
        image = np.array(cv.imread(path, 1))
        # print(image.shape)
        # Make copy of orriginal image
        image_copy = image.copy()
        # print(image_copy.shape)
        # Convert image to greyscale
        image_grey = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
        # print(image_grey.shape)
        # Blur image
        image_blur = cv.GaussianBlur(
            image_grey, ksize=(15, 15), sigmaX=10, sigmaY=10)
        # print(image_blur.shape)
        # Threshold
        image_thresh = cv.threshold(
            image_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        # print(image_thresh.shape)
        # dilate to merge lines or columns
        image_dilated = cv.dilate(
            image_thresh, self.rotate_kernel, iterations=2)
        # print(image_dilated.shape)
        # detect contours
        contours = cv.findContours(
            image_dilated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        # find largest contour
        largest_contour = contours[0]
        min_area_rect = cv.minAreaRect(largest_contour)

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
        rotation_matrix = cv.getRotationMatrix2D(centre, angle, 1.0)
        image_new = cv.warpAffine(image_grey, rotation_matrix, (w, h),
                                  flags=cv.INTER_CUBIC,
                                  borderMode=cv.BORDER_REPLICATE)
        # save image
        # cv.imwrite("Rotated_sheet.png", image_new)

        if show_images:
            min_rect_contour = np.int0(cv.boxPoints(min_area_rect))
            temp1 = cv.drawContours(
                image_copy.copy(), contours, -1, (255, 0, 0), 2)
            temp2 = cv.drawContours(
                image_copy.copy(), [min_rect_contour], -1, (255, 0, 0), 2)
            cv.namedWindow("Greyed image", cv.WINDOW_NORMAL)
            cv.namedWindow("Blurred image", cv.WINDOW_NORMAL)
            cv.namedWindow("Threshold image", cv.WINDOW_NORMAL)
            cv.namedWindow("Dilated image", cv.WINDOW_NORMAL)
            cv.namedWindow("All Contours", cv.WINDOW_NORMAL)
            cv.namedWindow("Largest Contour", cv.WINDOW_NORMAL)
            cv.imshow("Greyed imagage", image_grey)
            cv.waitKey()
            cv.imshow("Blurred image", image_blur)
            cv.waitKey()
            cv.imshow("Threshold image", image_thresh)
            cv.waitKey()
            cv.imshow("Dilated image", image_dilated)
            cv.waitKey()
            cv.imshow("All Contours", temp1)
            cv.waitKey()
            cv.imshow("Largest Contour", temp2)
            cv.waitKey()
            cv.waitKey(0)
            cv.destroyAllWindows()
            cv.waitKey(1)

        return image_new

    def table_locate(self, path, show_images: bool = False):
        """
        Find the contours for columns to prepare them for extraction
        WILL NOT WORK FOR TYPED DATA
        INPUT: Rotated image
        OUTPUT: Rectangular contour containing the table
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv.imread(path, 0))
        image_copy = image.copy()
        image_colour = cv.cvtColor(image_copy, cv.COLOR_GRAY2RGB)
        image_blur = cv.GaussianBlur(image, ksize=(9, 9), sigmaX=50,
                                     sigmaY=50)
        thresh_level, image_thresh = cv.threshold(
            image_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        # step 2 is to dilate the text to merge characters together
        # want a balance between under dilation and over dilation
        # This is for no lines and 6 iterations
        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (60, 10))
        # with lines in the table
        image_dilate = cv.dilate(
            image_thresh, self.table_kernel, iterations=1)

        # step 3 is to find contours
        contours = cv.findContours(
            image_dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        # Assume there is no rotation in the image as it is already corrected
        # lower left coordinates of bounding rectangle
        x = np.zeros((len(contours)), dtype=int)
        y = np.zeros((len(contours)), dtype=int)
        # width and height of the bounding rectangle
        w = np.zeros((len(contours)), dtype=int)
        h = np.zeros((len(contours)), dtype=int)
        # Loop through all contours to get the bounding boxes
        for ix, cont in enumerate(contours):
            x[ix], y[ix], w[ix], h[ix] = cv.boundingRect(cont)

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
        # cv.imwrite("test_cont_extract/Table.png", rectangle)

        if show_images:
            # min_rect_contour = np.int0(cv.boxPoints(min_area_rect))
            # temp1 = cv.drawContours(
            #    image_colour, contours, -1, (0, 255, 0), 2)
            for ix, cont in enumerate(contours):
                if cv.contourArea(cont) > 500:
                    temp1 = cv.rectangle(image_colour, (x[ix], y[ix]),
                                         (x[ix]+w[ix], y[ix]+h[ix]), (0, 255, 0), 2)
            cv.namedWindow("Input Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Blurred Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Threshold Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Dilated Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Contours", cv.WINDOW_NORMAL)
            cv.imshow("Input Image", image)
            cv.imshow("Blurred Image", image_blur)
            cv.imshow("Threshold Image", image_thresh)
            cv.imshow("Dilated Image", image_dilate)
            cv.imshow("Contours", temp1)
            cv.waitKey()
            cv.destroyAllWindows()
            cv.waitKey(1)
        return rectangle, thresh_level

    def crop_table(self, path):
        """
        Remove the header of the table to hopefully get the rows to be aligned
        Table layout has labels on top and left so keep the bottom and right 
        most pixels
        """
        image = np.array(cv.imread(path, 0))
        image_copy = image.copy()
        # image size thet is desired, may also remove the first column as it is
        # typed it is about 145 pixels
        height = image_copy.shape[0]
        width = image_copy.shape[1]
        height_goal = 2244  # manualy chosen by checking the table dimensions
        width_goal = width  # 2330

        crop_height = height - height_goal
        crop_width = width - width_goal

        image_crop = image_copy[crop_height:height, crop_width:width]
        return image_crop

    def remove_vert_lines(self, path):
        """
        Removes vertical lines from the table providing distinct columns
        separated by whitespace. Identifies objects in the image that have
        the structure of a vertical line and replaces the pixel value with
        that of a white pixel
        Input: Table image
        Oput: Table image with the vertical lines removed
        """
        image = np.array(cv.imread(path, 0))
        image_copy = image.copy()
        image_blur = cv.GaussianBlur(image, ksize=(5, 5), sigmaX=10,
                                     sigmaY=10)
        image_thresh = cv.threshold(
            image_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        # Detect vertical lines
        remove_vertical = cv.morphologyEx(
            image_thresh, cv.MORPH_OPEN, self.vertical_kernel, iterations=1)
        cnts = cv.findContours(
            remove_vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # Remove vertical lines
        for c in cnts:
            cv.drawContours(image_copy, [c], -1, (255, 255, 255), 5)

        return image_copy

    def remove_hori_lines(self, path):
        """
        Remove horizontal lines from the table
        Input: Table image
        Output: Image without horizontal lines
        """
        image = np.array(cv.imread(path, 0))
        image_copy = image.copy()
        image_blur = cv.GaussianBlur(image, ksize=(5, 5), sigmaX=10,
                                     sigmaY=10)
        image_thresh = cv.threshold(
            image_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        # Identifyhorizontal lines
        remove_horizontal = cv.morphologyEx(
            image_thresh, cv.MORPH_OPEN, self.horizontal_kernel, iterations=1)
        cnts = cv.findContours(
            remove_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # remove contours
        for c in cnts:
            cv.drawContours(image_copy, [c], -1, (255, 255, 255), 5)

        return image_copy

    def remove_lines(self, path):
        """
        Input: Grey image
        Output: Image with table lines removed
        """
        image = np.array(cv.imread(path, 0))
        image_copy = image.copy()
        image_blur = cv.GaussianBlur(image, ksize=(5, 5), sigmaX=10,
                                     sigmaY=10)
        image_thresh = cv.threshold(
            image_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        # Remove horizontal lines
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (80, 1))
        remove_horizontal = cv.morphologyEx(
            image_thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=1)
        cnts = cv.findContours(
            remove_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv.drawContours(image_copy, [c], -1, (255, 255, 255), 5)
        # Remove vertical lines
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 100))
        remove_vertical = cv.morphologyEx(
            image_thresh, cv.MORPH_OPEN, vertical_kernel, iterations=1)
        cnts = cv.findContours(
            remove_vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv.drawContours(image_copy, [c], -1, (255, 255, 255), 5)

        return image_copy

    def col_extract(self, path, show_images: bool = False):
        """
        Find the contours for columns to prepare them for extraction
        INPUT: Pre-processed image
        OUTPUT: Contours off the columns sorted left to right
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv.imread(path, 0))
        # colour image only to show green contours
        image_copy = image.copy()
        image_colour = cv.cvtColor(image_copy, cv.COLOR_GRAY2RGB)
        image_blur = cv.GaussianBlur(image, ksize=(9, 9), sigmaX=50,
                                     sigmaY=50)
        image_thresh = cv.threshold(
            image_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        # step 2 is to dilate the text to merge characters together
        # want a balance between under dilation and over dilation
        # This is for no lines and 6 iterations
        image_dilate = cv.dilate(image_thresh, self.col_kernel, iterations=5)

        # step 3 is to find contours
        contours = cv.findContours(
            image_dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=cv.contourArea, reverse=True)
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
            x[ix], y[ix], w[ix], h[ix] = cv.boundingRect(cont)
        boundary = np.transpose(np.array([x, y, h, w], dtype=int))

        boundary = np.stack(boundary)

        boundary = sorted(boundary, key=lambda x: x[0])

        if show_images:
            # min_rect_contour = np.int0(cv.boxPoints(min_area_rect))
            # temp1 = cv.drawContours(
            #    image_colour, contours, -1, (0, 255, 0), 2)
            for ix, cont in enumerate(contours):
                if cv.contourArea(cont) > 500:
                    temp1 = cv.rectangle(image_colour, (x[ix], y[ix]),
                                         (x[ix]+w[ix], y[ix]+h[ix]), (0, 255, 0), 2)
            cv.namedWindow("Input Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Blurred Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Threshold Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Dilated Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Contours", cv.WINDOW_NORMAL)
            cv.imshow("Input Image", image)
            cv.imshow("Blurred Image", image_blur)
            cv.imshow("Threshold Image", image_thresh)
            cv.imshow("Dilated Image", image_dilate)
            cv.imshow("Contours", temp1)
            cv.waitKey()
            cv.destroyAllWindows()
            cv.waitKey(1)
        return boundary

    def row_extract(self, path, show_images: bool = False):
        """
        Finds the contours around each cell
        INPUT: Pre-processed image
        OUTPUT: Countours of the cells
        """
        # Step 1 is to convert the image to binary
        image = np.array(cv.imread(path, 0))
        image_copy = image.copy()
        image_colour = cv.cvtColor(image_copy, cv.COLOR_GRAY2RGB)
        image_blur = cv.GaussianBlur(image, ksize=(9, 9), sigmaX=50,
                                     sigmaY=50)
        image_thresh = cv.threshold(
            image_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        # step 2 is to dilate the text to merge characters together
        # want a balance between under dilation and over dilation
        image_dilate = cv.dilate(image_thresh, self.row_kernel, iterations=10)

        # step 3 is to find contours
        contours = cv.findContours(
            image_dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=cv.contourArea, reverse=True)
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
            x[ix], y[ix], w[ix], h[ix] = cv.boundingRect(cont)
        boundary = np.transpose(np.array([x, y, h, w], dtype=int))
        # print(contours[1])
        # largest_contour = contours[0]
        # rect_cont = np.empty(shape=(len(contours), 4), dtype=int)
        # min_rect_contour = np.empty(shape=(len(contours), 4), dtype=int)
        # for ix, cont in enumerate(contours):
        #     # rect_cont[ix] = cv.minAreaRect(cont)
        #     min_rect_contour[ix] = np.int0(cv.boxPoints(cont))

        boundary = np.stack(boundary)
        # Curently sorted by distance from the origin (upper left corner)
        # Distance from top may be better
        boundary = sorted(
            boundary, key=lambda x: np.sqrt(x[0]*x[0] + x[1]*x[1]))
        # Want contours only larger than a specific size/ want to only extraxt
        # the cells and nothing between on accident
        # new_bound = boundary
        # for ix, cont in enumerate(cont):
        #     if cv.contourArea(cont) > 500:
        #         x[ix], y[ix], w[ix], h[ix] = cv.boundingRect(cont)

        # reduced_bounds = []
        # index = []

        # for i, bound in enumerate(boundary):
        #     if bound[2] > 30:
        #         index[i] = i
        # print(reduced_bounds.shape)
        if show_images:
            # min_rect_contour = np.int0(cv.boxPoints(min_area_rect))
            # temp1 = cv.drawContours(
            #    image_colour, contours, -1, (0, 255, 0), 2)
            for ix, cont in enumerate(contours):
                if cv.contourArea(cont) > 500:
                    temp1 = cv.rectangle(image_colour, (x[ix], y[ix]),
                                         (x[ix]+w[ix], y[ix]+h[ix]), (0, 255, 0), 2)
            cv.namedWindow("Input Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Blurred Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Threshold Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Dilated Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Contours", cv.WINDOW_NORMAL)
            cv.imshow("Input Image", image)
            cv.imshow("Blurred Image", image_blur)
            cv.imshow("Threshold Image", image_thresh)
            cv.imshow("Dilated Image", image_dilate)
            cv.imshow("Contours", temp1)
            cv.waitKey()
            cv.destroyAllWindows()
            cv.waitKey(1)
        return boundary

    def convert_to_binary(self, path, threshold, show_images: bool = False):
        """
        Convert the cells to binary using the total page otsu thresholding as
        the threshold level. This will result in lost information but it is 
        a tradeoff that is required to be made
        """
        image = np.array(cv.imread(path, 0))
        image_copy = image.copy()
        # image_blur = cv.GausianBlur(
        #     image_copy, ksize=(3, 3), sigmaX=5, sigmaY=5)
        image_thresh = cv.threshold(
            image_copy, threshold, 255, cv.THRESH_BINARY_INV)[1]

        if show_images:
            cv.namedWindow("Input Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Threshold Image", cv.WINDOW_NORMAL)
            cv.imshow("Input Image", image)
            cv.imshow("Threshold Image", image_thresh)
            cv.waitKey()
            cv.destroyAllWindows()
            cv.waitKey(1)

        return image_thresh

    def word_extract(self, path):
        """
        Get the largest bounding box in the cell image to extract just the 
        important information
        INPUT: path to binary image of the cell
        OUTPUT: roi of the cell
        """

        image = np.array(cv.imread(path, 0))
        image_copy = image.copy()
        # 1 erosion and 3 dilation is nearly there
        image_eroded = cv.erode(
            image_copy, self.erosion_kernel, iterations=1)
        image_dilated = cv.dilate(image_eroded, self.word_kernel, iterations=3)
        image_dilate = cv.dilate(image_dilated, kernel=cv.getStructuringElement(
            cv.MORPH_RECT, (3, 5)), iterations=1)

        if image_dilate.max() > 0:
            # ensures there is something to contour around
            contours = cv.findContours(
                image_dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
            # largest contour first
            contours = sorted(contours, key=cv.contourArea, reverse=True)
            largest_contour = contours[0]
            temp = cv.boundingRect(largest_contour)
            x, y, w, h = temp
            image_word = image_copy[y:y+h, x:x+w]
        else:
            image_word = image_copy

        return image_word

    def resize_image(self, path):
        """
        INPUT: Binary word image
        OUTPUT: Resized image to set dimensions
        """
        # length of longest cell is 200, most have a height of about 50
        # so this is a good starting point
        # different cases, height too small/large, width too small/large
        # or combination of the above. Too small case is easiest to deal
        # with. Add space equally around the word
        # TODO: change from pillow to numpy

        goal_width = 200
        goal_height = 50

        image = Image.open(path)
        width, height = image.size

        horizontal_adjustment = goal_width - width
        vertical_adjustment = goal_height - height
        left = math.floor(horizontal_adjustment/2)
        top = math.floor(vertical_adjustment/2)

        if width <= goal_width and height <= goal_height:
            resized_image = Image.new(image.mode, (goal_width, goal_height), 0)
            resized_image.paste(image, (left, top))
        elif width < goal_width:
            resized_image = Image.new(image.mode, (goal_width, height), 0)
            resized_image.paste(image, (left, 0))
            resized_image = resized_image.resize(
                (goal_width, goal_height), resample=Image.Resampling.LANCZOS)
        elif height < goal_height:
            resized_image = Image.new(image.mode, (width, goal_height), 0)
            resized_image.paste(image, (0, top))
            resized_image = resized_image.resize(
                (goal_width, goal_height), resample=Image.Resampling.LANCZOS)
        else:
            resized_image = image.resize(
                (goal_width, goal_height), resample=Image.Resampling.LANCZOS)
        return resized_image

    def clean_image(self, path, show_images: bool = False):
        """
        Attempts to fill gaps of individual characters where the scanning has 
        failed. The techniques used will be to blur the image, then take a 
        threshold to convert back to binary.
        INPUT: Resized image
        OUTPUT: Cleaned image
        """
        image = np.array(cv.imread(path, 0))
        image_copy = image.copy()
        image_blurred = cv.GaussianBlur(image_copy, ksize=(5, 5), sigmaX=5,
                                        sigmaY=5)
        image_thresh = cv.threshold(
            image_blurred, 175, 255, cv.THRESH_BINARY)[1]

        if show_images:
            cv.namedWindow("Input Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Blurred Image", cv.WINDOW_NORMAL)
            cv.namedWindow("Threshold Image", cv.WINDOW_NORMAL)
            cv.imshow("Input Image", image)
            cv.imshow("Blurred Image", image_blurred)
            cv.imshow("Threshold Image", image_thresh)
            cv.waitKey()
            cv.destroyAllWindows()
            cv.waitKey(1)

        return image_thresh

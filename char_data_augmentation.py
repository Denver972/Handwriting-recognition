# Methods to suplement training dataset

"""
packages used
"""
import math
import numpy as np
import pandas as pd
import cv2 as cv


class DataAugmentation():
    """
    Augment the training data by: Erosion causing thiner lines in text: Done
                                  Dilation causing thicker lines in the text: Done
                                  Random black pixels maybe
                                  Random white pixels maybe
                                  Black lines throught the image: Done
                                  White lines through the image : Done
                                  Blurring the image if grayscale
                                  Smudging in the image :Achieved with white lines
                                  TODO: Combination of techniques
    """

    def __init__(self, csv_file):
        """
        Here are the kernels used and randomness of the pixels/lines occuring
        """
        self.file = csv_file
        # self.datast_folder = "./RowDataset"
        # os.mkdir(self.datast_folder)
        self.df = pd.read_csv(csv_file)
        self.erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        self.dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    def all_augmentation(self, name):
        """
        Creates csv and images with all the augmentation techniques
        """

        erode = self.erosion()
        dilate = self.dilation()
        white = self.white_lines()
        black = self.black_lines()

        df_merged = pd.concat([self.df, erode, dilate, white, black])
        df_merged.to_csv(name, index=False)

        return 0

    def erosion(self):
        """
        INPUT: dataframe of the csv file
        OUTPUT: dataframe with the augmented data paths
        """
        df_erode = self.df.copy()
        df_erode["CharacterPath"] = df_erode["CharacterPath"].str.replace(
            "skeleton", "thinned")

        for name in self.df["CharacterPath"]:
            image = np.array(cv.imread(name, 0))
            image_copy = image.copy()
            image_thin = cv.erode(image_copy, self.erode_kernel, iterations=1)
            name = name.replace("skeleton", "thinned")
            cv.imwrite(name, image_thin)

        df_merged = pd.concat([self.df, df_erode])
        df_merged.to_csv("CharDatasetEroded.csv", index=False)

        return df_erode

    def dilation(self):
        """
        INPUT: dataframe of the csv file
        OUTPUT: dataframe with the augmented data paths
        """
        df_dilate = self.df.copy()
        df_dilate["CharacterPath"] = df_dilate["CharacterPath"].str.replace(
            "skeleton", "thickened")

        for name in self.df["CharacterPath"]:
            image = np.array(cv.imread(name, 0))
            image_copy = image.copy()
            image_thin = cv.dilate(
                image_copy, self.dilate_kernel, iterations=1)
            name = name.replace("skeleton", "thickened")
            cv.imwrite(name, image_thin)

        df_merged = pd.concat([self.df, df_dilate])
        df_merged.to_csv("CharDatasetDilated.csv", index=False)

        return df_dilate

    def white_lines(self):
        """
        INPUT: dataframe of the csv file
        OUTPUT: dataframe with the augmented data paths
        """
        df_white_lines = self.df.copy()
        df_white_lines["CharacterPath"] = df_white_lines["CharacterPath"].str.replace(
            "skeleton", "white_lines")

        for name in self.df["CharacterPath"]:
            image = np.array(cv.imread(name, 0))
            image_copy = image.copy()
            # change only a maximum of 35% of the columns and rows
            rows, cols = image_copy.shape
            row_sample = np.random.choice(math.floor(0.35*rows), 1)
            col_sample = np.random.choice(math.floor(0.35*cols), 1)
            rows_changed = np.random.choice(rows, row_sample, replace=False)
            cols_changed = np.random.choice(cols, col_sample, replace=False)
            image_copy[rows_changed, :] = 255
            image_copy[:, cols_changed] = 255
            name = name.replace("skeleton", "white_lines")
            cv.imwrite(name, image_copy)
        df_merged = pd.concat([self.df, df_white_lines])
        df_merged.to_csv("CharDatasetWhiteLines.csv", index=False)

        return df_white_lines

    def black_lines(self):
        """
        INPUT: dataframe of the csv file
        OUTPUT: dataframe with the augmented data paths
        """
        df_black_lines = self.df.copy()
        df_black_lines["CharacterPath"] = df_black_lines["CharacterPath"].str.replace(
            "skeleton", "black_lines")

        for name in self.df["CharacterPath"]:
            image = np.array(cv.imread(name, 0))
            image_copy = image.copy()
            # change only a maximum of 35% of the columns and rows
            rows, cols = image_copy.shape
            row_sample = np.random.choice(math.floor(0.35*rows), 1)
            col_sample = np.random.choice(math.floor(0.35*cols), 1)
            rows_changed = np.random.choice(rows, row_sample, replace=False)
            cols_changed = np.random.choice(cols, col_sample, replace=False)
            image_copy[rows_changed, :] = 0
            image_copy[:, cols_changed] = 0
            name = name.replace("skeleton", "black_lines")
            cv.imwrite(name, image_copy)
        df_merged = pd.concat([self.df, df_black_lines])
        df_merged.to_csv("CharDatasetBlackLines.csv", index=False)

        return df_black_lines

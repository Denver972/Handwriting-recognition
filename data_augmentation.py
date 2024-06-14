# Methods to suplement training dataset

"""
packages used
"""

import numpy as np
import pandas as pd
import cv2 as cv


class DataAugmentation():
    """
    Augment the training data by: Erosion causing thiner lines in text
                                  Dilation causing thicker lines in the text
                                  Random black pixels maybe
                                  Random white pixels maybe
                                  Black lines throught the image
                                  White lines through the image
                                  Blurring the image if grayscale
                                  Smudging in the image
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

    def all_augmentation(self):
        """
        Creates csv and images with all the augmentation techniques
        """
        erode = self.erosion()
        dilate = self.dilation()

        df_merged = pd.concat([self.df, erode, dilate])
        df_merged.to_csv("RowDatasetAugmented.csv", index=False)

        return 0

    def erosion(self):
        """
        INPUT: dataframe of the csv file
        OUTPUT: dataframe with the augmented data paths
        """
        df_erode = self.df.copy()
        df_erode["CellPath"] = df_erode["CellPath"].str.replace(
            "resized", "thinned")

        for name in self.df["CellPath"]:
            image = np.array(cv.imread(name, 0))
            # print(image)
            image_copy = image.copy()
            image_thin = cv.erode(image_copy, self.erode_kernel, iterations=1)
            name = name.replace("resized", "thinned")
            cv.imwrite(name, image_thin)

        df_merged = pd.concat([self.df, df_erode])
        df_merged.to_csv("RowDatasetEroded.csv", index=False)

        return df_erode

    def dilation(self):
        """
        INPUT: dataframe of the csv file
        OUTPUT: dataframe with the augmented data paths
        """
        df_dilate = self.df.copy()
        df_dilate["CellPath"] = df_dilate["CellPath"].str.replace(
            "resized", "thickened")

        for name in self.df["CellPath"]:
            image = np.array(cv.imread(name, 0))
            # print(image)
            image_copy = image.copy()
            image_thin = cv.dilate(
                image_copy, self.dilate_kernel, iterations=1)
            name = name.replace("resized", "thickened")
            cv.imwrite(name, image_thin)

        df_merged = pd.concat([self.df, df_dilate])
        df_merged.to_csv("RowDatasetThickened.csv", index=False)

        return df_dilate

    def white_lines(self):
        """
        INPUT: dataframe of the csv file
        OUTPUT: dataframe with the augmented data paths
        """
        return 0

    def black_lines(self):
        """
        INPUT: dataframe of the csv file
        OUTPUT: dataframe with the augmented data paths
        """
        return 0

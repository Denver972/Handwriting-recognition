# Test file to try out parts of code
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
"""
Testing file
"""
import numpy as np
import pandas as pd
import cv2
import os
from pre_processing import FileSeparation, ImageRotation, TableDetect, WordExtraction, ColumnExtraction, TableExtraction, RowExtraction, CharacterExtraction, FileConstructor, PreProcess
# import fitz
import timeit
import time
import torch

FILE = "MW1959.pdf"
# FILE = "Handwriting-recognition/temp/grid2_test.png"

# image_color = np.array(cv2.imread(FILE, 1))
# image_grey = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
# image_blur = cv2.GaussianBlur(
#     image_grey, ksize=(15, 15), sigmaX=10, sigmaY=10)
# image_thresh = cv2.threshold(
#     image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
# image_dilated = cv2.dilate(image_thresh, kernel, iterations=5)

# print(image_color.shape)
# # # imS = cv2.resize(image_color, (960, 540))
# # # plt.imshow(image_color)
# # # plt.show()
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# cv2.imshow("output", image_dilated)
# # # cv2.imshow("output", imS)
# cv2.waitKey(0)


# image_test = ImageRotation()

# image_test.skew_angle(file=FILE, show_images=True)
# for ix in range(0, 59):
#     file_temp = f"./PNGFolder/test{ix}.png"
#     print(image_test.skew_angle(file=file_temp, show_images=False))
# print(image_test.skew_angle(file=FILE))
# image_test.rotate_image(file=FILE)

# test_separation = FileSeparation()
# # test_separation.folder_creation()
# test_separation.file_split(file=FILE)
# test_separation.pdf_to_png(file=FILE)

# table = TableDetect()
# table.remove_lines(file=FILE, show_images=True)
# table.rows(file="result.png", show_images=True)

# words = WordExtraction()
# words.cell_locate(file=FILE, show_images=True)
# words.extraction(file=FILE)

# table = TableExtraction()
# table.table_locate(file=FILE, show_images=True)
# table.extraction(file=FILE)

# column = ColumnExtraction()
# column.col_locate(file="result.png", show_images=True)
# column.extraction(fileIn="NoVertLines.png", fileOut="Table.png")

# row = RowExtraction()
# row.row_locate(file=FILE, show_images=True)

# Below for loop extracts rows from columns
# for col in range(0, 11):
#     file = f"./test_cont_extract/column{col}.png"  # all lines included
#     fileOutCol = f"./column{col}.png"  # no horizontal lines
#     print(file)
#     # remove horizontal lines and place new image in fileOutCol
#     table = TableDetect()
#     table.remove_lines(file=file, fileOut=fileOutCol)
#     # loop through each fileOutCol and extract rows to their own folder
#     # called COLUMN{ix}
#     col_folder = f"COLUMN{col}"
#     os.mkdir(col_folder)
#     for r in range(0, 41):
#         fileOutRow = f"./COLUMN{col}"
#         row = RowExtraction()
#         # row.row_locate(file=fileOutCol, show_images=True)
#         row.extraction(fileIn=fileOutCol, fileOut=fileOutRow)

# # Character extraction
# for row in range(0, 41):
#     char = CharacterExtraction()
#     file = f"COLUMN2/row{row}.png"
#     char.char_locate(file=file, show_images=True)

# Test split and file organisation

start_time = timeit.default_timer()
test = PreProcess(file=FILE, year=1959)
test.construct()
print(timeit.default_timer() - start_time)

# print(np.array(im.show("./Year_1959/")))


# test model on actual data
# conv_model = torch.load("./TestModelDigits.pt")

# create dataframe of of the csv with just the ones that have labels

# data = pd.read_csv("1959Characters copy2.csv")
# print(data.columns)
# no_missing = data.dropna()
# print(no_missing.shape)
# # remove the labels from sheet0 as they are typed(first 69 characters)
# no_typed = no_missing.iloc[69:]
# # remove first column
# no_typed = no_typed.drop("Unnamed: 0", axis="columns")
# no_typed.to_csv("Dataset1.csv", index=False)

# # make a training set and a testing set
# data = pd.read_csv("Dataset1.csv")
# print(data.head())
# # print(data.head())
# # print(data.info())
# # print(data.columns)
# # print(data.at[1, "Label"])
# # train = data.sample(frac=0.8, random_state=200)
# # test = data.drop(train.index)
# # train.to_csv("Training1.csv", index=False)
# # test.to_csv("Testing1.csv", index=False)
# # print(data.nunique())
# lbl = data.Label

# print(lbl)

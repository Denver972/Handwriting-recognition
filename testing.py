# Test file to try out parts of code
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
"""
Testing file
"""
from row_pre_processing import PreProcess as rowPreProcess
import matplotlib.pyplot as plt
import torch
import time
import timeit
import numpy as np
import pandas as pd
import cv2 as cv
import os
from pre_processing_test import FileSeparation, ImageRotation, TableDetect, WordExtraction, ColumnExtraction, TableExtraction, RowExtraction, CharacterExtraction, FileConstructor, PreProcess
from char_data_augmentation import DataAugmentation
from row_pre_processing import PreProcess as rowPreProcess
# import fitz

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

# FILE = "Year_NoSmall1959/Sheet1/CropTable.png"
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
# start_time = timeit.default_timer()
# test = PreProcess(file=FILE, year=1959)
# test.construct()
# print(timeit.default_timer() - start_time)


# FILENAME = "./Year_1959/Sheet1/ColumnFolder3/RowFolder5/binary10.png"

# test = PreProcess(file=FILENAME, year=1959)
# test.skeletonize(path=FILENAME, show_images=True)
# test.convert_to_binary(path=FILENAME, show_images=True)
# print(np.array(im.show("./Year_1959/")))


# test model on actual data
# conv_model = torch.load("./TestModelDigits.pt")

# create dataframe of of the csv with just the ones that have labels

# data = pd.read_csv("1959Characters_new.csv")
# print(data.columns)
# no_missing = data.dropna()
# print(no_missing.shape)
# # remove the labels from sheet0 as they are typed(first 69 characters)
# no_typed = no_missing.iloc[69:]
# # remove first column
# # no_typed = no_typed.drop("Unnamed: 0", axis="columns")
# no_typed.to_csv("Dataset4.csv", index=False)
##########################################################
# make a training set and a testing set
# data = pd.read_csv("Dataset4.csv")
# print(data.head())
# print(data.info())
# print(data.columns)
# print(data.at[1, "Label"])
# train = data.sample(frac=0.8, random_state=200)
# test = data.drop(train.index)
# train.to_csv("Training4.csv", index=False)
# test.to_csv("Testing4.csv", index=False)
# print(data.nunique())
# lbl = data.Label
# whole = pd.read_csv("Dataset4.csv")
# train = pd.read_csv("Training4.csv")
# test = pd.read_csv("Testing4.csv")
# print(whole["Class"].value_counts())
# print(train["Class"].value_counts())
# print(test["Class"].value_counts())
# print(lbl)
######## Helper function to change labels to numbers and numbers to labels######
# df = pd.read_csv("CharactersTest3Model9Augmented.csv")
# # print(data["Label"].unique())
# classes = {
#     "0": 0,
#     "1": 1,
#     "2": 2,
#     "3": 3,
#     "4": 4,
#     "5": 5,
#     "6": 6,
#     "7": 7,
#     "8": 8,
#     "9": 9,
#     "m": 10,
#     # "M": 10,
#     # "/": 1,
#     ".": 11,
#     "-": 12,
#     "+": 13,
#     "c": 14,
#     "'": 15,
#     "w": 16,
#     # "W": 16,
#     "s": 17,
#     # "S": 17,
#     "d": 18,
#     "N": 19,
#     "e": 20,
#     "H": 21
# }
# num_label = data["Label"].map(classes)
# print(num_label)
# data["Class"] = num_label
# data.to_csv("Testing4.csv", index=False)
# TODO: figure out how to invert correctly
# df = pd.read_csv("PredictedEarlyThreshold.csv")
# inv_classes = {v: k for k, v in classes.items()}
# class_label = df["PredClass"].map(inv_classes)
# df["Label"] = class_label
# df.to_csv("CharactersTest3Model9Augmented.csv", index=False)

# print(data.head())

####################################################
# Create the two columns needed for the dataframe
# df = pd.read_csv("1959CharactersEarlyThreshold.csv")
# df["Label"] = 0
# df["Class"] = 0
# df.to_csv("1959CharactersEarlyThreshold.csv", index=False)

##############################################################################

# Replace part of a string with another string
# data = pd.read_csv("PredictedNoSmall.csv")
# # print(data.head())
# data["CharacterPath"] = data["CharacterPath"].str.replace(
#     "skeleton", "binary")
# # data.replace("resized", "binary", inplace=True)
# print(data.head())
# data.to_csv("PredictedNoSmall.csv", index=False)

########## FREQUENCY############

# data = pd.read_csv("Testing4.csv")
# print(data["Class"].value_counts())
############# NUMBER DETECTION#############
# data = pd.read_csv("Dataset4.csv")
# print(data["Label"].unique())
# classes = {
#     "0": 0,
#     "1": 0,
#     "2": 0,
#     "3": 0,
#     "4": 0,
#     "5": 0,
#     "6": 0,
#     "7": 0,
#     "8": 0,
#     "9": 0,
#     "m": 1,
#     "M": 1,
#     "/": 0,  # May need to consider this as part of the 1s
#     ".": 1,
#     "-": 1,
#     "+": 1,
#     "c": 1,
#     "'": 1,
#     "w": 1,
#     "W": 1,
#     "s": 1,
#     "S": 1,
#     "d": 1,
#     "N": 1,
#     "e": 1,
#     "H": 1
# }
# num_label = data["Label"].map(classes)
# # print(num_label)
# data["Number"] = num_label
# print(data.head())
# data.to_csv("Dataset5.csv", index=False)

#################### Remaking the csv file###############
# Data available: sheet, column, row and character placement
# start with sheet

# Create a new dataframe that contains the orriginal file path and then
# split into two at the row folder part

# START OF SPLITING THE DATAFRAME
# whole_data = pd.read_csv("RowDataset.csv")
# whole_data[["Root", "Year", "Sheet", "ColumnIndex", "RowIndex", "CharIndex"]
#            ] = whole_data.CharacterPath.str.split("/", expand=True)
# whole_data["CharIndex"] = whole_data["CharIndex"].str.replace("skeleton", "")
# whole_data["CharIndex"] = whole_data["CharIndex"].str.replace(".png", "")
# whole_data["RowIndex"] = whole_data["RowIndex"].str.replace("RowFolder", "")
# whole_data["ColumnIndex"] = whole_data["ColumnIndex"].str.replace(
#     "ColumnFolder", "")
# whole_data["Sheet"] = whole_data["Sheet"].str.replace("Sheet", "")
# whole_data["Year"] = whole_data["Year"].str.replace("Test3Year_", "")

# print(whole_data)
# upd_data = whole_data.drop(["CharacterPath", "Root"], axis=1)
# upd_data.to_csv("UpdatedTest3Model9.csv", index=False)

# END OF SPLITTING THE DATAFRAME


# whole_data = pd.read_csv("RowDataset.csv")
# whole_data[["Root", "Year", "Sheet", "ColumnIndex", "RowIndex"]
#            ] = whole_data.CellPath.str.split("/", expand=True)
# # whole_data["CharIndex"] = whole_data["CharIndex"].str.replace("skeleton", "")
# # whole_data["CharIndex"] = whole_data["CharIndex"].str.replace(".png", "")
# whole_data["RowIndex"] = whole_data["RowIndex"].str.replace("resized", "")
# whole_data["RowIndex"] = whole_data["RowIndex"].str.replace(".png", "")
# whole_data["ColumnIndex"] = whole_data["ColumnIndex"].str.replace(
#     "ColumnFolder", "")
# whole_data["Sheet"] = whole_data["Sheet"].str.replace("Sheet", "")
# whole_data["Year"] = whole_data["Year"].str.replace("Test3Year_", "")

# print(whole_data)
# upd_data = whole_data.drop(["CellPath", "Root"], axis=1)
# upd_data.to_csv("Updated_row.csv", index=False)

############## EXTRACT ONE CELL############

# df = pd.read_csv("Updated1959.csv")
# cell_block = df[(df["Sheet"] == 0) & (
#     df["ColumnIndex"] == 0) & (df["RowIndex"] == 0)]
# print(cell_block)

# # combine the characters into one cell
# cell = []
# for char in cell_block["Label"]:
#     cell.append(char)

# print("".join(cell))


##########################################
# START OF MERGING THE ROW COLUMNS
# want to fix the sheet and column index then merge the label column based on
# its charIndex. For this introduce a new column CellText

# df = pd.read_csv("UpdatedTest3Model9.csv")
# df = df.dropna(axis=0, subset="Label")
# print(df)


# # # Sheet Number
# for sx in df["Sheet"].unique():
#     # extract the number of columns in the sheet
#     sheet_list = []
#     column_list = []
#     row_list = []
#     cell_list = []
#     sheet_df = df[df["Sheet"] == sx]
#     # Column Number
#     for cx in sheet_df["ColumnIndex"].unique():
#         # extract the number of rows in the column
#         column_df = sheet_df[sheet_df["ColumnIndex"] == cx]
#         # row number
#         for rx in column_df["RowIndex"].unique():
#             # extract the number of characters in this cell
#             # cell_block = df[(df["Sheet"] == sx) & (
#             #     df["ColumnIndex"] == cx) & (df["RowIndex"] == rx)]
#             # print(cell_block)
#             cell = []
#             row_df = column_df[column_df["RowIndex"] == rx]
#             # print(row_df)
#             # iterate through the characters
#             # for char in cell_block["Label"]:
#             # change label between pred class and label can remove str when the
#             # the label is used ina column with characters and numbers
#             for char in row_df["Label"]:
#                 # append the character to the list
#                 cell.append(str(char))
#             # print the cell value
#             value = "".join(cell)
#             # print(f"{sx}, {cx}, {rx}")
#             # print(value)
#             sheet_list.append(sx)
#     # print(cell)
#     for sx in sheet_df["Sheet"].unique():
#         for cx in sheet_df["Column"].unique():
#             col_df = sheet_df[sheet_df["Column"] == cx]
#             # print(col_df)
#             # print("New Column")
#             for rx in col_df["Row"].unique():
#                 row_df = col_df[col_df["Row"] == rx]
#                 # print(row_df)
#                 cell_value = row_df["Cell"].values
#                 # sheet_df[(sheet_df["Column"] == cx)
#                 #                       & (sheet_df["Row"] == rx)]["Cell"]
#                 # print(cell_value)
#                 # TODO: Update to work with pandas 3.0
#                 df.loc[rx][cx] = cell_value
#                 # df.loc[rx, f"{cx}"]
#     # print(df)
#     df.to_csv(f"Model9Results/Test{px}.csv", index=False)            column_list.append(cx)
#             row_list.append(rx)
#             cell_list.append(value)
#             # Create new dataframe with the sheet, column, row and cell entry
#             csv_dict = {"Sheet": sheet_list, "Column": column_list,
#                         "Row": row_list, "Cell": cell_list}
#             text_df = pd.DataFrame(csv_dict)
#             text_df.to_csv(f"Model9Results/Sheet{sx}.csv", index=False)
#####################################
############# TAKE LONG DATA AND CHANGE TO TABLE FORMAT############
# Input: Sheet

# for px in range(0, 75):
#     sheet_df = pd.read_csv(f"Model9Results/Sheet{px}.csv")
#     # get column indeces
#     column_index = range(0, (max(sheet_df["Column"].unique())+1))
#     # get row indeces
#     row_index = range(0, (max(sheet_df["Row"].unique())+1))
#     # print(row_index[1])
#     df = pd.DataFrame(index=row_index, columns=column_index)
#     # df = pd.DataFrame(index=row_index)
#     # cell = sheet_df.loc[0].at["Cell"]
#     # sheet_df[(sheet_df["Column"] == 0)
#     #                 & (sheet_df["Row"] == 1)]["Cell"]
#     # print(cell)
#     for sx in sheet_df["Sheet"].unique():
#         for cx in sheet_df["Column"].unique():
#             col_df = sheet_df[sheet_df["Column"] == cx]
#             # print(col_df)
#             # print("New Column")
#             for rx in col_df["Row"].unique():
#                 row_df = col_df[col_df["Row"] == rx]
#                 # print(row_df)
#                 cell_value = row_df["Cell"].values
#                 # sheet_df[(sheet_df["Column"] == cx)
#                 #                       & (sheet_df["Row"] == rx)]["Cell"]
#                 # print(cell_value)
#                 # TODO: Update to work with pandas 3.0
#                 df.loc[rx][cx] = cell_value
#                 # df.loc[rx, f"{cx}"]
#     # print(df)
#     df.to_csv(f"Model9Results/Test{px}.csv", index=False)

# sheet_df = pd.read_csv("Updated_row.csv")
# sheet_chosen = sheet_df[sheet_df["Sheet"] == 61]
# # get column indeces
# column_index = range(0, (max(sheet_chosen["Column"].unique())+1))
# # get row indeces
# row_index = range(0, (max(sheet_chosen["Row"].unique())+1))
# # print(row_index[1])
# df = pd.DataFrame(index=row_index, columns=column_index)
# # df = pd.DataFrame(index=row_index)
# # cell = sheet_df.loc[0].at["Cell"]
# # sheet_df[(sheet_df["Column"] == 0)
# #                 & (sheet_df["Row"] == 1)]["Cell"]
# # print(cell)
# for sx in sheet_chosen["Sheet"].unique():
#     for cx in sheet_chosen["Column"].unique():
#         col_df = sheet_chosen[sheet_chosen["Column"] == cx]
#         # print(col_df)
#         # print("New Column")
#         for rx in col_df["Row"].unique():
#             row_df = col_df[col_df["Row"] == rx]
#             # print(row_df)
#             cell_value = row_df["Cell"].values
#             # sheet_df[(sheet_df["Column"] == cx)
#             #                       & (sheet_df["Row"] == rx)]["Cell"]
#             # print(cell_value)
#             # TODO: Update to work with pandas 3.0
#             df.loc[rx][cx] = cell_value
#             # df.loc[rx, f"{cx}"]
# # print(df)
# df.to_csv("True_sheet61.csv", index=False)


################################
# read the csv file
# df = pd.read_csv("Model9Results/Test15.csv")
# print(df)
#################################
# Just numbers so predlabel = label
# FILE = "UpdatedTest3.csv"
# df = pd.read_csv(FILE)
# df["Label"] = df["PredClass"]
# df.to_csv("PredictedCharactersTest3.csv", index=False)
# df["Label"]

# test = PreProcess(file=FILE, year=1959)
# test.create_sheet_csv(num_of_sheets=75)
# df = pd.read_csv("Test3Folder/Test1.csv")
# df = df.apply(str)
# print(df)
# Augmarntation
# FILE = "Training6.csv"
# test = DataAugmentation(csv_file=FILE)
# test.all_augmentation(name="Training6DatasetAugmented.csv")
# Histogram to separate characters
# FILE = "RowTest2Year_1959/Sheet1/ColumnFolder1/resized7.png"
# test = CharacterExtraction()
# hist, peak = test.histogram(FILE, show_images=True)
# print(hist)
# print(len(hist))
# plt.plot(range(0, len(hist)), hist)
# plt.plot(peak, hist[peak], "x")
# plt.show()
######### Thinned Image############
# FILE = "./Year_NoSmall1959/Sheet1/ColumnFolder2/row8.png"
# 10 16 38
# FILE = "./RowTest2Year_1959/Sheet1/ColumnFolder1/resized3.png"
# row_test = rowPreProcess(FILE, year=1959)
# clean = row_test.clean_image(FILE, show_images=True)
# test = PreProcess(FILE, year=1959)
# thinned = test.thinning(clean, show_images=True)
# # print(np.shape(thinned))
# median = test.segmentation(thinned, show_images=True)
# test.split_image(clean, median, show_images=True)

# Test smoothed Image


# def smoothing(file, show_images: bool = False):
#     """
#     Take in a character and make neew image with it being smoothed
#     like with emnist
#     """
#     image = np.array(cv2.imread(file, 0))
#     image_copy = image.copy()
#     image_blur = cv2.GaussianBlur(image_copy, ksize=(3, 3), sigmaX=5,
#                                   sigmaY=5)

#     if show_images:
#         cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
#         cv2.namedWindow("Blurred Image", cv2.WINDOW_NORMAL)
#         cv2.imshow("Input Image", image)
#         cv2.imshow("Blurred Image", image_blur)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
#         cv2.waitKey(1)

#     return image_blur


# FILE = "./Win1Year_1959/Sheet0/ColumnFolder1/RowFolder10/skeleton1.png"
# smoothing(file=FILE, show_images=True)

# master = pd.read_csv("Win1EMNISTtest.csv")

# for ix, name in enumerate(master["CharacterPath"]):
#     temp = smoothing(file=name)
#     new_name = name.replace("skeleton", "smooth")
#     cv2.imwrite(new_name, temp)


# img1 = cv.imread("column1.png")
# img2 = cv.imread("thresh_col.png")
# img3 = cv.imread("dil_col.png")
# img4 = cv.imread("con_col.png")

# row = np.concatenate((img1, img2, img3, img4), axis=1)
# cv.imwrite("ColumnProcess.png", row)
# cv.imshow("Table", row)
# cv.waitKey(0)
# cv.destroyAllWindows()
# Contour demonstration
FILE = "./RowTest2Year_1959/Sheet1/ColumnFolder1/row3.png"  # 8,16,38,3
test = CharacterExtraction()
test.char_locate(file=FILE, show_images=True)

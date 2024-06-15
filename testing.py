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
from pre_processing_test import FileSeparation, ImageRotation, TableDetect, WordExtraction, ColumnExtraction, TableExtraction, RowExtraction, CharacterExtraction, FileConstructor, PreProcess
from char_data_augmentation import DataAugmentation
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
# data = pd.read_csv("Testing4.csv")
# print(data["Label"].unique())
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
# # print(num_label)
# data["Class"] = num_label
# data.to_csv("Testing4.csv", index=False)
# TODO: figure out how to invert correctly
# df = pd.read_csv("PredictedEarlyThreshold.csv")
# inv_classes = {v: k for k, v in classes.items()}
# class_label = df["PredClass"].map(inv_classes)
# df["Label"] = class_label
# df.to_csv("PredictedEarlyThreshold.csv", index=False)

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
# whole_data = pd.read_csv("PredictedEarlyThreshold.csv")
# whole_data[["Root", "Year", "Sheet", "ColumnIndex", "RowIndex", "CharIndex"]
#            ] = whole_data.CharacterPath.str.split("/", expand=True)
# whole_data["CharIndex"] = whole_data["CharIndex"].str.replace("skeleton", "")
# whole_data["CharIndex"] = whole_data["CharIndex"].str.replace(".png", "")
# whole_data["RowIndex"] = whole_data["RowIndex"].str.replace("RowFolder", "")
# whole_data["ColumnIndex"] = whole_data["ColumnIndex"].str.replace(
#     "ColumnFolder", "")
# whole_data["Sheet"] = whole_data["Sheet"].str.replace("Sheet", "")
# whole_data["Year"] = whole_data["Year"].str.replace("Year_", "")

# print(whole_data)
# upd_data = whole_data.drop(["CharacterPath", "Root"], axis=1)
# upd_data.to_csv("UpdatedEarlyThresh.csv", index=False)

# END OF SPLITTING THE DATAFRAME


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

# df = pd.read_csv("UpdatedEarlyThresh.csv")
# df = df.dropna(axis=0, subset="Label")
# print(df)


# # Sheet Number
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
#             for char in row_df["Label"]:
#                 # append the character to the list
#                 cell.append(char)
#             # print the cell value
#             value = "".join(cell)
#             # print(f"{sx}, {cx}, {rx}")
#             # print(value)
#             sheet_list.append(sx)
#             column_list.append(cx)
#             row_list.append(rx)
#             cell_list.append(value)
#             # Create new dataframe with the sheet, column, row and cell entry
#             csv_dict = {"Sheet": sheet_list, "Column": column_list,
#                         "Row": row_list, "Cell": cell_list}
#             text_df = pd.DataFrame(csv_dict)
#             text_df.to_csv(f"TestThreshFolder/Sheet{sx}.csv", index=False)
#####################################
############# TAKE LONG DATA AND CHANGE TO TABLE FORMAT############
# Input: Sheet

# for px in range(0, 75):
#     sheet_df = pd.read_csv(f"TestThreshFolder/Sheet{px}.csv")
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
#     df.to_csv(f"TestThreshFolder/Test{px}.csv", index=False)

################################
# read the csv file
# df = pd.read_csv("TestThreshFolder/Test12.csv")
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
FILE = "Training5.csv"
test = DataAugmentation(csv_file=FILE)
test.all_augmentation(name="Training5DatasetAugmented.csv")

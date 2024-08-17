"""
Testing file
"""
import numpy as np
import pandas as pd
import cv2
import os
# from pre_process_windows import FileSeparation, ImageRotation, TableDetect, WordExtraction, ColumnExtraction, TableExtraction, RowExtraction, CharacterExtraction, FileConstructor, PreProcess
from char_data_augmentation import DataAugmentation
import timeit
import time
import torch
import torchvision
import matplotlib.pyplot as plt

# FILE = "MW1959.pdf"

# # Test split and file organisation
# start_time = timeit.default_timer()
# test = PreProcess(file=FILE, year=1959)
# test.construct()
# print(timeit.default_timer() - start_time)

# full_data = torchvision.datasets.EMNIST(
#     root="emnistFolder", split="balanced", download=True)
# print(full_data.class_to_idx)

# FILE = "Win1EMNISTtest.csv"
# csv = pd.read_csv(FILE)
# print(csv["PredClass"].value_counts())

##### Add labels to the completed csv#######

##### Table to html########
col_names = ["Time EST.", "No. of Nuclei (per box, beam, cc)",
             "Outside Temp. F", "Box Temp Fill bfr", "Box Temp Fill aft",
             "Box Temp Count bfr", "Box Temp Count aft",
             "Wait betw. fill and count (min)",
             "Pres wx (ww)", "Vis. (ft, mi.)", "Wind Dir.",
             "Wind Speed (mph)", "Cloud amount (tenths)",
             "Cloud Types (all)", "Cloud est ht (ft. over sta.)",
             "Rel. Hum. (%)", "Observer"]
times = ["0100", "0300", "0700", "1000", "1300", "1600", "1900", "2200"]
times_series = pd.Series(times)


df = pd.read_csv("True_sheet0.csv")
df.columns = col_names
df["Time EST."] = np.resize(times_series, df.shape[0])
df.to_html("True_sheet0.htm")
# Numbers only csv
# df = pd.read_csv("Dataset5.csv")
# number_df = df[df["Number"] == 0]
# number_df.to_csv("Number_dataset.csv", index=False)

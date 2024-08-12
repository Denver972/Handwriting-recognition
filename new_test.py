"""
Testing file
"""
import numpy as np
import pandas as pd
import cv2
import os
from pre_process_windows import FileSeparation, ImageRotation, TableDetect, WordExtraction, ColumnExtraction, TableExtraction, RowExtraction, CharacterExtraction, FileConstructor, PreProcess
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

full_data = torchvision.datasets.EMNIST(
    root="emnistFolder", split="balanced", download=True)
print(full_data.class_to_idx)

# FILE = "Win1EMNISTtest.csv"
# csv = pd.read_csv(FILE)
# print(csv["PredClass"].value_counts())
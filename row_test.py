# import numpy as np
# import pandas as pd
# import cv2 as cv
# import os
import timeit
# import time
from row_pre_processing import PreProcess
from data_augmentation import DataAugmentation
# import fitz

### PRE PROCESS###
# FILE = "MW1959.pdf"
# start_time = timeit.default_timer()
# test = PreProcess(file=FILE, year=1959)
# test.construct()
# print(timeit.default_timer() - start_time)
################
### DATA AUGMENTATION###
FILE = "RowDataset.csv"
test = DataAugmentation(csv_file=FILE)
test.all_augmentation()

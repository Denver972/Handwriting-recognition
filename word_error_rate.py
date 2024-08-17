"""
Calculates the portion of cells that are correct in a datasheet
"""
import numpy as np
import pandas as pd


def word_error_rate(file_true, file_predicted):
    """
    Input: Two html files containing true cells and the other with predicted cells
    Outpit: Portion of cells that are correct
    """

    true_sheet = pd.read_csv(file_true)
    pred_sheet = pd.read_csv(file_predicted)
    # convert to dataframes (for html)
    true_df = true_sheet
    pred_df = pred_sheet

    # replace Na's in predicted with EMPTY to match true sheet
    pred_df = pred_df.fillna("EMPTY")

    # remove first column
    # true_df = true_df.drop(axis=1, index=0)
    # pred_df = pred_df.drop(axis=1, index=0)

    # Convert dataframes to an array
    true_arr = true_df.to_numpy()
    pred_arr = pred_df.to_numpy()
    print(true_arr)
    total = 0
    correct = 0
    for ix in range(0, len(true_arr[:, 1])):
        for jx in range(0, len(true_arr[1, :])):
            if true_arr[ix, jx] == pred_arr[ix, jx]:
                correct += 1
            total += 1
            print(true_arr[ix, jx], pred_arr[ix, jx])

    total = total - 40  # accounts for first column not being in true
    print(total)
    print(correct)
    return correct/total


wer = word_error_rate(file_true="True_sheet0.csv",
                      file_predicted="TestThreshFolder/Test0.csv")
print(wer)

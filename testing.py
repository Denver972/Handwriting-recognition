# Test file to try out parts of code
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
"""
Testing file
"""
# import numpy as np
# import cv2
from pre_processing import FileSeparation, ImageRotation, TableDetect, WordExtraction
# import fitz

FILE = "result.png"

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

words = WordExtraction()
words.cell_locate(file=FILE, show_images=True)
# words.extraction(file=FILE)

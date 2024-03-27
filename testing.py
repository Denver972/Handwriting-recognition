# Test file to try out parts of code
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
"""
Testing file
"""

from pre_processing import ImageRotation

FILE = "Handwriting-recognition/temp/test_page.png"

# image_color = np.array(cv2.imread(FILE, 1))
# print(image_color)
# # imS = cv2.resize(image_color, (960, 540))
# # plt.imshow(image_color)
# # plt.show()
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# cv2.imshow("output", image_color)
# # cv2.imshow("output", imS)
# cv2.waitKey(0)

image_test = ImageRotation()

# image_test.skew_angle(file=FILE, show_images=True)

print(image_test.skew_angle(file=FILE, show_images=True))

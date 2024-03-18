# Image preprocessing. The goal is to first tilt the image so text runs
# horizontally. Then blur the text to merge the text in each column
# together create bounding boxes around each column. Then potentially
# sharpen the image and within each bounding box, put bounding boxes
# around each character. These characters will be indexed with the
# column number, line they occupy, and the column they occupy. This
# should allow for the format to be reconstructed after handwriting
# recognition has been performed.

class PreProcess():
    """
    This class will include all the methods required to pre process an 
    image for handwriting recognition. Each method in the class should 
    be called in order to identify where the characters are.
    """

    def __init__(self, file_path):
        pass

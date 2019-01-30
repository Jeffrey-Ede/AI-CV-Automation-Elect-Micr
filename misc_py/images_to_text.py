from PIL import Image
import pytesseract
import cv2
import os

"""
Function to convert all the images in a directory to text

Author: Jeffrey Ede
Email: j.m.ede@warwick.ac.uk
"""

def images2text(dir, single_file=False):
    """Converts all the images in a directory into text"""

    text = ""
    files = [dir+f for f in os.listdir(dir)]

    for f in files:
        #Retrieve grayscale image
        print(f)
        image = cv2.imread(f.replace("/", "\\"), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #write the grayscale image to disk as a temporary file so we can 
        #apply the OCR function to it
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)

        #Extract text
        text += pytesseract.image_to_string(Image.open(filename), config='--psm 6') + "\n"

        #Remove temporary file
        os.remove(filename)

    return text

if __name__ == "__main__":

    loc = "H:/ch4o3/"
    text = images2text(loc)

    with open(loc+"text.txt", 'w') as f:
        f.write(text)
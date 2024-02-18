import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'F:/Program Files/tesseract.exe'

custom_config = r"--oem 3 --psm 11 -c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz'"

def recognize_text(img):
    
    img_1 = img.copy()
    # img_1 = cv2.bilateralFilter(img_1,5, 55,60)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    _, img_1 = cv2.threshold(img_1, 250, 255, 1)

    return pytesseract.image_to_string(img_1, lang='eng', config = custom_config)

def get_path_text(img_path):
    return recognize_text(cv2.imread(img_path))

def remove_text(orig):
    img = orig.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8) 
    img = cv2.dilate(img, kernel, iterations = 1)
    return cv2.inpaint(orig, img, 7, cv2.INPAINT_NS)
    # return cv2.inpaint(orig, img, 7, cv2.INPAINT_NS)

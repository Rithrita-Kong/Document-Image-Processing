import cv2
import numpy as np
import tkinter as tk
from imutils.perspective import four_point_transform
from imutils import rotate_bound
import pytesseract


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

root = tk.Tk()

Width = root.winfo_screenwidth() - 150
Height = root.winfo_screenheight() - 150

Path = "output2/"


def blurring_process(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    blurred_image = cv2.GaussianBlur(blurred_image, (5, 5), 0)

    return blurred_image


def binarization(image):
    # Perform thresholding
    _, thresholded = cv2.threshold(
        image, 100, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(
        thresholded, cv2.MORPH_CLOSE, kernel, iterations=20)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=10)

    return opened


def process_white(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Create an empty RGB image with the same dimensions as the original image
    rgb_image = np.zeros_like(image)

    # Set the R, G, and B channels in the RGB image to the B channel values
    rgb_image[:, :, 0] = B  # Set R channel to B channel
    rgb_image[:, :, 1] = B  # Set G channel to B channel
    rgb_image[:, :, 2] = B  # Set B channel to B channel

    return rgb_image


def edgeDetection(image):
    canny = cv2.Canny(image, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    imgDial = cv2.dilate(canny, kernel, iterations=5)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    return imgThreshold


def find_document_contour(image):
    document_contour = np.array(
        [[0.05 * Width, 0.05 * Height], [Width, 0], [0, Height], [Width, Height]])

    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) > 250:
        return None, None

    else:
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 1000:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.015 * peri, True)

                if area > max_area and len(approx) == 4:
                    document_contour = approx
                    max_area = area

        return document_contour, max_area


def displayImage(name, img):
    img_height, img_width = img.shape[:2]

    scalex = Width / img_width
    scaley = Height / img_height
    scale = min(scalex, scaley)

    cv2.imshow(name, cv2.resize(img, None, fx=scale, fy=scale))


def saveFile(path, detected, crop):
    cv2.imwrite(f'{path}/detected.png', detected)
    cv2.imwrite(f'{path}/scanned.png', crop)


def ocr(path, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_osd(
        rgb, output_type=pytesseract.Output.DICT)
    rotated_image = rotate_bound(image, angle=results["rotate"])
    file = open(f"{path}/recognized.txt", "w")
    ocr_text = pytesseract.image_to_string(rotated_image)
    print("Text in the document: \n")
    print(ocr_text)
    file.write(ocr_text)
    file.close()


def warping(image, document_contour, copy, binarize):
    # Perform perspective correction
    warped = four_point_transform(image, document_contour.reshape(4, 2))

    # Draw bounding rectangle on the original image
    x, y, w, h = cv2.boundingRect(document_contour.astype(np.int64))
    cv2.rectangle(copy, (x, y), (x+w, y+h), (0, 255, 0), 3)
    # result(image, binarize, copy, warped)
    return warped, copy


def result(original, morphed, contour, warped):
    displayImage('Orignal Image', original)
    displayImage("Morphed Image", morphed)
    displayImage("Contour Image", contour)
    displayImage('Warped Image', warped)
    cv2.imwrite('output/scanned.png', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run(original):
    status = False
    copy = original.copy()

    # Preprocess the image
    print("First Step: \n")
    blur = blurring_process(original)

    # Uncomment this part to apply perspective correction
    canny1 = edgeDetection(blur)
    # displayImage("Canny1", canny1)

    binarize1 = binarization(canny1)

    document_contour1, area1 = find_document_contour(binarize1)

    print("Second Step: \n")
    binarize2 = binarization(blur)
    canny2 = edgeDetection(binarize2)
    document_contour2, area2 = find_document_contour(canny2)
    # Check area
    if document_contour1 is not None:

        if document_contour2 is not None:
            if area1 > 200000 or area2 > 200000:
                status = True

            if area2 > area1:
                document_contour = document_contour2
                binarize = binarize2

                warp, copy = warping(
                    original, document_contour, copy, binarize)
                return warp, copy, status

            elif area1 > area2:
                document_contour = document_contour1
                binarize = binarize1

                warp, copy = warping(
                    original, document_contour, copy, binarize)
                return warp, copy, status

            else:
                print('\nGoing to B-Channel\n')
                b_channel = process_white(original)
                b_blur = blurring_process(b_channel)
                b_binarization = binarization(b_blur)
                b_canny = edgeDetection(b_binarization)
                document_contour, area = find_document_contour(b_canny)
                if document_contour is not None:
                    if area > 200000:
                        status = True
                    warp, copy = warping(
                        original, document_contour, copy, b_binarization)

                    return warp, copy, status
                else:
                    print("Can't find any document in the image")
                    status = False

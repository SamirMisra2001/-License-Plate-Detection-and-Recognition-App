import streamlit as st
import cv2
import imutils
import pytesseract
from PIL import Image
import numpy as np

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.title("License Plate Detection and Recognition")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Display the original image
    st.subheader("Original Image")
    st.image(image, channels="RGB")

    # Resize image
    resized_image = imutils.resize(image)

    # Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter
    filtered_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

    # Edge detection
    edged = cv2.Canny(filtered_image, 30, 200)

    # Find contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    # Initialize license plate contour and cropped image
    screenCnt = None
    license_plate_img = None
    
    image2 = image.copy()
    cv2.drawContours(image2,cnts,-1,(0,255,0),3)
    image_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Loop over contours to find the license plate
    i = 7
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4:
            screenCnt = approx
            x, y, w, h = cv2.boundingRect(c)
            license_plate_img = resized_image[y:y + h, x:x + w]
            i+=1
            break

    if screenCnt is not None:
        # Draw the contour on the original image
        cv2.drawContours(resized_image, [screenCnt], -1, (0, 255, 0), 3)

        # Display the image with detected license plate
        st.subheader("Image with Detected License Plate")
        st.image(resized_image, channels="RGB")

        # Display the cropped license plate image
        st.subheader("Cropped License Plate")
        st.image(license_plate_img, channels="RGB")

        # Recognize text from the license plate
        # text = pytesseract.image_to_string(license_plate_img, config='--psm 8')
        # st.subheader("Detected License Plate Text")
        # st.write(text)
    else:
        st.write("License plate could not be detected.")

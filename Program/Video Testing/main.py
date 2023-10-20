from functionfile import cv2, run, displayImage, saveFile, ocr

# Use for video files
path = "Program/Video Testing/Video"
video_path = f"{path}/whitebackground.mp4"
cap = cv2.VideoCapture(video_path)

# Use for IP Webcam
# cap = cv2.VideoCapture(0)
# address = "http://1.8.0.0:8080/video"
# cap.open(address)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


while True:
    _, frame = cap.read()

    # Check if the frame is clear or not and return the score (>90 is good based on our observation)
    laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
    print(laplacian_var)

    # Automatic Scanning condition
    displayImage("Video: ", frame)
    if laplacian_var > 150:
        frame_copy = frame.copy()
        cropped, detected, status = run(frame)

        displayImage("Detected: ", detected)
        displayImage("Cropped: ", cropped)
        if status == True and laplacian_var > 170:
            filename = f"{path}/../output/detect.png"

            displayImage("Detected", detected)

            # Save contour box image
            cv2.imwrite(filename, detected)
            filename = f"{path}/../output/scanned.png"

            # Save cropped image
            cv2.imwrite(filename, cropped)
            break

    pressed_key = cv2.waitKey(1) & 0xFF
    if pressed_key == 27:
        break

cap.release()
cv2.destroyAllWindows()

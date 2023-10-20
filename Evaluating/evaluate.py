import cv2
import os
from functionfiles import run
import keyboard  # Import the keyboard library

# Define ANSI escape codes for text color
GREEN_TEXT = '\033[92m'
RESET_COLOR = '\033[0m'


# Define the path to the folder containing your image files
folder_path = 'Dataset/Straight/Image/'

# List all files in the folder
image_files = [f for f in os.listdir(
    folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

total_files = len(image_files)
count = 1
# Initialize a counter for successful detections
successful_detections = 0

# Loop through each image file
for image_file in image_files:
    # Construct the full path to the current image file
    image_path = os.path.join(folder_path, image_file)

    # Show Progress
    print(f"Image File: {image_file}")
    print(f"Progress: {count} / {total_files}")

    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is not None:
        count += 1

        while True:
            # Process the image using your 'run' function
            cropped, detected = run(image)

            # Check if the key '9' was pressed using the keyboard library
            if keyboard.is_pressed('9'):
                successful_detections += 1
                print(GREEN_TEXT + "\nSuccessful Detection:",
                      successful_detections, RESET_COLOR)
                cv2.imwrite(
                    f"Dataset/Straight/Correct/Image/{count}.jpg", image)
                cv2.imwrite(
                    f"Dataset/Straight/Correct/Detected/{count}.jpg", detected)
                cv2.imwrite(
                    f"Dataset/Straight/Correct/Cropped/{count}.jpg", cropped)
                print("")
                break  # Exit the loop when '9' is pressed

            if keyboard.is_pressed('1'):
                cv2.imwrite(f"New Dataset/Straight/Wrong/{count}.jpg", image)
                break

            # Check if the key '0' was pressed to move to the next image
            if keyboard.is_pressed('0'):
                break  # Exit the loop when '0' is pressed

# Close any remaining OpenCV windows
cv2.destroyAllWindows()

# Print the total number of successful detections
print(GREEN_TEXT + "Total Successful Detections:",
      successful_detections, RESET_COLOR)

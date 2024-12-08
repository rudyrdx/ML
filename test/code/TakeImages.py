import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3448)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 808)
#framerate
cap.set(cv2.CAP_PROP_FPS, 30)
# Automatically adjust exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto exposure

# # Function to improve image processing
# def improve_image(image):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Apply histogram equalization
#     equalized = cv2.equalizeHist(gray)
#     # Convert back to BGR
#     improved = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
#     return improved

# def deflickr(image):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Apply a Gaussian blur to reduce noise and improve flicker removal
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Apply adaptive thresholding to remove flicker
#     deflickered = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     # Convert back to BGR
#     deflickered_bgr = cv2.cvtColor(deflickered, cv2.COLOR_GRAY2BGR)
#     return deflickered_bgr

def decode(frame):
    left = frame[:, 64:1280 + 48]
    right = frame[:, 1280 + 48:1280 + 48 + 1264]
    return left, right

for i in range(40):
    # Show the live camera feed with decoded frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Decode the frame into left and right images
        left, right = decode(frame)
        # left = improve_image(left)
        # right = improve_image(right)
        # Downsample the frames
        left_resized = cv2.resize(left, (640, 400))  # Resize to half the original size
        right_resized = cv2.resize(right, (640, 400))  # Resize to half the original size
        
        cv2.imshow("Decoded Frames (Left | Righwt)", left_resized)
        cv2.imshow("Decoded Frames (Left | Right)", right_resized)
        
        # Wait for user to press Enter or Esc
        key = cv2.waitKey(1)
        if key == 13:  # Enter key
            cv2.imwrite("picl" + str(i)+ ".jpg", left)
            cv2.imwrite("picr" + str(i)+ ".jpg", right)
            break
        elif key == 27:  # Esc key
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("done!")
cap.release()
cv2.destroyAllWindows()

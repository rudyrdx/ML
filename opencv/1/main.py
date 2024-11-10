# Write a code to convert the image from the given color model to different
# color models.

import cv2
import matplotlib.pyplot as plt

# Load the image in BGR (default color model in OpenCV)
image = cv2.imread('img.jpg')

# Convert to different color models
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Display all color models
color_models = {
"BGR (Original)": image,
"RGB": image_rgb,
"HSV": image_hsv,
"Grayscale": image_gray,
"YCrCb": image_ycrcb,
"LAB": image_lab
}

# Plot each color model
plt.figure(figsize=(10, 8))
for i, (title, img) in enumerate(color_models.items()):
    plt.subplot(2, 3, i + 1)
    if len(img.shape) == 2: # Grayscale image
        plt.imshow(img, cmap='gray')

    else: # Color image
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
plt.tight_layout()
plt.show()
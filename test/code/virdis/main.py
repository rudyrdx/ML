import cv2
import numpy as np

# Assuming you have functions to load or acquire your left and right images as NumPy arrays
# For example:
# left_img = cv2.imread('left_image.png')
# right_img = cv2.imread('right_image.png')
cap = cv2.VideoCapture(0)
cap.set(3, 3000)

def decode(frame):
    left_start = 48
    left_width = 128
    right_start = 1328  # Approx 1280 + 48
    right_width = 1280

    # Crop the images
    left = frame[:, left_start:left_start + left_width]
    right = frame[:, right_start:right_start + right_width]
    return left, right
# Replace these with your actual image acquisition code
left_img = ...  # Your code to get the left image
right_img = ...  # Your code to get the right image

# Convert images to grayscale (required for disparity computation)
gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Create StereoSGBM matcher object
window_size = 5  # Size of the block window. You can adjust this value.
min_disp = 0
num_disp = 16 * 8  # Must be divisible by 16

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Compute the disparity map
disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

# Replace all zero and negative disparity values with a small minimum value to avoid division by zero
disparity[disparity <= 0] = 0.1

# Camera parameters
B = 9.0  # Baseline in cm
f_cm = 30.0  # Focal length in cm

# Convert focal length from cm to pixels
# You need to know your camera's sensor width in cm to do this conversion accurately
# For example:
# sensor_width_cm = ...  # Sensor width in cm
# image_width_px = 1280  # Image width in pixels
# f_px = (f_cm / sensor_width_cm) * image_width_px

# If you do not have the sensor width, you can estimate the focal length in pixels if you know the field of view:
# f_px = (image_width_px / 2) / tan(FOV / 2)

# For now, let's assume you have the focal length in pixels from calibration
f_px = ...  # Focal length in pixels (you need to provide this value)

# Calculate the depth map
depth_map = (f_px * B) / disparity  # Depth in cm

# Normalize the depth map for visualization (optional)
depth_map_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_vis = np.uint8(depth_map_vis)

# Display the depth map
cv2.imshow('Depth Map', depth_map_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

# If you want to save the depth map image:
# cv2.imwrite('depth_map.png', depth_map_vis)
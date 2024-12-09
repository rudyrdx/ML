import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

# Load precomputed camera parameters
ret = np.load('param_ret.npy')   # Camera calibration success flag (unused here)
K = np.load('param_K.npy')       # Camera intrinsic matrix
dist = np.load('param_dist.npy') # Distortion coefficients
h, w = (800, 1264)               # Dimensions of the input frame

# Compute an optimal new camera matrix for undistortion
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

# Precompute undistortion and rectification maps for fast remapping
mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, new_camera_matrix, (w, h), cv2.CV_16SC2)

# Initialize video capture (change source as needed)
cap = cv2.VideoCapture(0)  # Use webcam as the video source
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3448)  # Set camera resolution width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 808)  # Set camera resolution height

# Create a kernel for morphological operations (used for denoising)
kernel = np.ones((13, 13), np.uint8)

# Stereo matcher settings
win_size = 4
min_disp = 10
max_disp = 16 * 2 + 10
num_disp = max_disp - min_disp  # Number of disparities (must be divisible by 16)

# Configure the StereoSGBM matcher for disparity calculation
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=5,
    uniquenessRatio=10,
    speckleWindowSize=1000,
    speckleRange=10,
    disp12MaxDiff=25,
    P1=8 * 3 * win_size**2,  # Penalty on the disparity smoothness (smaller regions)
    P2=32 * 3 * win_size**2  # Penalty on the disparity smoothness (larger regions)
)

# Function to split stereo frames into left and right images
def decode(frame):
    # Initialize empty arrays for left and right images
    left = np.zeros((800, 1264, 3), np.uint8)
    right = np.zeros((800, 1264, 3), np.uint8)

    # Extract left and right images from the input frame
    for i in range(800):
        left[i] = frame[i, 64:1280 + 48]
        right[i] = frame[i, 1280 + 48:1280 + 48 + 1264]

    return left, right

# Function to estimate distance based on disparity
def get_distance(d):
    return 30 * 10 / d  # Scale factor for distance calculation (arbitrary)

while True:
    ret, frame = cap.read()  # Read a frame from the video source
    if not ret:  # Exit loop if no frame is captured
        break

    start = time.time()  # Start timing for FPS calculation

    # Decode the stereo frame into left and right images
    right, left = decode(frame)

    # Downsample and convert images to grayscale for faster disparity computation
    img_1_downsampled = cv2.pyrDown(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
    img_2_downsampled = cv2.pyrDown(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))

    new_w, new_h = img_1_downsampled.shape  # Get dimensions of downsampled images

    # Compute the disparity map using the stereo matcher
    disp = stereo.compute(img_1_downsampled, img_2_downsampled)

    # Normalize disparity for visualization (step 1 of denoising)
    denoised = ((disp.astype(np.float32) / 16) - min_disp) / num_disp
    dispc = (denoised - denoised.min()) * 255
    dispC = dispc.astype(np.uint8)  # Convert to 8-bit image for visualization

    # Apply morphological closing to further denoise the disparity map (step 2)
    denoised = cv2.morphologyEx(dispC, cv2.MORPH_CLOSE, kernel)

    # Normalize the denoised disparity map for better visualization
    disp_normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)

    # Adjust the range: map closer objects to higher intensity
    disp_adjusted = disp_normalized.astype(np.uint8)

    # Apply a gradient colormap for multi-color depth visualization
    disp_Color = cv2.applyColorMap(disp_adjusted, cv2.COLORMAP_OCEAN)

    # Create a reprojection matrix (Q) for converting disparity to 3D points
    f = 0.3 * w  # Approximate focal length (in pixels)
    Q = np.float32([
        [1, 0, 0, -0.5 * new_w],  # X-axis offset
        [0, -1, 0, 0.5 * new_h],  # Y-axis offset (invert for correct orientation)
        [0, 0, 0, f],             # Focal length
        [0, 0, 1, 0]              # Depth scaling factor
    ])
    points = cv2.reprojectImageTo3D(disp, Q)  # Reproject disparity map to 3D space

    # Extract Z-values (depth) from the 3D points
    z_values = points[:, :, 2].flatten()
    indices = z_values.argsort()  # Sort Z-values for percentile calculations

    # Calculate 30% of total depth points for statistics
    percentage = 25280
    min_distance = np.mean(np.take(z_values, indices[0:percentage]))  # Average of lowest 30% distances
    avg_distance = np.mean(z_values)  # Average of all distances
    max_distance = np.mean(np.take(z_values, indices[-percentage:]))  # Average of highest 30% distances

    # Combine left image with colorized disparity map for visualization
    left_resized = cv2.resize(left, (632, 400))  # Resize left image for overlay
    color_depth = cv2.addWeighted(left_resized, 0.4, disp_Color, 0.4, 0)  # Blend images

    # Calculate frames per second (FPS)
    end = time.time()
    fps = 1 / (end - start)

    # Display calculated distances and FPS on the blended image
    cv2.putText(color_depth, f"Minimum: {round(min_distance, 1)}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(color_depth, f"Average: {round(avg_distance, 1)}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(color_depth, f"Maximum: {round(max_distance, 1)}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(color_depth, f"FPS: {round(fps)}", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the colorized depth map and blended image
    cv2.imshow("Depth", disp_Color)
    cv2.imshow("Color & Depth", color_depth)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

# Prepare parameters
chessboard_size = (9, 6)  # Inner corners (cols, rows)
square_size = 1.0         # Size of a square in your defined unit (e.g., cm or m)
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Lists to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints_left = []  # 2D points in left camera
imgpoints_right = []  # 2D points in right camera

# Process the images
for i in range(40):  # Loop through all 40 pairs
    img_left = cv2.imread(f"picl{i}.jpg")
    img_right = cv2.imread(f"picr{i}.jpg")

    if img_left is None or img_right is None:
        print(f"Failed to read image pair picl{i}.jpg and picr{i}.jpg")
        continue

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        objpoints.append(objp)

        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners_left)

        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        imgpoints_right.append(corners_right)

        print(f"Chessboard corners found in pair picl{i}.png and picr{i}.png")
    else:
        print(f"Chessboard not found in pair picl{i}.png and picr{i}.png")

# Calibrate each camera
ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

# Stereo calibration
ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, 
    mtx_left, dist_left, mtx_right, dist_right, 
    gray_left.shape[::-1], criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
)

# Rectify cameras
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1], R, T)

# Compute rectification maps
map_left_x, map_left_y = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, gray_left.shape[::-1], cv2.CV_16SC2)
map_right_x, map_right_y = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, gray_right.shape[::-1], cv2.CV_16SC2)

# Rectify and save images
# for i in range(40):
#     img_left = cv2.imread(f"picl{i}.jpg")
#     img_right = cv2.imread(f"picr{i}.jpg")

#     if img_left is None or img_right is None:
#         continue

#     rectified_left = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
#     rectified_right = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

#     cv2.imwrite(f"r/rectified_left_{i}.jpg", rectified_left)
#     cv2.imwrite(f"r/rectified_right_{i}.jpg", rectified_right)

print("Stereo calibration and rectification complete.")
# Save the rectification maps to an XML file
cv_file = cv2.FileStorage("stereo_rectify_maps.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", map_left_x)
cv_file.write("Left_Stereo_Map_y", map_left_y)
cv_file.write("Right_Stereo_Map_x", map_right_x)
cv_file.write("Right_Stereo_Map_y", map_right_y)
cv_file.release()
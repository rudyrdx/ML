import cv2
import numpy as np

# Define the intrinsic matrices
K1 = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]])  # Example matrix for Camera 1
K2 = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]])  # Example matrix for Camera 2

# Define the extrinsic parameters
# Rotation and translation matrices (relative positions of the two cameras)
R = np.eye(3)  # Assuming cameras are aligned for simplicity
t = np.array([[10], [0], [0]])  # Translation vector (10 cm)

# Compute projection matrices
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for Camera 1
P2 = K2 @ np.hstack((R, t))  # Projection matrix for Camera 2

# Click points on the images
def click_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["points"].append((x, y))
        print(f"Point selected: {(x, y)}")

def reconstruct_3D(pt1, pt2, P1, P2):
    """Reconstruct a 3D point from two image points."""
    A = [
        pt1[0] * P1[2] - P1[0],
        pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0],
        pt2[1] * P2[2] - P2[1],
    ]
    A = np.array(A).reshape(-1, 4)

    # Solve the system using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]  # The solution is the last row of Vt
    X /= X[-1]  # Dehomogenize (convert to 3D coordinates)

    return X[:3]

def main():
    # Load the stereo images
    img1 = cv2.imread("left.jpg")  # Replace with your left image path
    img2 = cv2.imread("right.jpg")  # Replace with your right image path

    points1 = {"points": []}
    points2 = {"points": []}

    cv2.imshow("Select points on Image 1", img1)
    cv2.setMouseCallback("Select points on Image 1", click_points, points1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("Select points on Image 2", img2)
    cv2.setMouseCallback("Select points on Image 2", click_points, points2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points1["points"]) != len(points2["points"]):
        print("Error: Number of points selected in both images must be the same.")
        return

    distances = []
    for pt1, pt2 in zip(points1["points"], points2["points"]):
        pt1_hom = np.array([pt1[0], pt1[1], 1])  # Homogeneous coordinates
        pt2_hom = np.array([pt2[0], pt2[1], 1])  # Homogeneous coordinates

        X = reconstruct_3D(pt1_hom, pt2_hom, P1, P2)
        distance = np.linalg.norm(X)
        distances.append(distance)
        print(f"3D Point: {X}, Distance: {distance:.2f} cm")

if __name__ == "__main__":
    main()

"""
(.env) PS E:\ML\ML\opencv\triangulation> python main.py
Point selected: (184, 468)
Point selected: (465, 458)
3D Point: [-4.840644    7.93593292 10.6744646 ], Distance: 14.15 cm
"""
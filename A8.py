import cv2
import matplotlib.pyplot as plt

# --- Read two images ---
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpeg', cv2.IMREAD_GRAYSCALE)

# --- Create SIFT detector ---
sift = cv2.SIFT_create()

# --- Detect keypoints and compute descriptors ---
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# --- Match features using BFMatcher ---
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# --- Sort matches by distance (best first) ---
matches = sorted(matches, key=lambda x: x.distance)

# --- Draw top 30 matches ---
result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)

# --- Display results ---
plt.figure(figsize=(12, 6))
plt.title("SIFT Feature Matching")
plt.imshow(result)
plt.axis("off")
plt.show()

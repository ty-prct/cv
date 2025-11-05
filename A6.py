import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read stereo images
img1 = cv2.imread('image1.jpg', 0)
img2 = cv2.imread('image2.jpeg', 0)

# Detect ORB keypoints and descriptors
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Select good matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]])

# Compute Fundamental Matrix using 8-point algorithm
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

print("Fundamental Matrix:\n", F)

# Draw epilines for visualization
def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1_color = cv2.line(img1_color, (x0,y0), (x1,y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(np.int32(pt1)), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(np.int32(pt2)), 5, color, -1)
    return img1_color, img2_color

# Select inliers and compute epilines
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
lines1 = lines1.reshape(-1,3)

img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img6)
plt.show()


#------------------------------------------------------------
import cv2
import numpy as np

# Read two stereo images
img1 = cv2.imread('image1.jpg', 0)
img2 = cv2.imread('image2.jpeg', 0)

# Detect ORB features and compute descriptors
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match features using Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched points
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:20]])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:20]])

# Compute Fundamental Matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

print("Fundamental Matrix:\n", F)

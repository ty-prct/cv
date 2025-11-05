import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('image1.jpg')

rows, cols = img.shape[:2]

# Define corresponding points (source → destination)
src_pts = np.float32([[100,100],[400,100],[100,400],[400,400]])
dst_pts = np.float32([[80,120],[420,80],[100,420],[420,420]])

# Compute Homography matrix
H, _ = cv2.findHomography(src_pts, dst_pts)

# Apply perspective transform
warped = cv2.warpPerspective(img, H, (cols, rows))

# Display results
plt.subplot(1,2,1), plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); plt.title('Original')
plt.subplot(1,2,2), plt.imshow(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB)); plt.title('Warped (Homography)')
plt.show()

print("Homography Matrix:\n", H)

import cv2, numpy as np, matplotlib.pyplot as plt

# Read image
img = cv2.cvtColor(cv2.imread('image1.jpg'), cv2.COLOR_BGR2RGB)

# Define 4 source and destination points
pts1 = np.float32([[500,500],[450,50],[50,300],[450,300]])
pts2 = np.float32([[200,300],[400,200],[400,300],[400,300]])

# Compute perspective transform and apply it
M = cv2.getPerspectiveTransform(pts1, pts2)
warp = cv2.warpPerspective(img, M, (400,300))

# Display
plt.subplot(1,2,1); plt.imshow(img); plt.title('Original')
plt.subplot(1,2,2); plt.imshow(warp); plt.title('Perspective Transformed')
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Read image ---
img = cv2.imread('image1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 1️⃣ Edge Detection ---
edges = cv2.Canny(gray, 20, 40)

# --- 2️⃣ Line Detection ---
img_lines = img.copy()
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=80, maxLineGap=10)
if lines is not None:
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# --- 3️⃣ Corner Detection ---
img_corners = img.copy()
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)
for c in corners:
    x, y = c.ravel()
    cv2.circle(img_corners, (x, y), 3, (0, 0, 255), -1)

# --- Display Results (3 separate images) ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Edge Detection")
plt.imshow(edges, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Line Detection")
plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title("Corner Detection")
plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()

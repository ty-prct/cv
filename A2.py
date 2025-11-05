import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('image1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols = img.shape[:2]

# Translation
M1 = np.float32([[1, 0, 100], [0, 1, 50]])
trans = cv2.warpAffine(img, M1, (cols, rows))

# Rotation
M2 = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rot = cv2.warpAffine(img, M2, (cols, rows))

# Scaling
scale = cv2.resize(img, None, fx=0.5, fy=0.5)

# Reflection (horizontal flip)
ref = cv2.flip(img, 1)

# Shearing
M3 = np.float32([[1, 0.5, 0], [0.5, 1, 0]])
shear = cv2.warpAffine(img, M3, (int(cols*1.5), int(rows*1.5)))

# Display results
titles = ['Original', 'Translated', 'Rotated', 'Scaled', 'Reflected', 'Sheared']
images = [img, trans, rot, scale, ref, shear]

plt.figure(figsize=(10,6))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.show()

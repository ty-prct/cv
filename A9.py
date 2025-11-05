import cv2, matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

img = cv2.imread('image1.jpg', 0)

# ORB (alternative to SURF)
orb = cv2.ORB_create()
kp, _ = orb.detectAndCompute(img, None)
img_orb = cv2.drawKeypoints(img, kp, None, (0,255,0), 4)

# HOG Feature Descriptor
_, hog_img = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2), visualize=True)
hog_img = exposure.rescale_intensity(hog_img, in_range=(0,255)).astype('uint8')

# Display
plt.subplot(1,2,1), plt.imshow(img_orb, cmap='gray'), plt.title("ORB Features"), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(hog_img, cmap='gray'), plt.title("HOG Descriptor"), plt.axis('off')
plt.show()

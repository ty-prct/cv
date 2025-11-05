import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('images/chessboard.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the number of inner corners per chessboard row and column
pattern_size = (7, 7)  # adjust if your board is different (e.g., 9x6)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

if ret:
    cv2.drawChessboardCorners(img, pattern_size, corners, ret)
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb)
    plt.title("Chessboard Corners")
    plt.axis('off')
    plt.show()
else:
    print("❌ No corners detected. Try changing pattern_size or using a clearer image.")


#-------------------------------------------------------------------

import cv2
from matplotlib import pyplot as plt

img = cv2.imread('images/chessboard.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pattern_size = (7, 7)

ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
if ret:
    cv2.drawChessboardCorners(img, pattern_size, corners, ret)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

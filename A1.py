import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image1.jpg')    

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

resize = cv2.resize(img, (300, 300))          

_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY) 

blur = cv2.GaussianBlur(img, (7, 7), 0)     

cv2.imwrite('processed.jpg', gray)                   

# Display all results
titles = ['Original', 'Resized', 'Grayscale', 'Threshold', 'Blurred']
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
          cv2.cvtColor(resize, cv2.COLOR_BGR2RGB), 
          gray, 
          thresh, 
          cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)]

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray' if i in [2,3] else None)
    plt.title(titles[i])
    plt.axis('off')
plt.show()

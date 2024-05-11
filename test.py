import cv2
import numpy as np

# Load maze image
maze_image = cv2.imread("C:\\Users\\trhoa\\Downloads\\a.jpg")

# Convert image to HSV color space
hsv_image = cv2.cvtColor(maze_image, cv2.COLOR_BGR2HSV)

# Define lower and upper thresholds for blue wall
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Threshold the HSV image to get only blue colors
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Define lower and upper thresholds for white path
lower_white = np.array([0, 0, 200])
upper_white = np.array([255, 30, 255])

# Threshold the HSV image to get only white colors
white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

# Combine the masks
combined_mask = cv2.bitwise_or(blue_mask, white_mask)

# Apply Canny edge detection on the combined mask
edges = cv2.Canny(blue_mask, 10, 20)

gray = cv2.cvtColor(maze_image, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
# Display the result
egd = cv2.Canny(bfilter, 50, 150)
cv2.imshow('Edge Detection', cv2.resize(blue_mask,(400,400)) )
cv2.imshow('combo Detection', cv2.resize(egd,(400,400)) )

cv2.waitKey(0)
cv2.destroyAllWindows()

# Reference: https://blog.csdn.net/zhy29563/article/details/119039163

# Create a ChArUco board image for camera calibration

import cv2

## parameters for Charuco board
square_x = 5                           # Number of squares in X direction
square_y = 7                           # Number of squares in Y direction
square_length = 40                    # Square side length (milimeters)
marker_length = 20                    # Marker side length (milimeters)
margins = square_length - marker_length  # Margins size 

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)    #dictionary

image_size_x = square_x * square_length + 2 * margins
image_size_y = square_y * square_length + 2 * margins
image_size = (image_size_x , image_size_y)
print(image_size)


## create board
board = cv2.aruco.CharucoBoard_create(square_x, square_y, square_length, marker_length, dictionary)

# show created board
board_image = cv2.aruco_CharucoBoard.draw(board, image_size, marginSize = 20, borderBits = 1)
#cv2.namedWindow('Charuco board image', cv2.WINDOW_NORMAL)
cv2.imshow('Charuco board image', board_image)    

# save image
cv2.imwrite('CharucoBoard.png', board_image)

key = cv2.waitKey(1)
# Press esc or 'q' to close the image window
if key & 0xFF == ord('q') or key == 27:
    cv2.destroyAllWindows()


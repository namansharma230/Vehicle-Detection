import cv2
import numpy as np
from time import sleep

# minimum width of rectangle
width_min = 80
# minimum height of rectangle
height_min = 80
# Error allowable between pixels
offset = 6
# to put the line on frame as when any vehicle crosses the line and then it gets counted
position_line = 550
delay = 60
# a list if anything detected it can get appended
detec = []
count = 0

# just to find the mid-point of rectangle on the vehicle for red dot
def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# create a video capture object and help to display the video
cap = cv2.VideoCapture(r"C:\Users\Naman Sharma\Desktop\Folders\ML Projects\Vehicle Detection\video.mp4")
# this is one of the algorithms in cv2 and it is known as a subtractor. It is used to subtract the background of our object in the video
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    # return the specific frame and read the video
    ret, frame1 = cap.read()
    tempo = float(1 / delay)
    sleep(tempo)
    # used to convert the color of a specific frame here it is frame1
    # cvtColor means convert color
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Gaussian blur is the result of blurring an image. It is commonly used when reducing the size of an image.
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    # now we create a variable as an image subtractor and we applying the algo on it
    img_sub = subtractor.apply(blur)
    # Dilates an image by using a specific structuring element. The function dilates the source image using the specified structuring element that
    # determines the shape of a pixel.
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    # Returns a structuring element of the specified size and shape for morphological operations(Morphology is a broad set of image
    # processing operations that process images based on shapes. In a morphological operation, each pixel in the image is adjusted based on the
    # value of other pixels in its neighborhood). The function constructs and returns the
    # structuring element that can be further passed to construct an arbitrary binary mask yourself and use it as the structuring element.
    # MORPH_ELLIPSE an elliptic structuring element, that is, a filled ellipse inscribed into the rectangle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # this is simply giving the black and white phase of our video as in such as a background
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    # just to count the number of objects as in our case here it thins the image from getting to subtractor
    contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # here up to it works as it does all background in black color and all vehicles in white ones.
    # just giving the specification of line like position or color or etc., etc.
    cv2.line(frame1, (25, position_line), (1200, position_line), (255, 127, 0), 3)
    # for putting rectangle on vehicles
    for (i, c) in enumerate(contorno):
        # here x, y represents the x and y plane
        # rectangle contains length and breadth here I take w for width and h for height or length
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= width_min) and (h >= height_min)
        if not validar_contorno:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame1, "VEHICLE COUNT : " + str(count), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 2)
        centro = find_center(x, y, w, h)
        detec.append(centro)
        # for creating a circle
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

    # for printing the output and count the number of vehicles on the window
    for (x, y) in detec:
        if y < (position_line + offset) and y > (position_line - offset):
            count += 1
            # when any vehicle passes through the line the color changes.
            cv2.line(frame1, (25, position_line), (1200, position_line), (0, 127, 255), 3)
            detec.remove((x, y))
            print("car is detected : " + str(count))
            cv2.putText(frame1, "VEHICLE COUNT : " + str(count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("Video Original", frame1)
    cv2.imshow("Detectar", dilatada)
    # to stop the window
    if cv2.waitKey(1) == 27:
        break

# after completing all tasks just to release all windows
cv2.destroyAllWindows()
cap.release()

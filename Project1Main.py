import numpy as np
import cv2
from __main__ import *
import imutils
import math
from homography import homo

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

x1 = 5
x2 = 150
x3 = 15
x4 = 5
y1 = 5
y2 = 5
y3 = 150
y4 = 150
xp1 = 100
xp2 = 200
xp3 = 220
xp4 = 100
yp1 = 100
yp2 = 80
yp3 = 80
yp4 = 200
A = np.array([
    [-x1, -y1, -1, 0, 0, 0, x1 * xp1, y1 * xp1, xp1],
    [0, 0, 0, -x1, -y1, -1, x1 * yp1, y1 * yp1, yp1],
    [-x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2],
    [0, 0, 0, -x2, -y2, -1, x2 * yp2, y2 * yp2, yp2],
    [-x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3],
    [0, 0, 0, -x3, -y3, -1, x3 * yp3, y3 * yp3, yp3],
    [-x4, -y4, -1, 0, 0, 0, x4 * xp4, y4 * xp4, xp4],
    [0, 0, 0, -x4, -y4, -1, x4 * yp4, y4 * yp4, yp4]
])
prgRun=True

im_width=320
im_height=240

print('Initializations complete')

def main(prgRun):

    datachoice=int(input('\nWhich video data would you like to use? \nPlease enter 1, 2, or 3: '))
    # section = input('\nIdentify QR code? Impose Image? Impose Cube? \nPlease enter 1, 2, or 3: ')

    if datachoice == 1:
        video = cv2.VideoCapture('data_1.mp4')
    if datachoice == 2:
        video = cv2.VideoCapture('data_2.mp4')
    if datachoice == 3:
        video = cv2.VideoCapture('data_3.mp4')

    if datachoice != 1 and datachoice !=2 and datachoice !=3:
        print('End Program')
        # prgRun=False
        # return prgRun


    # Read until video is completed
    while (video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == True:
            ###########################
            frame = imutils.resize(frame, width=320)
            ogframe=frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame, (10, 10), 0)
            # frame = cv2.bilateralFilter(frame, 9, 75, 75)
            # (thresh, frame) = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
            # frame, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            (thresh, frame) = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            frame = cv2.drawContours(ogframe, contours, 1, (0, 255, 0), 3)

            ###TEST




            ###########################
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    H=homo(A)



    # return prgRun


print('Function Initializations complete')


if __name__ == '__main__':
    print('Start')
    while prgRun==True:
        prgRun=main(prgRun)



    print('Goodbye!')
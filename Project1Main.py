import numpy as np
import cv2
from __main__ import *
import imutils
import math
from homography import homo
from Amatrix import Amatrix
from dewarp import dewarp

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)


prgRun = True

im_width = 320
im_height = 240








def main(prgRun):
    print('Initializations complete')
    framecount=1
    datachoice = int(input('\nWhich video data would you like to use? \nPlease enter 1, 2, or 3: '))
    # section = input('\nIdentify QR code? Impose Image? Impose Cube? \nPlease enter 1, 2, or 3: ')

    if datachoice == 1:
        video = cv2.VideoCapture('data_1.mp4')
    if datachoice == 2:
        video = cv2.VideoCapture('data_2.mp4')
    if datachoice == 3:
        video = cv2.VideoCapture('data_3.mp4')

    if datachoice != 1 and datachoice != 2 and datachoice != 3:
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
            ogframe = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            # frame = cv2.bilateralFilter(frame, 9, 75, 75)
            (thresh, frame) = cv2.threshold(frame, 220, 255, cv2.THRESH_BINARY)
            # frame = np.float32(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame = cv2.GaussianBlur(frame, (1, 1), 0)
            # frame = cv2.Canny(frame, 220, 255)
            # cv2.imshow('Thresh', frame)
            # print(frame.shape)

            cnts, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if framecount>1:
                cntsmissing=(cv2.contourArea(oldcnts[1])-cv2.contourArea(cnts[1]))**2
                if cntsmissing >3200000:
                    cnts=oldcnts
            oldcnts=cnts

            epsilon = 0.1 * cv2.arcLength(cnts[1], True)
            corners=cv2.approxPolyDP(cnts[1], epsilon, True)

            for i in range(len(corners)):
                x, y = corner_points[i][0], corner_points[i][1]
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            # print('##########')
            # print(corners)
            # print(len(corners))
            # input()

            frame = cv2.drawContours(ogframe, cnts, -1, (0, 255, 0), 2)

            # frame = np.float32(frame)
            # dst = cv2.cornerHarris(frame, 2, 3, 0.04)
            # dst = cv2.dilate(dst, None)
            # ogframe[dst > 0.01 * dst.max()] = [0,255]

            ###########################
            # Display the resulting frame
            cv2.imshow('Frame', frame)


            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            try:
            # if True:


                cnts = imutils.grab_contours(cnts)
                cntsx = cnts[:, :, 0]
                cntsy = cnts[:, :, 1]
                maxX = np.argmax(cntsx)
                minX = np.argmin(cntsx)
                maxY = np.argmax(cntsy)
                minY = np.argmin(cntsy)


                TL=np.array([cntsx[minX],cntsy[maxY]])
                TR=np.array([cntsx[maxX],cntsy[maxY]])
                LL=np.array([cntsx[minX],cntsy[minY]])
                LR=np.array([cntsx[maxX],cntsy[minY]])

                # print("#####################")
                # print(cnts)
                # print("#####################")
                # print(TL)
                # print(TR)
                # print(LL)
                # print(LR)

                A = Amatrix(TL, TR, LL, LR)
                # print(A)

                H = homo(A)
                # print(H)

                pixX=frame.shape[1]
                pixY=frame.shape[0]

                DWF = np.zeros([180, 320, 3])
                # print(DWF)

                DWF=dewarp(DWF,frame,H,pixX,pixY)


                cv2.imshow('DWF',DWF)


            except:
                None
                # print('###################')
                # print(cnts)
            framecount=framecount+1

        # Break the loop
        else:
            break



    # return prgRun


print('Function Initializations complete')

if __name__ == '__main__':
    print('Start')
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()

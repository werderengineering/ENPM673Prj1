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

    framecount = 1

    dcoeff=np.array([0.06981980863464919, -0.2293512169497289, -0.00525889574956216, -0.001081794502850245])

    ICV = np.array([
        [1465.1743559463634, 0, 501.5010208509487],
        [0, 1478.4325536850997, 905.945757447447],
        [0, 0, 1]
    ])

    resolution=np.array([1080, 1920])

    blobParams = cv2.SimpleBlobDetector_Params()
    blobParams.filterByColor = True
    blobParams.blobColor = 255

    blobParams.filterByArea = True
    blobParams.maxArea =60000

    blobVer = (cv2.__version__).split('.')
    if int(blobVer[0]) < 3:
        blob = cv2.SimpleBlobDetector(blobParams)
    else:
        blob = cv2.SimpleBlobDetector_create(blobParams)

    blob = cv2.SimpleBlobDetector_create(blobParams)

    print('Initializations complete')

    datachoice=1
    section=3
    # datachoice = int(input('\nWhich video data would you like to use? \nPlease enter 1, 2, or 3: '))
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

    lena = cv2.imread("Lena.png")





    # Read until video is completed
    while (video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == True:

            ###########################
            frame = imutils.resize(frame, width=320)
            ogframe = frame
            clnframe=frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mask = cv2.inRange(frame, 180, 255)
            frame = cv2.bitwise_or(frame, frame, mask=mask)


            blobOrigin = blob.detect(mask)

            for i in range(len(blobOrigin)):
                px = int(blobOrigin[i].pt[0])
                py = int(blobOrigin[i].pt[1])
                blobRadius = int(blobOrigin[i].size)


                # frame=frame[py-blobRadius:py+blobRadius,px-blobRadius:px+blobRadius]
                # ogframe = ogframe[py - blobRadius:py + blobRadius, px - blobRadius:px + blobRadius]
                #
                # cv2.imshow('modified frame', frame)


                try:
                    center = cv2.drawKeypoints(frame, blobOrigin, np.array([]), (0, 255, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    # # print(center)
                    # cv2.imshow('keypoints', center)


                    cnts, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:]
                # print(cnts)
                except:
                    None

                if framecount>1:
                    cntsmissing=(cv2.contourArea(oldcnts[1])-cv2.contourArea(cnts[1]))**2
                    if cntsmissing >3200000:
                        cnts=oldcnts
                oldcnts=cnts
                #
                # epsilon = 0.1 * cv2.arcLength(cnts[1], True)
                # corners=cv2.approxPolyDP(cnts[1], epsilon, True)

                frame = cv2.drawContours(ogframe, cnts, -1, (0, 255, 0), 2)


                ###########################
                # Display the resulting frame
                # cv2.imshow('Frame', frame)


                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                # try:
                if True:

                    cnt=cnts[0]

                    x, y, w, h = cv2.boundingRect(cnt)
                    # cv2.rectangle(ogframe, (x, y), (x + w, y + h), (0, 255, 255), 2)

                    rect = cv2.minAreaRect(cnt)

                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    try:

                        cv2.drawContours(ogframe, [box], 0, (0, 0, 255), 2)

                        cv2.imshow('box',ogframe)
                    except:
                        None


                    TL=np.array([box[0][0],box[0][1]])
                    LL=np.array([box[1][0],box[1][1]])
                    LR=np.array([box[2][0],box[2][1]])
                    TR = np.array([box[3][0], box[3][1]])

                    ############################WARNING##########################

                    C = np.matmul(np.ones([4, 2]), ([[px, 0], [0, py]]))

                    R = np.ones([4, 2]) * blobRadius

                    boxnew=C+R-box

                    # print(boxnew)

                    boxnew=box
                    ############################WARNING##########################


                    if section==1:
                        OutputFrame = np.zeros([180, 320, 3])
                        OutputFrame = np.float64(DWF)
                        H = homo(boxnew,9)
                        # DWF = dewarp(OutputFrame, AppliedFrame, H, box)

                    if section==2:
                        OutputFrame=clnframe
                        AppliedFrame=lena
                        # print(lena.shape)
                        H = homo(boxnew,512)
                        # DWF = dewarp(OutputFrame, AppliedFrame, H, box)

                    # DWF = dewarp(OutputFrame, AppliedFrame, H, box)

                    if section==3:

                        ################BOX BUILDING 101####################

                        boxtop=np.array([box[:,0]+10,box[:, 1] - 30]).T

                        ################BOX Homography?#####################

                        H = homo(box, boxtop)


                        # P = np.matmul(K, np.matrix(R))
                        # P = P / P[2, 3]



                        ############Build the sides of the box##############
                        boxL=np.array([
                            box[0],
                            boxtop[0],
                            boxtop[1],
                            box[1],
                        ])

                        boxR = np.array([
                            box[2],
                            boxtop[2],
                            boxtop[3],
                            box[3],
                        ])



                        cv2.drawContours(ogframe, [boxtop], 0, (255, 0, 255), 2)
                        cv2.drawContours(ogframe, [boxL], 0, (127, 0, 0), 2)
                        cv2.drawContours(ogframe, [boxR], 0, (255, 0, 0), 2)






                        cv2.imshow('box', ogframe)



                    #
                    #
                    try:
                        None
                        # cv2.imshow('DWF',DWF)

                    except:
                        None


                # except:
                #     None
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

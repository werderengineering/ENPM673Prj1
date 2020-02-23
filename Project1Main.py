import numpy as np
import cv2
from __main__ import *
import imutils
import math
from homography import homo

from dewarp import dewarp
from findAngle import findAngleAndID
from scipy import ndimage


print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

prgRun = True

im_width = 320
im_height = 240


def main(prgRun):
    QR1framecount = 1
    QR2framecount = 1
    QR3framecount = 1
    cnts1=0
    cnts2=0
    cnts3=0

    dcoeff = np.array([0.06981980863464919, -0.2293512169497289, -0.00525889574956216, -0.001081794502850245])

    ICV = np.array([
        [1465.1743559463634, 0, 501.5010208509487],
        [0, 1478.4325536850997, 905.945757447447],
        [0, 0, 1]
    ])

    resolution = np.array([1080, 1920])

    blobParams = cv2.SimpleBlobDetector_Params()
    blobParams.filterByColor = True
    blobParams.blobColor = 255

    blobParams.filterByCircularity = False

    blobParams.filterByArea = True
    blobParams.maxArea = 70000

    blobVer = (cv2.__version__).split('.')
    if int(blobVer[0]) < 3:
        blob = cv2.SimpleBlobDetector(blobParams)
    else:
        blob = cv2.SimpleBlobDetector_create(blobParams)

    blob = cv2.SimpleBlobDetector_create(blobParams)

    print('Initializations complete')

    datachoice = 2
    section = 3
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
            clnframe = frame
            resetframe=frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayframe = frame

            mask = cv2.inRange(frame, 170, 255)
            frame = cv2.bitwise_or(frame, frame, mask=mask)

            blobOrigin = blob.detect(mask)


            # print('\nFresh frame')

            for qR in range(len(blobOrigin)):
                frame=resetframe

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grayframe = frame

                mask = cv2.inRange(frame, 180, 255)
                frame = cv2.bitwise_or(frame, frame, mask=mask)

                # cv2.imshow('frame',frame)
                # cv2.imshow('ogframe', ogframe)


                px = int(blobOrigin[qR].pt[0])
                py = int(blobOrigin[qR].pt[1])
                blobRadius = int(blobOrigin[qR].size*.8)

                # print(blobOrigin)
                #
                # print('\nnumber of blobs',len(blobOrigin))
                # print(px)
                # print(py)

                ogframe=resetframe


                frame = frame[py - blobRadius:py + blobRadius, px - blobRadius:px + blobRadius]
                ogframe = ogframe[py - blobRadius:py + blobRadius, px - blobRadius:px + blobRadius]


                # cv2.imshow('ogframe',ogframe)

                try:
                    clnframe = cv2.drawKeypoints(clnframe, blobOrigin, np.array([]), (0, 255, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    # print(center)
                    # cv2.imshow('keypoints', center)

                    cnts, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    frame = cv2.drawContours(ogframe, cnts, -1, (0, 255, 0), 2)

                    # clnframe = cv2.drawContours(clnframe, cnts, 1, (0, 255, 0), 2)
                # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:]
                #     print(cnts)
                except:
                    'No Contours found'

                try:
                    if qR==0:
                        if QR1framecount > 1:
                            cntsmissing = (cv2.contourArea(oldcnts1[1]) - cv2.contourArea(cnts[1])) ** 2
                            if cntsmissing > 3200000:
                                cnts = oldcnts1
                        oldcnts1 = cnts
                        QR1framecount = QR1framecount + 1


                    elif qR==1:
                        if QR2framecount > 1:
                            cntsmissing = (cv2.contourArea(oldcnts2[1]) - cv2.contourArea(cnts[1])) ** 2
                            if cntsmissing > 3200000:
                                cnts = oldcnts2
                        oldcnts2 = cnts
                        QR2framecount = QR2framecount + 1


                    elif qR==2:
                        if QR3framecount > 1:
                            cntsmissing = (cv2.contourArea(oldcnts3[1]) - cv2.contourArea(cnts[1])) ** 2
                            if cntsmissing > 3200000:
                                cnts = oldcnts3
                        oldcnts3 = cnts
                        QR3framecount = QR3framecount + 1
                except:
                    print('Contour grab Fail, using old contours')

                    if qR == 0:
                        cnts = oldcnts1
                    elif qR == 1:
                        cnts = oldcnts2
                    elif qR == 2:
                        cnts = oldcnts3




                #
                # epsilon = 0.1 * cv2.arcLength(cnts, True)
                # corners=cv2.approxPolyDP(cnts, epsilon, True)



                ###########################
                # Display the resulting frame


                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                # try:
                if True:


                    cnt = cnts[0]

                    x, y, w, h = cv2.boundingRect(cnt)
                    # cv2.rectangle(clnframe, (x, y), (x + w, y + h), (0, 255, 255), 2)

                    rect = cv2.minAreaRect(cnt)

                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    # try:
                    #
                    # cv2.drawContours(frame, [box], 0, (255, 255, 0), 2)
                    #
                    #     cv2.imshow('box',ogframe)
                    # except:
                    #     None
                    # cv2.imshow('QR', frame)

                    TL = np.array([box[0][0], box[0][1]])
                    LL = np.array([box[1][0], box[1][1]])
                    LR = np.array([box[2][0], box[2][1]])
                    TR = np.array([box[3][0], box[3][1]])


                    C = np.matmul(np.ones(box.shape), ([[px, 0], [0, py]]))

                    R = np.ones(box.shape) * blobRadius

                    boxnew = C +box-R
                    #
                    # # print(boxnew)

                    # boxnew = box

                    ############ROTATIONofIMAGEMATRIX#############

                    cntblck = cnts[1]

                    x, y, w, h = cv2.boundingRect(cntblck)
                    grect = cv2.minAreaRect(cntblck)

                    gbox = cv2.boxPoints(grect)
                    gbox = np.int0(gbox)
                    # print(gbox[0])
                    # grayframe=frame


                    # grayframe=grayframe[y:y + int(h*1.1), x:x+int(w*1.1)]
                    # mask = cv2.inRange(grayframe, 180, 255)
                    # grayframe = cv2.bitwise_or(grayframe, grayframe, mask=mask)

                    Center = np.matmul(np.ones([4, 2]), ([[px, 0], [0, py]]))
                    Radius = np.ones([4, 2]) * blobRadius
                    gbox = Center + gbox - Radius

                    # grayframe=grayframe[y:y + h, x:x+w]

                    # print(grayframe.shape)

                    OutputFrame = np.zeros([300, 300, 3])
                    # OutputFrame = np.float64(OutputFrame)

                    ThisOn = grayframe
                    That = OutputFrame
                    inhere = gbox
                    H = homo(gbox,0)
                    grayframe = dewarp(OutputFrame, grayframe, H, inhere)



                    try:
                        # cv2.imshow('grayframe',grayframe)
                        angle, ID = findAngleAndID(grayframe, 0, 0)

                        # print(angle)

                        if angle ==180:
                            angle = 0
                            # print('TR')

                        elif angle==0:
                            angle=180
                            # print('BL')

                        elif abs(angle)==90:
                            angle=-angle


                        # print("Angle: ", angle)

                        lenaR = ndimage.rotate(lena, angle)

                        if qR==0:
                            pangle1=angle
                        elif qR == 1:
                            pangle2 = angle

                        elif qR == 2:
                            pangle3 = angle

                    except:
                        if qR==0:
                            lenaR = ndimage.rotate(lena, pangle1)
                        elif qR == 1:
                            lenaR = ndimage.rotate(lena, pangle2)

                        elif qR == 2:
                            lenaR = ndimage.rotate(lena, pangle3)






                    # cv2.imshow('LenaR',lenaR)

                    ############ROTATIONofIMAGEMATRIX#############

                    if section == 1:
                        OutputFrame = np.zeros([180, 320, 3])
                        OutputFrame = np.float64(OutputFrame)
                        H = homo(boxnew, 0)
                        # DWF = dewarp(OutputFrame, AppliedFrame, H, box)

                    if section == 2:
                        ThisOn = lenaR
                        That = clnframe
                        inhere=boxnew
                        # print(lena.shape)
                        H = homo(boxnew, 512)
                        clnframe = dewarp(That, ThisOn, H, inhere)



                    if section == 3:
                        ################BOX BUILDING 101####################
                        box=boxnew.astype(int)

                        boxtop = np.array([box[:, 0] + 10, box[:, 1] - 30]).T



                        ################BOX Homography?#####################

                        H = homo(box, boxtop)

                        # boxtop=cube_top(perjectionMatrix, corners_inFrame, corners_flatView)

                        ############Build the sides of the box##############
                        boxL = np.array([
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

                        # cv2.imshow('frame',frame)

                        # print(box)
                        # print(boxnew.astype(int))

                        cv2.drawContours(clnframe, [box], 0, (255, 127, 255), 2)
                        cv2.drawContours(clnframe, [boxtop], 0, (255, 0, 255), 2)
                        cv2.drawContours(clnframe, [boxL], 0, (127, 0, 0), 2)
                        cv2.drawContours(clnframe, [boxR], 0, (255, 0, 0), 2)

                        # print('QR: ', qR)
                        # cv2.imshow('see this', clnframe)
                        # input("Press enter")
                # except:
                #     None



            try:
                # None
                if section == 1:
                    frame = cv2.drawContours(ogframe, cnts, -1, (0, 255, 0), 2)

                    frame = cv2.drawContours(ogframe, cnts, 2, (255, 0, 255), 2)
                    cv2.imshow('Tracking', frame)

                elif section == 2:
                    # input('Generate final frame:')
                    cv2.imshow('DWF', clnframe)
                    # cv2.imshow('Finalframe',finalframe)
                else:
                    try:
                        cv2.imshow('box', clnframe)
                    except:
                        None
            except:
                None

                # print('###################')
                # print(cnts)


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

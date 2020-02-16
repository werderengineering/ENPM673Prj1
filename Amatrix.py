from __main__ import *

def Amatrix(TL,TR,LL,LR):


    xp1 = 0
    xp2 = 320
    xp3 = 320
    xp4 = 0

    yp1 = 0
    yp2 = 0
    yp3 = 180
    yp4 = 180

    x1 = TL[0]
    x2 = TR[0]
    x3 = LR[0]
    x4 = LL[0]

    y1 = TL[1]
    y2 = TR[1]
    y3 = LR[1]
    y4 = LL[1]

    # x1 = 0
    # x2 = 320
    # x3 = 320
    # x4 = 0
    #
    # y1 = 0
    # y2 = 0
    # y3 = 180
    # y4 = 180
    #
    # xp1 = TL[0][0]
    # xp2 = TR[0][0]
    # xp3 = LR[0][0]
    # xp4 = LL[0][0]
    #
    # yp1 = TL[1][0]
    # yp2 = TR[1][0]
    # yp3 = LR[1][0]
    # yp4 = LL[1][0]

    A = -np.array([
        [-x1, -y1, -1, 0, 0, 0, x1 * xp1, y1 * xp1, xp1],
        [0, 0, 0, -x1, -y1, -1, x1 * yp1, y1 * yp1, yp1],
        [-x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2],
        [0, 0, 0, -x2, -y2, -1, x2 * yp2, y2 * yp2, yp2],
        [-x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3],
        [0, 0, 0, -x3, -y3, -1, x3 * yp3, y3 * yp3, yp3],
        [-x4, -y4, -1, 0, 0, 0, x4 * xp4, y4 * xp4, xp4],
        [0, 0, 0, -x4, -y4, -1, x4 * yp4, y4 * yp4, yp4]
    ])
    return A
from __main__ import *


def square(side):
    """return four corner of square, the point vector is in row"""
    return np.array([[0, 0], [0, side], [side, side], [side,0]])



def homo(p1, p2):

    try:

        if p2==0:
            p2 = square(100)
        if p2==512:
            p2 = square(512)

    except:
        p2=p2


    # h, status = cv2.findHomography(p1, p2)


    """Input two stack of points, each stack has four points vector in row,
    return homograph between two stack"""
    # check input points
    assert p1.shape == (4, 2), "P1 has size: " + str(p1.shape)
    assert p2.shape == (4, 2), "P2 has size: " + str(p2.shape)
    # assign values
    [x1, y1] = p1[0]
    [x2, y2] = p1[1]
    [x3, y3] = p1[2]
    [x4, y4] = p1[3]
    [xp1, yp1] = p2[0]
    [xp2, yp2] = p2[1]
    [xp3, yp3] = p2[2]
    [xp4, yp4] = p2[3]


    A = np.array([
        [-x1, -y1, -1, 0, 0, 0, x1 * xp1, y1 * xp1, xp1],
        [0, 0, 0, -x1, -y1, -1, x1 * yp1, y1 * yp1, yp1],
        [-x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2],
        [0, 0, 0, -x2, -y2, -1, x2 * yp2, y2 * yp2, yp2],
        [-x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3],
        [0, 0, 0, -x3, -y3, -1, x3 * yp3, y3 * yp3, yp3],
        [-x4, -y4, -1, 0, 0, 0, x4 * xp4, y4 * xp4, xp4],
        [0, 0, 0, -x4, -y4, -1, x4 * yp4, y4 * yp4, yp4]
    ], dtype=np.float64)

    u, s, v = np.linalg.svd(A)
    X = v[:][8]   # right singular vector
    X = X / v[8][8]
    H = np.reshape(X, (3, 3))   # make H a matrix
    # H = np.linalg.pinv(H)
    H = H / H[2][2]     # normalize
    # print('\nh', h)
    # print('\nH',H)

    # H=h
    return H
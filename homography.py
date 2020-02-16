from __main__ import *


def square(side):
    """return four corner of square, the point vector is in row"""
    return np.array([[0, 0], [0, side], [side, side], [side,0]])



def homo(p1, p2):

    if p2==9:
        p2 = square(100)
    else:
        p2 = square(512)

    h, status = cv2.findHomography(p1, p2)


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
    [xp3, yp3] = p2[3]
    [xp4, yp4] = p2[3]

    # [x1, y1] = p2[0]
    # [x2, y2] = p2[1]
    # [x3, y3] = p2[2]
    # [x4, y4] = p2[3]
    # [xp1, yp1] = p1[0]
    # [xp2, yp2] = p1[1]
    # [xp3, yp3] = p1[3]
    # [xp4, yp4] = p1[3]
    """
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
    """
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
    # assert np.linalg.norm(X, axis=0).astype(int) == 1, "X is " + str(X)
    X = X / v[8][8]
    H = np.reshape(X, (3, 3))   # make H a matrix
    # H = np.linalg.pinv(H)
    H = H / H[2][2]     # normalize
    # print('\nh', h)
    # print('\nH',H)

    H=h
    return H
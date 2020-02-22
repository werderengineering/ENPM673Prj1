"""IMPORT"""
import numpy as np
import cv2


def square(side=200):
    """return four corner of square, the point vector is in row"""
    return np.array([[0, 0], [side, 0], [side, side], [0, side]])


def Estimated_Homography(p1, p2=square()):
    """Input two stack of points, each stack has four points vector in row,
    return the homography matrix from p1 to p2"""
    # check input points
    assert p1.shape == (4, 2), "P1 has size: " + str(p1.shape)
    assert p2.shape == (4, 2), "P1 has size: " + str(p2.shape)
    # assign values
    [x1, y1] = p1[0]
    [x2, y2] = p1[1]
    [x3, y3] = p1[2]
    [x4, y4] = p1[3]
    [xp1, yp1] = p2[0]
    [xp2, yp2] = p2[1]
    [xp3, yp3] = p2[2]
    [xp4, yp4] = p2[3]
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
    A = -np.array([
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
    X = v[:][8] / v[8][8]   # right singular vector
    H = np.reshape(X, (3, 3))   # make H a matrix
    H = H / H[2][2]     # normalize
    return H


def TwoDtohomogeneous(p_2D):
    """change the coordinates from 2D to 2D homogeneous"""
    assert p_2D.shape[1] == 2, "Input points stack has # of shape " + str(p_2D.shape)
    p_2D_homogeneous = np.ones([p_2D.shape[0], 3])
    p_2D_homogeneous[:, 0:2] = p_2D[:, :]
    p_2D_homogeneous = p_2D_homogeneous.astype(int)
    assert p_2D_homogeneous.shape[1] == 3
    return p_2D_homogeneous


def homogenousToTwoD(p_2D_homogeneous):
    """change the coordinates from 2D homogeneous to 2D"""
    assert p_2D_homogeneous.shape[1] == 3, "Input points stack has # of shape " + str(p_2D_homogeneous.shape)
    z = np.asarray(p_2D_homogeneous[:, 2]).reshape(p_2D_homogeneous.shape[0], 1)
    p_2D_homogeneous = p_2D_homogeneous/z  # make sure it is homogeneous
    p_2D = p_2D_homogeneous[:, 0:2]
    p_2D = p_2D.astype(int)
    assert p_2D.shape[1] == 2
    return p_2D


def perspectiveTransfer_image(img_perspect1, side_AR_Tag_estimation, homo):
    """switch the image perspective from 1 to 2"""
    img_perspect1 = np.asarray(img_perspect1)
    img_perspect1 = img_perspect1.astype(np.uint8)
    (range_row, range_col) = img_perspect1.shape
    img_perspect2 = cv2.warpPerspective(img_perspect1, homo, (side_AR_Tag_estimation, side_AR_Tag_estimation))
    img_perspect2 = img_perspect2.astype(np.uint8)
    """
    img_perspect2 = np.zeros(img_perspect1.shape)
    for row in range(0, range_row):  # column of img_perspect1
        for col in range(0, range_col):  # row of img_perspect1
            coord_pixel_orginal_homo = TwoDtohomogeneous(np.array([[row, col]]))    # make frame pixel from 2D to homogeneous
            pixel_coord_transferomed_homo = np.transpose(np.dot(homo, coord_pixel_orginal_homo.transpose()))    # cast frame pixel from original view to flatten view pixel
            pixel_coord_transferomed = homogenousToTwoD(pixel_coord_transferomed_homo)      # go back from homogeneous to 2D
            pixel_coord_transferomed = pixel_coord_transferomed.reshape(2)      # regulate the format
            if (pixel_coord_transferomed[0] < range_row) and (pixel_coord_transferomed[1] < range_col) and (pixel_coord_transferomed[0] > -1) and (pixel_coord_transferomed[1] > -1):   # if the flatten view in bound
                img_perspect2[pixel_coord_transferomed[0], pixel_coord_transferomed[1]] = img_perspect1[row, col]
    """
    return img_perspect2


def perspectiveTransfer_coord(p_2D, homo):
    """change the coordinates from 2D to 2D homogeneous
    the input points should be like:
    np.array([[x0, y0],
    [x1, y1],
    ...
    ])
    output points follow the same format
    """
    assert p_2D.shape[1] == 2, "Input points stack has shape: " + str(p_2D.shape)
    p_2D_homo = TwoDtohomogeneous(p_2D)
    assert p_2D_homo.shape[1] == 3, "After homogeneous, input points stack has shape: " + str(p_2D_homo.shape)
    p_2D_homo_transfromed = np.transpose(np.dot(homo, p_2D_homo.transpose()))   # make first step of homograph transformation
    p_2D_transfromed = homogenousToTwoD(p_2D_homo_transfromed)
    """
    z = np.asarray(p_2D_homo_transfromed[:, 2]).reshape(p_2D_homo_transfromed.shape[0], 1)  # third element after first homograph transformation
    p_2D_homo_transfromed = p_2D_homo_transfromed/z    # divide third element, third element will be 1
    p_2D_transfromed = p_2D_homo_transfromed[:, 0:2]
    p_2D_transfromed = p_2D_transfromed.astype(int)     # force pixel coordinates to be integer
    """

    assert p_2D_transfromed.shape[1] == 2, "Output points stack has # of shape " + str(p_2D_transfromed.shape)
    return p_2D_transfromed

import cv2
import numpy as np
import math

flag = True
side_AR_Tag_estimation = 150
Number_grid = 8

def loadImages_refMarker():
    """input reference marker images"""
    global AR_tag_reference
    AR_tag_reference = cv2.imread("./data/ref_marker.png")


def clockwiseangle_and_distance(point):
    # Vector between point and the origin: v = p - o
    # origin = [2, 3]
    origin = [0, 0]
    refvec = [0, 1]
    vector = [point[0] - origin[0], point[1] - origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0] / lenvector, vector[1] / lenvector]
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2 * math.pi + angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


def distanceAndAngle(p1, p2):
    """compute the Euclidean distance between points p1 and points p2, input point in row"""
    assert p1.shape[1] == 2 or p1.shape[1] == 3
    assert p2.shape[1] == 2 or p2.shape[1] == 3
    r = np.linalg.norm(p1, axis=1)  # distance between origin and p1
    d = np.linalg.norm(p1 - p2, axis=1)  # distance between p1 and p2
    theta = d / r  # angle of rotation need to merge p1 and p2
    assert d.shape[0] == p1.shape[0]
    return d, theta


def findCorners(image_gray, flag=False):
    """find the corners of a AR tag, the input AR tag may only the out layer and inner layer"""
    if flag:
        print("findCorners: input image size: " + str(image_gray.shape))
    """return a stack of corner by harris corner detection from input image"""
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    # image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_gray = np.float32(image_gray)

    # ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, chessboard_flags)
    img_corners = cv2.cornerHarris(image_gray, 2, 3, 0.04)  # get a image only shows corners
    if flag:  # is harris corner detection
        img = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        img[img_corners > 0.01 * img_corners.max()] = [0, 0, 255]
        cv2.imshow('harris corner detection', img)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    img_corners = cv2.dilate(img_corners, None)
    ret, img_corners = cv2.threshold(img_corners, 0.01 * img_corners.max(), 255, 0)
    dst = np.uint8(img_corners)

    # find centroids that gives a initial corner location
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image_gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    index_totalCentroid = 0  # the index indicates where the total centroid is in the corner tuple
    corners = corners[index_totalCentroid + 1:, :]

    print("Before sort: " + str(corners))
    """sort the corner from top-right to bottom-right to bottom-left to top-left"""
    corners, center_original = shiftToASpot(corners, np.array([0, 0]))
    corners = sorted(corners, key=clockwiseangle_and_distance)
    corners, shouldBeZeros = shiftToASpot(np.asarray(corners), center_original)
    print("After sort: " + str(corners))

    if flag:
        corners_int = corners.astype(int)
        # Now draw them
        # res = np.hstack((centroids, corners))
        # res = np.int0(res)
        # for i in range(index_totalCentroid+1, corners.shape[0]):
        #     cv2.circle(img, (corners[i][0], corners[i][1]), 5, (0, 255, 0), -1)
        img = cv2.drawContours(img, [corners_int], -1, (0, 255, 0), 3)
        cv2.imshow("Circled corners", img)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    """There should be exact 10 corners in a 2D AR tag"""
    if corners.shape == (11, 2):  # if the corner contains the total centroid
        corners = corners[index_totalCentroid + 1:, :]
    assert corners.shape == (10, 2), "# of corner is " + str(corners.shape[0])
    return corners


def shiftToASpot(points, spot):
    """
    get a set of points in rows: [[x1, y1][x2, y2][x3, y3][x4, y4]...],
    and change the frame of coordinates to the geometrical center of points
    by subtract each points coordinates by the mean
    """
    assert points.shape[1] == spot.shape[0], "You wanna move from dimension of " + str(
        points.shape) + " to dimension of " + str(spot.shape)
    center = np.mean(points, axis=0)
    points_shift = points - center + spot
    assert center.shape[0] == points.shape[1]  # make sure the mean were taken along column
    assert points_shift.shape == points.shape
    return points_shift, center


def cutImageToBlocks(img_bw_AR_tags, NumberOfMesh):
    mesh_size_row = int(img_bw_AR_tags.shape[0] / NumberOfMesh)
    mesh_size_col = int(img_bw_AR_tags.shape[1] / NumberOfMesh)
    blocks = np.zeros([NumberOfMesh, NumberOfMesh])
    for row in range(0, NumberOfMesh):
        for col in range(0, NumberOfMesh):
            """start from here"""
            img_bw_mesh = img_bw_AR_tags[row*mesh_size_row:(row+1)*mesh_size_row, col*mesh_size_col:(col+1)*mesh_size_col]
            meean_img_mesh = np.mean(img_bw_mesh)
            if meean_img_mesh > 128:
                blocks[row, col] = 255
    return blocks.astype(int)


def findAngleAndID(img_gray_AR_tags, contour_outer, contour_inner):
    """figure out the angle and ID"""
    angle = 0   # the counter-wise rotational angle from upright AR tag to this being detected AR tag
    ID_one, ID_two, ID_three, ID_four = 0, 0, 0, 0      # ID of being detected AR tag
    ret, img_bw_AR_tags = cv2.threshold(img_gray_AR_tags, 200, 255, cv2.THRESH_BINARY)
    blocks = cutImageToBlocks(img_bw_AR_tags, Number_grid)  # get a 2D array that contains median of each grid of image
    if flag:
        None
        # print("The compressed version (8x8 grid with 8 bits value) of AR tag image is \n" + str(blocks))
    """detect the direction, judge if (2,2),(2,5),(5,2)(5,5) has value of 1, where upright direction has (5,5) to be 255"""
    if blocks[2, 2] == 255:
        angle = 180
        ID_one = blocks[4, 4]
        ID_two = blocks[4, 3]
        ID_three = blocks[3, 3]
        ID_four = blocks[3, 4]
        if flag:
            None
            # print("unique corner in top left")
    elif blocks[2, 5] == 255:
        angle = -90
        ID_one = blocks[4, 3]/255
        ID_two = blocks[3, 3]/255
        ID_three = blocks[3, 4]/255
        ID_four = blocks[4, 4]/255
        if flag:
            None
            # print("unique corner in top right")
    elif blocks[5, 5] == 255:
        angle = 0
        ID_one = blocks[3, 3]/255
        ID_two = blocks[3, 4]/255
        ID_three = blocks[4, 4]/255
        ID_four = blocks[4, 3]/255
        if flag:
            None
            # print("unique corner in bottom right")
    elif blocks[5, 2] == 255:
        angle = 90
        ID_one = blocks[3, 4]/255
        ID_two = blocks[4, 4]/255
        ID_three = blocks[4, 3]/255
        ID_four = blocks[3, 3]/255
        if flag:
            None
            # print("unique corner in bottom left")
    if flag:
        None
        # print("The ID follows, one: " + str(ID_one) + " two: " + str(ID_two) + " three: " + str(ID_three) + " four: " + str(ID_four))
    return angle, int(ID_one) + 2*int(ID_two) + 4*int(ID_three) + 8*int(ID_four)

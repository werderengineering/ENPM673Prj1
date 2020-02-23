from __main__ import *

flag = True
side_AR_Tag_estimation = 150
Number_grid = 8



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

def cube_top(perjectionMatrix, corners_inFrame, corners_flatView):
    """perjection Matrix is the perjection matrix from 3D to 2D
    corners_inFrame is a stack of points in frame view
    corners_flatView is a stack of points in flat view"""
    perjectionMatrix = perjectionMatrix / perjectionMatrix[2, 3]
    """make a top square in 3D homogeneous"""
    side = side_AR_Tag_estimation
    topCorners_flatView_3D_homogeneous = np.zeros([corners_flatView.shape[0], 4])
    topCorners_flatView_3D_homogeneous[:, 0] = corners_flatView[:, 0]
    topCorners_flatView_3D_homogeneous[:, 1] = corners_flatView[:, 1]
    topCorners_flatView_3D_homogeneous[:, 2] = topCorners_flatView_3D_homogeneous[:, 2] - side
    topCorners_flatView_3D_homogeneous[:, 3] = topCorners_flatView_3D_homogeneous[:, 3] + 1
    topCorners_flatView_3D_homogeneous = topCorners_flatView_3D_homogeneous.transpose()
    """cast the top corner to frame view"""
    topCorners_flatView_2D_homogeneous = np.dot(perjectionMatrix, topCorners_flatView_3D_homogeneous)
    topCorners_flatView_2D_homogeneous = topCorners_flatView_2D_homogeneous/topCorners_flatView_2D_homogeneous[2, :]
    topCorners_flatView_2D_homogeneous = topCorners_flatView_2D_homogeneous.transpose()
    return topCorners_flatView_2D_homogeneous

def projectionMatrix(intrinsic_camera_matrix, homo):
    Kinv = np.linalg.inv(intrinsic_camera_matrix)
    B = np.dot(Kinv, homo)

    det_B = np.linalg.det(B)
    if det_B < 0:
        B = B * -1
    elif det_B > 0:
        B = B
    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]
    Lamda = 2 / (np.linalg.norm(b1) + np.linalg.norm(b2))
    r1 = b1 * Lamda
    r2 = b2 * Lamda
    r3 = np.cross(r1, r2)
    t = (b3 * Lamda).reshape([3, 1])
    r = np.array([r1, r2, r3]).transpose()
    homogeneousTransformationMatrix = np.hstack([r, t])
    perjectionMatrix = np.dot(intrinsic_camera_matrix, homogeneousTransformationMatrix)
    return perjectionMatrix/perjectionMatrix[2, 3]
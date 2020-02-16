from __main__ import *

def dewarp(DWF,frame,H,box,ogframe):

    # print('############')
    x_max, y_max = np.amax(box, axis=0)
    x_min, y_min = np.amin(box, axis=0)


    Hinv=np.linalg.pinv(H)

    Hinv=Hinv/Hinv[2,2]
    # Hinv = H/ H[2, 2]

    for ix in range(x_min,x_max):
        for iy in range(y_min,y_max):
            p1=np.transpose(np.array([iy,ix,1]))

            p2=np.matmul(Hinv,p1)

            p2=(p2/p2[2]).astype(int)

            # print([p2[0],p2[1]])

            if p2[0]<0 or p2[1] <0 or p2[0] >= frame.shape[0] or p2[1] >= frame.shape[1]:
                None
            else:

                # print('############')
                # print(frame[p2[0],p2[1]])
                DWF[iy][ix]=ogframe[p2[0],p2[1]]

    # print(DWF.shape)
    return DWF
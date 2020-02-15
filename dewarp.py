from __main__ import *

def dewarp(DWF,frame,H,pixX,pixY):

    # Hinv=np.linalg.inv(H)
    #
    # Hinv=Hinv/Hinv[2,2]

    Hinv = H / H[2, 2]

    for ix in range(0,pixX):
        for iy in range(0,pixY):
            p1=np.transpose(np.array([iy,ix,1]))

            p2=np.dot(Hinv,p1)

            p2=(p2/p2[2]).astype(int)

            # print([p2[0],p2[1]])

            if p2[0]<0 or p2[1] <0 or p2[0] >= frame.shape[0] or p2[1] >= frame.shape[1]:
                None
            else:

                # print('############')
                # print(frame[p2[0],p2[1]])
                DWF[iy,ix]=frame[p2[0],p2[1]]

    # print(DWF.shape)
    return DWF
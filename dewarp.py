from __main__ import *

def dewarp(DWF,frame,H,box):


    x_max, y_max = np.amax(box, axis=0).astype(int)
    x_min, y_min = np.amin(box, axis=0).astype(int)

    Hinv=H


    for ix in range(x_min,x_max):
        for iy in range(y_min,y_max):
            p1=np.transpose(np.array([ix,iy,1]))

            p2=-np.matmul(Hinv,p1)

            p2=(p2/p2[2]).astype(int)

            try:


                if (p2[0] < frame.shape[1]) and (p2[1] < frame.shape[0]) and (p2[0] > -1) and (p2[1] > -1):
                    # print('############')
                    # print(frame[p2[0],p2[1]])
                    DWF[iy,ix]=frame[p2[0],p2[1]]
            except:
                None
                print('Cant dewarp')

    # print(DWF.shape)
    return DWF
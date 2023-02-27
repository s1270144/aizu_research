import numpy as np
import cv2
import time
from BladeCapture1 import BladeCapture
from cmath import phase
from math import degrees
import math
import cmath

PCOM = 5
comx = 0
comy = 0
comh = 0
mvhgt = 1920
mvwdh = 1080
fov = 214
fr = 30
size = 1.0
offset = 10
qnum = 2
d12 = 1.17
d23 = 0.9
d31 = 1.17

# Capture Setting (Hard Code)
# Captrur: 1440x1440, Cutout: 500x500, Shrink 0.6
#v1 = ThreadingVideoCapture(0,1440,1440,15,500,500,0.6,2)
#v2 = ThreadingVideoCapture(2,1440,1440,15,500,500,0.6,2)
#v3 = ThreadingVideoCapture(4,1440,1440,15,500,500,0.6,2)
#v1 = BladeCapture('drone3C.mp4')
v1 = BladeCapture('./20220822_insta360/1/Cam1-converted_1.mp4',mvhgt,mvwdh,fr,size,offset,qnum)
#v1 = BladeCapture('C:\\Users\\s1270144\\Husha\\cam1-converted20.mp4',mvhgt,mvwdh,fr,size,offset,qnum)
#v2 = BladeCapture('C:\\Users\\s1270144\\Husha\\cam2-converted.mp4',mvhgt,mvwdh,fr,size,offset,qnum)
#v3 = BladeCapture('C:\\Users\\s1270144\\Husha\\cam3-converted.mp4',mvhgt,mvwdh,fr,size,offset,qnum)

# Capture Error Handling
if not v1.isOpened():
    raise RuntimeError("Error: V1 is not open")
#if not v2.isOpened():
#    raise RuntimeError("Error: V2 is not open")
#if not v3.isOpened():
#    raise RuntimeError("Error: V3 is not open")

# Capture Start
v1.start()
#v2.start()
#v3.start()

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

while True:
    # Cycle Time Calculation
    tim = time.time()
    # Extract Info
    ret1, frm1, t1 = v1.read()
#    ret2, frm2, t2 = v2.read()
#    ret3, frm3, t3 = v3.read()
    tim2 = time.time()
    print("Main: ReadTime:", tim2 - tim)

    # Error Handling
    if not ret1:
        print ('Main: CAM ERROR')
        break

    tim4 = time.time()

    # Visualization
    frm4 = np.zeros(frm1.shape, dtype = np.uint8)
    cv2.drawMarker(frm4, (int(t1[0]/2), int(t1[1]/2)), [64,64,255], markerType=cv2.MARKER_CROSS, markerSize=4, thickness=2, line_type=cv2.LINE_8)
    #cv2.drawMarker(frm4, (int(t2[0]/4), int(t2[1]/4)), [64,64,255], markerType=cv2.MARKER_CROSS, markerSize=4, thickness=2, line_type=cv2.LINE_8)
    #cv2.drawMarker(frm4, (int(t3[0]/4), int(t3[1]/4)), [64,64,255], markerType=cv2.MARKER_CROSS, markerSize=4, thickness=2, line_type=cv2.LINE_8)
    himg1 = np.hstack((frm4, frm1))
    #himg2 = np.hstack((frm2, frm3))
    mImg = himg1


    cv2.imshow('frame', cv2.resize(mImg,(mvhgt,mvwdh)))
    key = cv2.waitKey(int(1000/30))
    #cv2.imwrite(str(tim)+'_1.png', mImg)
    tim5 = time.time()
    print("Main: VisualTime:", tim5 - tim4)

    if key == ord('q'):
        v1.loopflag = False
        #v2.loopflag = False
        #v3.loopflag = False
        time.sleep(1)
        v1.kill()
        #v2.kill()
        #v3.kill()
        break
    elif key == ord('z'):
        v1.loopflag = False
        #v2.loopflag = False
        #v3.loopflag = False
        time.sleep(1)
        v1.end()
        #v2.end()
        #v3.end()
        v1.gCutInit()
        #v2.gCutInit()
        #v3.gCutInit()
        v1.loopflag = True
        #v2.loopflag = True
        #v3.loopflag = True
        v1.start()
        #v2.start()
        #v3.start()
    print("Main: Throughput - ",time.time()-tim)

v1.release()
#v2.release()
#v3.release()
cv2.destroyAllWindows()

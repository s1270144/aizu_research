import threading
import queue
import numpy as np
import cv2
import time
import sys
import math
from collections import deque
from matplotlib import pyplot as plt

class BladeCapture():

    # Primitive Color
    CR_RECT = [255,0,0]     # rectangle color
    CR_CENT = [64,172,0]    # Center Pole Position
    CR_CAMR = [172,0,64]    # Right Camera Position
    CR_CAML = [0,64,172]    # Left Camera Position
    CR_FG = [0,255,0]       # Sure FG
    CR_BG = [0,0,255]       # Sure BG
    CR_PFG = [193,193,193]  # Prob FG
    CR_PBG = [31,31,31]     # Prob BG

    # Region Color Definition
    DRAW_BG = {'color' : CR_BG, 'val' : 0}
    DRAW_FG = {'color' : CR_FG, 'val' : 1}
    DRAW_PR_BG = {'color' : CR_PBG, 'val' : 2}
    DRAW_PR_FG = {'color' : CR_PFG, 'val' : 3}

    # Global Variables
    src = None                      # Source File or Camera ID
    video = None                    # Videocapture
    fourcc = None                   # FourCC
    width, height, fps = 1920, 1080, 30 # Video width, height, fps
    rect = (0,0,0,0)                # Target Rectangle
    offset = 10                     # Target rectangle offset
    point = [0,0,0,0]               # Target Point
    center = (0,0)                  # Image Center
    cppos = (0,0)                   # UAV Center Pole Position
    rcpos = (0,0)                   # Right Camera Position
    lcpos = (0,0)                   # Left Camera Position
    tgtpnt = (0,0)                  # Blade Target
    pretgtpnt = (0,0)
    size = 1.0                      # Image Shrinking
    thickness = 5                   # Mask Drawing Thckness
    q, p = None, None               # q: Image buffer, p: Target point buffer
    qq, pp = None, None               # q: Image buffer, p: Target point buffer
    ix, iy = 0, 0                   # Mouse
    xx, yy = [0,0], [0,0]           # left, right, top, bottom bounds of boundary
    rectanP = [0 for i in range(4)] # Target Rectangle
    loopflag = True
    errorCntShrink = 0
    errorCnt = 0
    resizeFlag = True
    target = (0,400) # right tip
    # Image Buffers
    #img, img2, mask, out, cont, nimg, nmsk = None, None, None, None, None, None, None

    # Global flags
    _value = DRAW_FG                # Drawing
    _drawingMask = False            # Mask drawing
    _drawingRect = False            # Target rectangle drawing
    _enableRect = False             # Rectangle drawed
    _enableMask = -1                # Mask drawed
    _mode = 1                       #
    _stopped = False                #
    _routine = False                # Setup or _routine

    # Class Initialization
    def __init__(self, src, width=1920, height=1080, fps=15, size=1.0, offset=10,max_queue_size=8):
        # Sourse Definition (capture ID, resource)
        self.src = src
        self.fps = fps
        self.mq = max_queue_size
        # Capture Setting
        self.video = cv2.VideoCapture(src)
        self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.video.set(cv2.CAP_PROP_FPS, fps)
        # Capture Setting Refrection
        self.fourcc = self.decode_fourcc(self.video.get(cv2.CAP_PROP_FOURCC))
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        # Process Setting
        self.size = size
        self.offset = offset
        self.center = (int(width*size/2),int(height*size/2))
        print(self.center)
        self.qq = deque(maxlen=1)
        self.pp = deque(maxlen=1)
        self.q = queue.Queue(maxsize=max_queue_size)
        self.p = queue.Queue(maxsize=max_queue_size)
        # Class Inner Value Setting
        self._stopped = False
        self._mode = 1
        # Grubcut Initialization
        self.gCutInit()

    ###
    # FourCC Decoding from Video Capture
    def decode_fourcc(self,v):
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

    ###
    # Mouse Interface
    def onmouse(self, event, x, y, flags, param):

        # Pointing center pole position
        if self._mode == 0:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.cppos = (x,y)
                self.img = self.img2.copy()
                cv2.drawMarker(self.img, self.cppos, self.CR_CENT, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                cv2.drawMarker(self.img, self.rcpos, self.CR_CAMR, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                cv2.drawMarker(self.img, self.lcpos, self.CR_CAML, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                cv2.rectangle(self.img, (self.rect[0], self.rect[1]), (self.rect[0]+self.rect[2], self.rect[1]+self.rect[3]), self.CR_RECT, 2)
                ### Todo - Draw mask ###

        # Pointing right camera position
        elif self._mode == 3:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.rcpos = (x,y)
                self.img = self.img2.copy()
                cv2.drawMarker(self.img, self.cppos, self.CR_CENT, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                cv2.drawMarker(self.img, self.rcpos, self.CR_CAMR, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                cv2.drawMarker(self.img, self.lcpos, self.CR_CAML, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                cv2.rectangle(self.img, (self.rect[0], self.rect[1]), (self.rect[0]+self.rect[2], self.rect[1]+self.rect[3]), self.CR_RECT, 2)
                ### Todo - Draw mask ###

        # Pointing right camera position
        elif self._mode == 4:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.lcpos = (x,y)
                self.img = self.img2.copy()
                cv2.drawMarker(self.img, self.cppos, self.CR_CENT, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                cv2.drawMarker(self.img, self.rcpos, self.CR_CAMR, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                cv2.drawMarker(self.img, self.lcpos, self.CR_CAML, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                cv2.rectangle(self.img, (self.rect[0], self.rect[1]), (self.rect[0]+self.rect[2], self.rect[1]+self.rect[3]), self.CR_RECT, 2)
                ### Todo - Draw mask ###

        # Drawing target rectangle
        elif self._mode == 1:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._drawingRect = True
                self.ix, self.iy = x,y
                self.point[0], self.point[1] = self.ix, self.iy
                cv2.drawMarker(self.img, self.cppos, self.CR_CENT, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                cv2.drawMarker(self.img, self.rcpos, self.CR_CAMR, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                cv2.drawMarker(self.img, self.lcpos, self.CR_CAML, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self._drawingRect == True:
                    self.img = self.img2.copy()
                    cv2.drawMarker(self.img, self.cppos, self.CR_CENT, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                    cv2.drawMarker(self.img, self.rcpos, self.CR_CAMR, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                    cv2.drawMarker(self.img, self.lcpos, self.CR_CAML, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                    cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.CR_RECT, 2)
                    self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                    self._enableMask = 0
            elif event == cv2.EVENT_LBUTTONUP:
                if self._drawingRect:
                    self._drawingRect = False
                    self._enableRect = True
                    self.point[2], self.point[3] = x, y
                    cv2.drawMarker(self.img, self.cppos, self.CR_CENT, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                    cv2.drawMarker(self.img, self.rcpos, self.CR_CAMR, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                    cv2.drawMarker(self.img, self.lcpos, self.CR_CAML, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
                    cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.CR_RECT, 2)
                    self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                    self._enableMask = False
                    self.pimg = self.img[self.rect[1]:self.rect[1]+self.rect[3],self.rect[0]:self.rect[0]+self.rect[2],:]
                    self.pmask = self.pimg[:,:,0]
                    print(" Now press the key 'n' a few times until no further change \n")

        # Draw Touchup Curves
        elif self._mode == 2:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._drawingMask = True
                cv2.circle(self.img, (x,y), self.thickness, self._value['color'], -1)
                cv2.circle(self.mask, (x,y), self.thickness, self._value['color'], -1)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self._drawingMask:
                    cv2.circle(self.img, (x, y), self.thickness, self._value['color'], -1)
                    cv2.circle(self.mask, (x, y), self.thickness, self._value['color'], -1)
            elif event == cv2.EVENT_LBUTTONUP:
                if self._drawingMask:
                    self._drawingMask = False
                    self._enableMask = True
                    cv2.circle(self.img, (x, y), self.thickness, self._value['color'], -1)
                    cv2.circle(self.mask, (x, y), self.thickness, self._value['color'], -1)
        # Else: Nothing
        else:
            None

    #=====================================
    # graphCut Method
    # - We should not share the global variable here
    # img:
    def graphCut(self, img):
        # Model for Background and Foreground (Initialize)
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        off = self.offset #10
        pimg = self.pimg
        pmsk = self.pmask
        h,w,ll = pimg.shape
        print("fps: ",self.fps)
        try:
            # Tracking: Template Matching
            if self._routine :
                # Target Rectangle: 5x5 wide size
                #x1, y1, x2, y2 = max([0,self.rect[0]-2*self.rect[2]]), max([0,self.rect[1]-2*self.rect[3]]), min([self.width-1,self.rect[0]+3*self.rect[2]]), min([self.height-1,self.rect[1]+3*self.rect[3]])
                x1, y1, x2, y2 = max([0,self.rectanP[0]-w]), max([0,self.rectanP[1]-h]), min([self.width-1,self.rectanP[2] +w]), min([self.height-1,self.rectanP[3]+h])
                #print(x1, y1, x2, y2)
                cutimg = img[int(y1):int(y2),int(x1):int(x2),:]
                res = cv2.matchTemplate(cutimg[:,:,0], pimg[:,:,0], cv2.TM_CCOEFF_NORMED)
                #cv2.imshow('image',cutimg[:,:,0])
                #cv2.imshow('template',pimg[:,:,0])
                #print(res/np.amax(res))
                #cv2.imshow('cmap',res/np.amax(res))
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                # Pickup target rectangle for Grabcut
                x1, y1 = x1 + top_left[0], y1 + top_left[1]
                x2, y2 = x1 + w, y1 + h
                print("testtestt",top_left)
            else:
                # Pickup target rectangle for Grabcut
                x1, y1, x2, y2 = self.rect[0], self.rect[1], self.rect[0]+self.rect[2], self.rect[1]+self.rect[3]
                top_left =(0,0)
        except:
            print("ERROR_IN_TM")
            import traceback
            traceback.print_exc()
        x3, y3, x4, y4 = self.rectanP[0], self.rectanP[1], self.rectanP[2], self.rectanP[3]
        # Preparation for GrabCut
        # Cut position should manage by LectanP and offset (top_left[0] + top_left[1])
        cutimg  = img[int(y1-off):int(y2+off),int(x1-off):int(x2+off),:]  # Cutted default image
        cutimg2 = cutimg
#       cutgray = cv2.cvtColor(cutimg, cv2.COLOR_BGR2GRAY)  # Cutted grayscale image
        cutgray = cutimg[:,:,0]  # Cutted grayscale image
        cutmsk  = np.zeros(cutimg.shape[:2],dtype=np.uint8) # Cutted mask Image
        cutmsk2 = cutmsk.copy()                             # Cutted masked Region
        cutrct  = (off,off,self.rect[2],self.rect[3])       # Cutted area rectangle
        # Mask region setting (to change internal setting)
        cutmsk[cutmsk == 0] = 2 #PBG
        cutmsk[self.mask[y3-off:y4+off,x3-off:x4+off,2]==255] = 0 #BG
        cutmsk[self.mask[y3-off:y4+off,x3-off:x4+off,1]==255] = 1 #FG
        cutmsk[self.mask[y3-off:y4+off,x3-off:x4+off,1]==193] = 3 #PFG
        cutmsk[self.mask[y3-off:y4+off,x3-off:x4+off,1]==31]  = 2 #PBG
        #cv2.imshow('cutimg', cutimg)
        #cv2.imshow('cutmsk', cutmsk*50)
        #cv2.waitKey(10)
        s = cutmsk.shape
        hB, wB, cB = cutimg.shape
        hA, wA, cA = hB, wB, cB
        print(s, ",", cutrct," -  (",wB,"," ,hB,")")
        self.shcnt = 0;
        #'''Resize-> shrink ###############################################################################################
        while(hA > 200 or wA > 200):
            self.errorCntShrink += 1
            self.shcnt += 1
            cutimg = cv2.resize(cutimg , (int(wA*0.5), int(hA*0.5)), interpolation=cv2.INTER_NEAREST)
            #cutgray = cv2.cvtColor(cutimg, cv2.COLOR_BGR2GRAY)  # Cutted grayscale image
            cutmsk  = cv2.resize(cutmsk, (int(wA*0.5), int(hA*0.5)), interpolation=cv2.INTER_NEAREST) # Cutted mask Image
            hA, wA, cA = cutimg.shape
            sA = cutmsk.shape
            cutrct  = (int(off/(2*self.shcnt)),int(off/(2*self.shcnt)),int(self.rect[2]/(2*self.shcnt)),int(self.rect[3]/(2*self.shcnt)))
            print("Shrink:", self.shcnt, ", ", sA, ",", cutrct," -  (",wA,"," ,hA,")" )
        #'''

        # Grabcut Region Detection
        try:
            if (self._enableRect and not self._enableMask):   # grabcut with rect
                cv2.grabCut(cutimg, cutmsk, cutrct, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_RECT)
                self._enableMask = 1
            elif (self._enableRect and self._enableMask):     # grabcut with mask
                cv2.grabCut(cutimg, cutmsk, cutrct, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_MASK)
            else:
                return
        except:
            try:
                cv2.grabCut(cutimg, cutmsk, cutrct, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_RECT)
                self._enableMask = 1
            except:
                print("ERROR_IN_GCUT")
                import traceback
                traceback.print_exc()

        #'''Resize-> originalSize ###########################################################################################
        if(self.shcnt > 0):
            #cutimg = cv2.resize(cutimg , (wB, hB))
            cutimg = cutimg2
            cutgray = cv2.cvtColor(cutimg, cv2.COLOR_BGR2GRAY)  # Cutted grayscale image
            cutmsk  = cv2.resize(cutmsk , (wB, hB), interpolation=cv2.INTER_NEAREST)# Cutted mask Image
            cutmsk2 = cutmsk.copy()  # Cutted masked Region
            cutrct  = (off,off,self.rect[2],self.rect[3])
        #'''

        # Pickup masked region and image
        cutmsk2 = np.where((cutmsk==1) + (cutmsk==3), 255, 0).astype('uint8')
        cutgray2 = cv2.bitwise_and(cutgray, cutgray, mask=cutmsk2)
        #cv2.imshow('cutmsk2', cutmsk2)
        self.pmask = cutmsk
        #cv2.imshow('cutgray',cutgray2)
        #cv2.waitKey(3)
        # Contour Extraction
        contours, hierarchy = cv2.findContours(cutgray2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        X, Y = [], []   # Contour List
        mi, ms = 0,0    # Maxarea index & Maxarea size

        # Extract maximum area index and size
        for i in range(len(contours)):
            if ms < cv2.contourArea(contours[i]):
                mi = i
                ms = cv2.contourArea(contours[i])
        if ms == 0:
            print("Cannot Track")
            return
        m1 = np.zeros(cutimg.shape[:2], dtype = np.uint8)   # Working Contour
        m2 = m1.copy()                                      # Cutted Next Mask
        # Extract maximum contour
        cv2.fillPoly(m1,pts=[contours[mi]], color=255)
        # Morphology Calculation
        k1 = np.ones((5,5),np.uint8)            # Definite Foreground Kernel
        k2 = np.ones((40,40),np.uint8)          # Probably Background Kernel
        k3 = np.ones((50,50),np.uint8)          # Definite Background Kernel
        k4 = np.ones((3,3),np.uint8)            # Definite Foreground Kernel
        er = cv2.erode(m1,k1,iterations = 1)    # Definite Foreground
        # Small region - ersion, Big-region - thinning
        if (ms < 100):
            s1 = er.copy()
            print('copy')
        else:
            s1 = cv2.ximgproc.thinning(er, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            print('thinning')
        s2 = cv2.dilate(s1,k4,iterations = 1)
        #cv2.imshow('test',s1)
        er2 = cv2.dilate(er,k1,iterations = 1)  # Probably Foreground
        di1 = cv2.dilate(er,k2,iterations = 1)  # Probably Background
        di2 = cv2.dilate(er,k3,iterations = 1)  # Definite Background
        # Joint information for next mask
        m2[er == 0] = 2                         # Geneally, every pixel was prob. back
        m2[di2 == 255] = 0                      # Def. back line
        m2[di1 == 255] = 2                      # Inner prob. back
        m2[er2 == 255] = 3                      # Prob. fore & exact shape
        m2[s2 == 255] = 1                       # Def. fore.

        # Find blade tip (far from camera center)
        dist = 0.0
        #center = (self.center[0]-x1, self.center[1]-y1) # Cutted Center
        center = self.target
        # Limitation: Only contoured region
        for i in range(len(contours[mi])):
            # Limitation: Only foreground (estimated) region
            if(m2[contours[mi][i][0][1],contours[mi][i][0][0]] == 1 or m2[contours[mi][i][0][1],contours[mi][i][0][0]] == 3):
                dd = math.sqrt((center[0]-contours[mi][i][0][0])**2 + (center[1]-contours[mi][i][0][1])**2)
                if dist < dd:
                    dist = dd
                    self.tgtpnt = (x1 + contours[mi][i][0][0], y1 + contours[mi][i][0][1])
                X.append(contours[mi][i][0][0])
                Y.append(contours[mi][i][0][1])

        #print(self.tgtpnt)
        #print(x1 + min(X) - 40)

        # Calculate next rectangle region
        self.rectanP = [0 for i in range(4)]
        if len(X)!=0 and len(Y)!=0:
            self.rectanP[0] = max(x1 + min(X) - 70, self.tgtpnt[0] - 150)   # left
            self.rectanP[1] = max(y1 + min(Y) - 70, self.tgtpnt[1] - 150)   # top
            self.rectanP[2] = min(x1 + max(X) + 70, self.tgtpnt[0] + 150)   # right
            self.rectanP[3] = min(y1 + max(Y) + 70, self.tgtpnt[1] + 150)   # bottom

        #print(self.rectanP)

        # Pickup next mask
        self.img = img.copy()
        if len(X)!=0 and len(Y)!=0:
            self.pimg = img[self.rectanP[1]:self.rectanP[3],self.rectanP[0]:self.rectanP[2],:]
        #cv2.imshow('pimg', pimg)
        #cv2.waitKey(1000)
        cv2.circle(self.img, self.tgtpnt, 10, (0, 0, 255), -1, 8, 0) #(青,緑,赤)
        self.errorCnt += 1
        cv2.imwrite('./right_tip/{}_tip.png'.format(str(self.errorCnt)), self.img)
        cv2.circle(self.img, (x1,y1), 7, (255, 0, 0), -1, 8, 0) # preFrame
        cv2.rectangle(self.img, (self.rectanP[0], self.rectanP[1]), (self.rectanP[2],self.rectanP[3]), self.CR_RECT, 2) #next rectangle

        # Record the tip coordinates
        '''
        file = "cam1_70_130.txt"
        fileobj = open(file, "a", encoding = "utf_8")
        fileobj.write("tgtpnt: " + str(self.tgtpnt) + "   rectanP: " + str(self.rectanP))
        fileobj.write("\n")
        fileobj.close()
        '''

        bimg = self.img[y1-10:y2+10, x1-10:x2+10, 0]
        gimg = self.img[y1-10:y2+10, x1-10:x2+10, 1]
        rimg = self.img[y1-10:y2+10, x1-10:x2+10, 2]
        bimg[m2==0] = self.CR_BG[0]
        gimg[m2==0] = self.CR_BG[1]
        rimg[m2==0] = self.CR_BG[2]
        bimg[m2==1] = self.CR_FG[0]
        gimg[m2==1] = self.CR_FG[1]
        rimg[m2==1] = self.CR_FG[2]
        bimg[m2==3] = self.CR_PFG[0]
        gimg[m2==3] = self.CR_PFG[1]
        rimg[m2==3] = self.CR_PFG[2]
        self.img[y1-10:y2+10, x1-10:x2+10, 0] = bimg
        self.img[y1-10:y2+10, x1-10:x2+10, 1] = gimg
        self.img[y1-10:y2+10, x1-10:x2+10, 2] = rimg
        if self._routine and len(X)!=0 and len(Y)!=0:
            # NEXTIMG: For visualization
            nmsk = np.zeros(self.img.shape, dtype = np.uint8) # Next mask buffer
            #self.rect = (self.rectanP[0],self.rectanP[1],self.rectanP[2]-self.rectanP[0],self.rectanP[3]-self.rectanP[1])
            bimg = nmsk[y1-10:y2+10, x1-10:x2+10, 0]
            gimg = nmsk[y1-10:y2+10, x1-10:x2+10, 1]
            rimg = nmsk[y1-10:y2+10, x1-10:x2+10, 2]
            bimg[m2==0] = self.CR_BG[0]
            gimg[m2==0] = self.CR_BG[1]
            rimg[m2==0] = self.CR_BG[2]
            bimg[m2==1] = self.CR_FG[0]
            gimg[m2==1] = self.CR_FG[1]
            rimg[m2==1] = self.CR_FG[2]
            bimg[m2==2] = self.CR_PBG[0]
            gimg[m2==2] = self.CR_PBG[1]
            rimg[m2==2] = self.CR_PBG[2]
            bimg[m2==3] = self.CR_PFG[0]
            gimg[m2==3] = self.CR_PFG[1]
            rimg[m2==3] = self.CR_PFG[2]
            nmsk[y1-10:y2+10, x1-10:x2+10, 0] = bimg
            nmsk[y1-10:y2+10, x1-10:x2+10, 1] = gimg
            nmsk[y1-10:y2+10, x1-10:x2+10, 2] = rimg
            self.mask = nmsk.copy()

    #==================================
    # gCutInit: グラブカットの初期設定値を収集する
    # - 基本的に初期設定値はクラスGlobal変数に格納しておきたい
    # 初期設定値として必要なもの
    #  > 初期マスク画像 (FG: CR_CENT(0,255,0), BG: CR_BG (0,0,255), PR_FG: (255,127,63), PR_BG: (31,31,31))
    #   - マスクへの変換: 3番目が 0, 255, 63, 31になるので、それぞれ 1, 0, 3, 2にgraphCut内で変換
    #  > 初期領域 (rect: x1, y1, w, h)
    #   - 使用領域への変換: カットする場所はx1,y1,x1+w-1, y1+h-1になる
    #  > ターゲットマストの付け根位置 (x,y)
    #  > 他の２つのカメラ位置 (x1, y1), (x2, y2) -> 時計周りに指定
    def gCutInit(self):
        # 初期設定値終了フラグ
        self._routine = False;
        self._enableRect = True;
        while True:
            ret, img = self.video.read() # read a shot
            self.img = img.copy()                                    #Visualize Buffer
            self.img2 = img.copy()                                   # Working Buffer
            self.mask = np.zeros(self.img.shape, dtype = np.uint8)   # Mask Buffer (CR_PBG)
            self.out = np.zeros(self.img.shape, dtype = np.uint8)    # Output Buffer
            self.cont = np.zeros(self.img.shape, dtype = np.uint8)   # Contour Image Buffer
            # Windows Definition
            cv2.namedWindow('input',cv2.WINDOW_GUI_NORMAL)
            cv2.setMouseCallback('input', self.onmouse)
            cv2.moveWindow('input', 80,10)

            # Local Variable Initialization

            cv2.drawMarker(self.img, self.cppos, self.CR_CENT, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
            cv2.drawMarker(self.img, self.rcpos, self.CR_CAMR, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)
            cv2.drawMarker(self.img, self.lcpos, self.CR_CAML, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)

            # Help Indicate
            print(" Instructions: \n")
            print(" Draw a rectangle around the object using right mouse button \n")

            while(1):
                # View input and output images
                cv2.imshow('input', self.img)

                k = cv2.waitKey(10)
                # key bindings [ESC,0,1,2,3,4,r,n]
                if k == 27:         # esc to exit
                    self._routine = True;
                    print(self._routine, self._mode, self._enableMask, self._enableRect)
                    cv2.destroyAllWindows()
                    return
                elif k == ord('1'): # BG drawing
                    if(self._enableRect):
                        print(" mark background regions with left mouse button \n")
                        self._mode = 2
                        self._value = self.DRAW_BG
                    else:
                        print(" Mark interest region first.\n")
                elif k == ord('2'): # FG drawing
                    if(self._enableRect):
                        print(" mark foreground regions with left mouse button \n")
                        self._mode = 2
                        self._value = self.DRAW_FG
                    else:
                        print(" Mark interest region first.\n")
                elif k == ord('3'): # PR_BG drawing
                    if(self._enableRect):
                        print(" Mark PROBABLY background regions with left mouse button \n")
                        self._mode = 2
                        self._value = self.DRAW_PR_BG
                    else:
                        print(" Mark interest region first.\n")
                elif k == ord('4'): # PR_FG drawing
                    if(self._enableRect):
                        print(" mark PROBABLY foreground regions with left mouse button \n")
                        self._mode = 2
                        self._value = self.DRAW_PR_FG
                    else:
                        print(" Mark interest region first.\n")
                elif k == ord('c'): # center pole pointing
                    if(self._enableRect):
                        print(" point the top of center pole of the drone \n")
                        self._mode = 0
                    else:
                        print(" Mark interest region first.\n")
                elif k == ord('r'): # right camera pointing
                    if(self._enableRect):
                        print(" point the left camera position of the drone \n")
                        self._mode = 3
                    else:
                        print(" Mark interest region first.\n")
                elif k == ord('l'): # left camera pointing
                    if(self._enableRect):
                        print(" point the right camera position of the drone \n")
                        self._mode = 4
                    else:
                        print(" Mark interest region first.\n")
                elif k == ord('a'): # reset everything and get current frame
                    print("resetting and get current frame\n")
                    self._mode = 1
                    self.point = [0,0,0,0]
                    flag = 0
                    self._drawingRect = False
                    self._drawingMask = False
                    self._enableRect = False
                    self._enableMask = -1
                    self._value = self.DRAW_FG
                    break # restart this function
                elif k == ord('n'): # segment the image
                    print(""" For finer touchups, mark foreground and background after pressing keys 0-4
                    and again press 'n' \n""")
                    try:
                        self.graphCut(self.img2)
                    except:
                        print("ERROR_IN_GCUTINIT")
                        import traceback
                        traceback.print_exc()

    def start(self):
        self.started = threading.Event()
        self.thread = threading.Thread(target=self.update )
        self._stopped = False
        self.thread.start()
        return self

    def begin(self):
        self.loopflag = True
        self.started.set()

    def end(self):
        self.started.clear()
        print("\nend")

    def kill(self):
        self.started.set()
        self._stopped = False
        self.thread.join()

    def update(self):
        # Start Thread
        self.qq = deque(maxlen=1)
        self.pp = deque(maxlen=1)
        self.q = queue.Queue(maxsize=self.mq)
        self.p = queue.Queue(maxsize=self.mq)
        while self.loopflag:
            ctime = time.time()
            # Thread Stop
            if self._stopped:
                return
            # get a Image
            ret, img = self.video.read()

            # if there is no image: stop
            if not ret:
                self.stop()
                return
            # Visualize Buffer
            self.img = img.copy()
            try:
                self.graphCut(img)
            except:
                print("ERROR_IN_UPDATE")
                import traceback
                traceback.print_exc()
                #cv2.destroyAllWindows()
                #self.gCutInit()
            if self.q.full():
                self.q.get()
                self.p.get()
            self.q.put((ret, cv2.resize(self.img,(720,720)), ctime))
            self.p.put((self.tgtpnt, ctime))
        self.started.wait()

    def readset(self):
        return(self.cppos, self.rcpos, self.lcpos)

    def read(self):
        #ret, frm, tim = self.qq.pop()
        #tgtpnt, tim2 = self.pp.pop()
        ret, frm, tim = self.q.get()
        tgtpnt, tim2 = self.p.get()
        err = time.time() - tim
        print(self.src, ":", err)
        return (ret,frm,tgtpnt)

    def stop(self):
        self._stopped = True

    def release(self):
        self._stopped = True
        self.video.release()

    def isOpened(self):
        return self.video.isOpened()

    def get(self,i):
        return self.video.get(i)

    def join(self):
        self.thread.join()

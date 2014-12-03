import cv2
import cv2.cv as cv
import math
import numpy as np
import moustache
import Tkinter
import dialog
import ntpath


# initialize the current frame of the video, along with the list of
# ROI points along with whether or not this is input mode
frame = None
roiPts = []
inputMode = False

def _ROI_tracking(frame, x_cord, height, y_cord, width):
    """
    set up the ROI for tracking
    Finds the Region of Interest (ROI) and
    finds the histogram weights for meanshift
    Outputs: returns the histogram for the tracking
    """

    roi = frame[x_cord:x_cord + height, y_cord:y_cord + width]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 180, cv2.NORM_MINMAX)
    return roi_hist
    
    
def selectROI(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode

    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        
def roi_selection():
    global frame, roiPts, inputMode
    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    
    # if the see if the ROI has been computed
    if roiBox is not None:
        # convert the current frame to the HSV color space
        # and perform mean shift
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
        # apply cam shift to the back projection, convert the
        # points to a bounding box, and then draw them
        (ret2, roiBox) = cv2.CamShift(backProj, roiBox, termination)
        pts = np.int0(cv2.cv.BoxPoints(ret2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)		
    # show the frame and record if the user presses a key
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # handle if the 'i' key is pressed, then go into ROI
    # selection mode
    if key == ord("i") and len(roiPts) < 4:
        # indicate that we are in input mode and clone the
        # frame
        inputMode = True
        orig = frame.copy()
        # keep looping until 4 reference ROI points have
        # been selected; press any key to exit ROI selction
        # mode once 4 points have been selected
        while len(roiPts) < 4:
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)
        # determine the top-left and bottom-right points
        roiPts = np.array(roiPts)
        sum_pts = roiPts.sum(axis = 1)
        top_left = roiPts[np.argmin(sum_pts)]
        bottom_right = roiPts[np.argmax(sum_pts)]
        # grab the ROI for the bounding box and convert it
        # to the HSV color space
        roi = orig[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        # compute a HSV histogram for the ROI and store the
        # bounding box
        roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
        roiBox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

def track_face(video, filename):
    face_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')
    big_eye_pair_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_mcs_eyepair_big.xml')
    small_eye_pair_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_mcs_eyepair_small.xml')
    eye_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier(
        'haarcascades/haarcascade_mcs_mouth.xml')
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode
    cap = video
    toList = []
    faceColor = (255, 255, 255) # white
    leftEyeColor = (0, 0, 255) # red
    rightEyeColor = (0, 255, 255) # yellow
    noseColor = (0, 255, 0) # green
    mouthColor = (255, 0, 0) # blue
    t = 0
    ret, image = video.read()
    frame_height, frame_width, frame_layers = image.shape
    fourcc = cv.CV_FOURCC('M', 'P', '4', 'V')
    print filename
    feature_test = cv2.VideoWriter('Overlaid Videos/',filename,'.mov', 
        fourcc, 30, (frame_width, frame_height))
    
    prev_e = [0, 0, 0, 0]
    prev_m = [0, 0, 0, 0]
    prev_f = [0, 0, 0, 0]
    
    # setup the mouse callback
    cv2.namedWindow("Overlayed Video")
    #cv2.setMouseCallback("Frame", selectROI)
    
    (chosen_mustache, chosen_eyes) = dialog.StartDialog()
    
    mustache_s = cv2.imread(chosen_mustache, -1)
    visor_s = cv2.imread(chosen_eyes, -1)

    while(1):
        if not ret:
            break
        framre = image.copy()
        
        #this is intended to be the method that keeps track of the
        #user selected ROI however, it is very broken and does not
        #track accurately and 
        roi_selection()
        
        mustache = mustache_s.copy()
        visor = visor_s.copy()
        #roi_overlay = overlay_image.copy()

        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, 
                        cv.CV_HAAR_SCALE_IMAGE |
                        cv.CV_HAAR_FIND_BIGGEST_OBJECT)
        
        if len(faces) > 0:
            prev_f = moustache.markov_add(prev_f, faces[0])
            print " ",
        else:
            print "!",
        x, y, width, height = prev_f
        print ("%03d (%3d,%3d) %3d" % (t, x, y, width)), ("x%3d" % (height)), " ",
        face_gray = gray[y:y+height, x:x+width]
        l_face_gray = face_gray[height/2:, :]
        u_face_gray = face_gray[:3*height/4, :]
        # cv2.imwrite("test_output/debug.png", l_face_gray)
        eyes = eye_cascade.detectMultiScale(u_face_gray)
        mouths = mouth_cascade.detectMultiScale(l_face_gray, 1.3, 5, cv.CV_HAAR_SCALE_IMAGE)
        # for (ex, ey, ew, eh) in eyes:
        # cv2.rectangle(image, (ex+x, ey+y), (ex+ew+x, ey+eh+y), leftEyeColor)
        if len(eyes) != 2:
            x1, y1, x2, y2 = prev_e
            print "!",
        else:
            x1a, y1a = moustache.box_center(*eyes[0])
            x2a, y2a = moustache.box_center(*eyes[1])
            x1, y1, x2, y2 = moustache.markov_add(prev_e, (x1a, y1a, x2a, y2a))
            prev_e = [x1a, y1a, x2a, y2a]
            print " ",
        print ("%1d (%3d,%3d)--(%3d,%3d)" % (len(eyes), x1, y1, x2, y2)), " "
        visor = moustache.scale_to(visor, width)
        visor = moustache.align_slope(visor, x1, y1, x2, y2)
        image = moustache.blit_draw(image, visor, (y1 + y2) / 2 + y, (x1 + x2) / 2 + x)
        if len(mouths) > 0:
            prev_m = moustache.markov_add(prev_m, mouths[0])
        mx, my, mw, mh = prev_m
        # cv2.rectangle(image, (mx+x, my+y+height/2), (mx+x+mw, my+y+mh+height/2), mouthColor)
        mustache = moustache.scale_to(mustache, width)
        image = moustache.blit_draw(image, mustache, my + y + mh / 2 + height / 2, mx + x + mw / 2)
        # cv2.rectangle(image, (x, y), (x+width, y+height), faceColor)

        '''if usen:
            for (x, y, width, height) in noses:
                cv2.rectangle(image, (x, y), (x+width, y+height), noseColor)''' 

        # cv2.imwrite("tracked_images/image_%(number)03d.jpg" % {"number" : t}, image)
        cv2.imshow("Overlayed Video", image)
        cv2.waitKey(1)
        feature_test.write(image)
        t += 1
        ret, image = video.read()

    cv2.destroyAllWindows()
    video.release()
    feature_test.release()

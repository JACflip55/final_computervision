"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
import cv2.cv as cv
import math
import numpy as np
import moustache


class Face(object):
    def __init__(self):
        self.faceRect = None
        self.leftEyeRect = None
        self.rightEyeRect = None
        self.eyes = None
        self.noseRect = None
        self.mouthRect = None


def _testing_balls():

    filename = "test_data/ball_3_frames/frame_%03d.png"
    video = cv2.VideoCapture(filename)
    bounds = track_ball_3(video)


def _test_face():

    filename = "test_data/face_frames/frame_%03d.png"
    video = cv2.VideoCapture(filename)
    bounds = track_face(video)


def _test_ETH():

    filename = "test_data/seq_eth/seq_eth.avi"
    video = cv2.VideoCapture(filename)
    bounds = ETH_tracking(video)


def read_and_grayscale(video):
    """
    Arguments: Video same as original argument
    Reads the first frame and coverts it grayscale

    Outputs
    ret: is the next frame in the sequence
    frame: the current frame
    gray: grayscale image
    """
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return ret, frame, gray


def findcircles(gray, part):
    """
    Finds the intial points (x,y and the radius) for the circles.
    HoughCircles does the detection of the circles for each part

    Arguments:
    gray: grayscale Image input
    part: the problem number (1-4)

    Outputs:
    _x_cord, _y_cord, _height, _width: X and Y coordinates
    and the height and width of the bounding box
    """
    if part == 3:
        c = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 50, 100)
        _x_cord = int(c[0][0][0] - c[0][0][2] - 1)
        _y_cord = c[0][0][1] - c[0][0][2]
        _height = int(1.5 * c[0][0][2])
        _width = int(1.5 * c[0][0][2])
        return _x_cord, _y_cord, _height, _width
    elif part != 4:
        c = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 3, 1)
    else:
        c = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 5, 1)
    if c is not None:
        _x_cord = c[0][0][0] - c[0][0][2]
        _y_cord = c[0][0][1] - c[0][0][2]
        _height = int(2 * c[0][0][2])
        _width = int(2 * c[0][0][2])
        return _x_cord, _y_cord, _height, _width
    else:
        return 0, 0, 10, 10


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


def _term_crit():
    """ finds the criteria for stopping the algorithm
        if a specific accuracy is met
        # Setup the termination criteria, either
        10 iteration or move by atleast 1 # pt
     """

    return (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


def track_ball(cap, frame, x_cord, y_cord, height, width, ret):
    """
    Arguments:
    cap: is the video that is passed in the original arguments
    frame: The current frame the video is on.
    x_cord: The min x coordinate of the bounding box in the intial frame
    y_cord: The min y coordinate of the bounding box in the intial frame
    height: The height  of the bounding box region
    Width: The width of the bounding box region

    Outputs:
    A list of four-tuples containing the four edges of the bounding box
    """
    roi_hist = _ROI_tracking(frame, x_cord, y_cord, height, width)
    term_crit = _term_crit()
    track_window = (x_cord, y_cord, height, width)
    tolist = [(x_cord, y_cord, height, width)]
    while(1):
        ret, frame = cap.read()

        if ret is True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            # Draw it on image
            x, y, width, height = track_window
            tolist.append((x, y, x + width, y + height))
            """
            cv2.imshow('img2', frame)

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k) + ".jpg", img2)
                """
        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    return tolist
    
    
def find_eyes(frame, cascade, fallbackCascade):
    minPairSize = (10, 10)
    haarScale = 1.3
    minNeighbors = 0
    #cv2.equalizeHist(frame, frame)

    eyes = cascade.detectMultiScale(frame, 1.5, 
                        minNeighbors, cv.CV_HAAR_FIND_BIGGEST_OBJECT
                        ,minPairSize, (99, 24))

    if len(eyes) == 0:
        eyes = fallbackCascade.detectMultiScale(frame, 1.2,
                        minNeighbors, cv.CV_HAAR_FIND_BIGGEST_OBJECT
                        |cv.CV_HAAR_SCALE_IMAGE, (5,5))

    return eyes


def track_face(video):
    face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    big_eye_pair_cascade = cv2.CascadeClassifier(
        'haarcascade_mcs_eyepair_big.xml')
    small_eye_pair_cascade = cv2.CascadeClassifier(
        'haarcascade_mcs_eyepair_small.xml')
    eye_cascade = cv2.CascadeClassifier(
        'haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier(
        'haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier(
        'haarcascade_mcs_mouth.xml')
    cap = video
    toList = []
    faceColor = (255, 255, 255)  # white
    leftEyeColor = (0, 0, 255)  # red
    rightEyeColor = (0, 255, 255)  # yellow
    noseColor = (0, 255, 0)  # green
    mouthColor = (255, 0, 0)  # blue
    t = 0
    ret, image = video.read()
    frame_height, frame_width, frame_layers = image.shape
    fourcc = cv.CV_FOURCC('M', 'P', '4', 'V')
    feature_test = cv2.VideoWriter('test_output/feature_test.mov', fourcc, 30, (frame_width, frame_height))
    
    prev_e = [0,0,0,0]
    prev_m = [0,0,0,0]
    prev_f = [0,0,0,0]
    
    mustache_s = cv2.imread('CurlyMustache.png',-1)
    visor_s = cv2.imread('visor.png',-1)
    
    while(1):
        if ret is True:
            mustache = mustache_s.copy()
            visor = visor_s.copy()
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(gray, gray)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, cv.CV_HAAR_SCALE_IMAGE | cv.CV_HAAR_FIND_BIGGEST_OBJECT)
            
            if len(faces) > 0:
                prev_f = faces[0]
                print " ",
            else:
                print "!",
            x, y, width, height = prev_f
            print ("%03d (%3d,%3d) %3d" % (t,x,y,width)),("x%3d" % (height))," ",
            face_gray = gray[y:y+height,x:x+width]
            l_face_gray = face_gray[height/2:,:]
            u_face_gray = face_gray[:3*height/4,:]
            #cv2.imwrite("test_output/debug.png", l_face_gray)
            eyes = eye_cascade.detectMultiScale(u_face_gray)
            mouths = mouth_cascade.detectMultiScale(l_face_gray, 1.3, 5, cv.CV_HAAR_SCALE_IMAGE)
            #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(image, (ex+x, ey+y), (ex+ew+x, ey+eh+y), leftEyeColor)
            if len(eyes) != 2:
                x1,y1,x2,y2 = prev_e
                print "!",
            else:
                x1,y1 = moustache.box_center(*eyes[0])
                x2,y2 = moustache.box_center(*eyes[1])
                prev_e = [x1,y1,x2,y2]
                print " ",
            print ("%1d (%3d,%3d)--(%3d,%3d)" % (len(eyes),x1,y1,x2,y2))," "
            visor = moustache.scale_to(visor,width)
            visor = moustache.align_slope(visor,x1,y1,x2,y2)
            image = moustache.blit_draw(image,visor, (y1 + y2) / 2 + y, (x1 + x2) / 2 + x)
            if len(mouths) > 0:
                prev_m = mouths[0]
            mx, my, mw, mh = prev_m
            #cv2.rectangle(image, (mx+x, my+y+height/2), (mx+x+mw, my+y+mh+height/2), mouthColor)
            mustache = moustache.scale_to(mustache, width)
            image = moustache.blit_draw(image, mustache, my + y + mh / 2 + height / 2, mx + x + mw / 2)
            #cv2.rectangle(image, (x, y), (x+width, y+height), faceColor)

            '''if usen:
                for (x, y, width, height) in noses:
                    cv2.rectangle(image, (x, y), (x+width, y+height), noseColor)''' 

            cv2.imwrite("tracked_images/image_%(number)03d.jpg" % {"number" : t}, image)
            feature_test.write(image)
        else:
            break
        t += 1
        ret, image = video.read()

    cv2.destroyAllWindows()
    video.release()
    feature_test.release()


def ETH_tracking(cap):

    fgbg = cv2.BackgroundSubtractorMOG2()
    while(1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)

        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

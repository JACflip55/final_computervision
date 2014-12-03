import cv2
import cv2.cv as cv
import math
import numpy as np
import moustache
import Tkinter
import dialog

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


def track_face(video):
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
    
    (chosen_mustache, chosen_eyes) = dialog.StartDialog()
    
    mustache_s = cv2.imread(chosen_mustache,-1)
    visor_s = cv2.imread(chosen_eyes,-1)

    while(1):
        if ret is True:
            mustache = mustache_s.copy()
            visor = visor_s.copy()
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(gray, gray)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, cv.CV_HAAR_SCALE_IMAGE | cv.CV_HAAR_FIND_BIGGEST_OBJECT)
            
            if len(faces) > 0:
                prev_f = moustache.markov_add(prev_f,faces[0])
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
                x1a,y1a = moustache.box_center(*eyes[0])
                x2a,y2a = moustache.box_center(*eyes[1])
                x1,y1,x2,y2 = moustache.markov_add(prev_e,(x1a,y1a,x2a,y2a))
                prev_e = [x1a,y1a,x2a,y2a]
                print " ",
            print ("%1d (%3d,%3d)--(%3d,%3d)" % (len(eyes),x1,y1,x2,y2))," "
            visor = moustache.scale_to(visor,width)
            visor = moustache.align_slope(visor,x1,y1,x2,y2)
            image = moustache.blit_draw(image,visor, (y1 + y2) / 2 + y, (x1 + x2) / 2 + x)
            if len(mouths) > 0:
                prev_m = moustache.markov_add(prev_m,mouths[0])
            mx, my, mw, mh = prev_m
            #cv2.rectangle(image, (mx+x, my+y+height/2), (mx+x+mw, my+y+mh+height/2), mouthColor)
            mustache = moustache.scale_to(mustache, width)
            image = moustache.blit_draw(image, mustache, my + y + mh / 2 + height / 2, mx + x + mw / 2)
            #cv2.rectangle(image, (x, y), (x+width, y+height), faceColor)

            '''if usen:
                for (x, y, width, height) in noses:
                    cv2.rectangle(image, (x, y), (x+width, y+height), noseColor)''' 

            #cv2.imwrite("tracked_images/image_%(number)03d.jpg" % {"number" : t}, image)
            feature_test.write(image)
        else:
            break
        t += 1
        ret, image = video.read()

    cv2.destroyAllWindows()
    video.release()
    feature_test.release()

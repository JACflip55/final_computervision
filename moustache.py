
import cv2
import math
import numpy

s_fname = 'CurlyMustache.png'

def draw(src_fname, dst_fname, face_y, face_x, face_w):
	mustache = scale_to(cv2.imread(s_fname,-1),face_w)
	height, width = mustache.shape[:2]
	dst = cv2.imread(src_fname)
	d_height, d_width = dst.shape[:2]

	for y in range (0, height): #(face_y, face_y + height):
		for x in range (0, width): #(face_x, face_x + width):
			dest_x = translate(x,width,face_x)
			dest_y = translate(y,height,face_y)
			if (dest_x >= 0) and (dest_x < d_width) and (dest_y >= 0) and (dest_y < d_height):
				if mustache[y][x].any() != 0:
					dst[dest_y][dest_x] = blend(dst[dest_y][dest_x], mustache[y][x])

	cv2.imwrite(dst_fname, dst)

def translate(cord,scale,position):
	return position + cord - (scale / 2)

def blend(base, overlay):
	alpha = overlay[3] / 255.0
	return base * (1 - alpha) + overlay[:3] * alpha

def compute_scale(overlay, target):
	height, width = overlay.shape[:2]
	scale = 1.0 * target / width
	return (int(width * scale),int(height * scale))

def scale_to(overlay, target):
	return cv2.resize(overlay,compute_scale(overlay,target))

"""
video = cv2.VideoCapture('test_data/face_frames/frame_%03d.png')

# use the default face classifier (provided by opencv)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
# array to return bounding box sizes
bounds = []
    
# current frame that is being processed
frame_number = 0
    
while True:
    # read a single frame from the video stream
    valid_frame, frame = video.read()
        
    if (frame_number == 0):
        writer = cv2.VideoWriter('test.mov', CV_FOURCC('P','I','M','1'), 30, (640, 426))
    # continue iterating through frames until no more remain
    if not valid_frame:
        break

    # use tracking on grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # run classifier on face
    faces = face_cascade. \
        detectMultiScale(gray, scaleFactor=1.2,
                        minNeighbors=4,
                        flags=cv2.cv.CV_HAAR_SCALE_IMAGE |
                        cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)

    # different behavior based on number of faces detected
    if(len(faces) == 1):
        # if only one face is detected,
        #   append the bounding box to the return array
        (x, y, w, h) = faces[0]
        bounds.append((x, y, w, h))
    # print (x, y, x + w, y + h)
    else:
        # if 0 or more than 1 face is detected,
        #   use bounding box from previous frame
        # relies on the fact that motion is smooth
        #   in general
        if(frame_number >= 1):
            bounds.append(boxes[frame_number - 1])
        else:
            bounds.append((0, 0, 0, 0))
        
    # increment frame count
    frame_number += 1
    
# release the stream
video.release()

#open writer
if not writer.isOpened():
    writer.open('test.mov', CV_FOURCC('P','I','M','1'), 30, (640, 426))

imread(
       #306-506 244-328
       
    for x in range(244, 328):
       for y in range(306, 506):
       if mustache[x-244][y-306] != [0, 0, 0]:
            dst[x][y] = mustache[x-w][y-h]
bounds
"""

import cv2
import math
import numpy

s_fname = 'CurlyMustache.png'

def draw(src, face_y, face_x, face_w):
	mustache = scale_to(cv2.imread(s_fname,-1),face_w)
	height, width = mustache.shape[:2]
	dst = src.copy()
	d_height, d_width = dst.shape[:2]

	for y in range (0, height):
		for x in range (0, width):
			dest_x = translate(x,width,face_x)
			dest_y = translate(y,height,face_y)
			if (dest_x >= 0) and (dest_x < d_width) and (dest_y >= 0) and (dest_y < d_height):
				if mustache[y][x].any() != 0:
					dst[dest_y][dest_x] = blend(dst[dest_y][dest_x], mustache[y][x])

	#cv2.imwrite("debug.png", dst)
	return dst

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

def process_frame(frame,dst_fname):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2,
                        minNeighbors=4,
                        flags=cv2.cv.CV_HAAR_SCALE_IMAGE |
                        cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
	for (x, y, w, h) in faces:
		print "(",x,",",y,") [",w,",",h,"]"
		face_gray = gray[y:y+h, x:x+w]
		mouth = mouth_cascade.detectMultiScale(face_gray,
						flags=cv2.cv.CV_HAAR_SCALE_IMAGE |
                        cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
		(mx, my, mw, mh) = mouth[0]
		print "(",mx,",",my,") [",mw,",",mh,"]"
		print "<",(my + y + mh / 2),",",(mx + x + mw / 2),">"
		frame = draw(frame, my + y + mh / 2, mx + x + mw / 2, w)

	cv2.imwrite(dst_fname, frame)

def test():
	process_frame(cv2.imread('frame_264.png'),'new.png')
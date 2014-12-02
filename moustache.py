import cv2
import math
import numpy

def draw(src, overlay, face_y, face_x):
	height, width = overlay.shape[:2]
	dst = src.copy()
	d_height, d_width = dst.shape[:2]
	
	#print "[",width,",",height,"]"
	
	for y in range (0, height):
		for x in range (0, width):
			dest_x = translate(x,width,face_x)
			dest_y = translate(y,height,face_y)
			
			if (dest_x >= 0) and (dest_x < d_width) and (dest_y >= 0) and (dest_y < d_height):
				if overlay[y][x].any() != 0:
					#print "(",x,",",y,") -> (",dest_x,",",dest_y,")"
					dst[dest_y][dest_x] = blend(dst[dest_y][dest_x], overlay[y][x])

	
	return dst

def blit_draw(src, overlay, face_y, face_x):
	height, width = overlay.shape[:2]
	d_height, d_width = src.shape[:2]
	
	x1a = translate(0,width,face_x)
	y1a = translate(0,height,face_y)
	x2 = min(x1a+width,d_width)
	y2 = min(y1a+height,d_height)
	x1 = max(x1a,0)
	y1 = max(y1a,0)
	
	ex1 = max(-x1a,0)
	ey1 = max(-y1a,0)
	ex2 = x2 - x1a
	ey2 = y2 - y1a
	
	zoom = src[y1:y2,x1:x2]
	overlay = overlay[ey1:ey2,ex1:ex2]
	
	height, width = overlay.shape[:2]
	
	#main = numpy.zeros((height,width,3), dtype=numpy.uint8)
	#alpha = numpy.zeros((height,width,), dtype=numpy.uint8)
	#cv2.mixChannels(overlay,(main,alpha),(0,0,1,1,2,2,3,3))
	b,g,r,a = cv2.split(overlay)
	main = cv2.merge((b,g,r))
	alpha = cv2.merge((a,a,a))
	beta = 255 - alpha
	
	main_b = cv2.multiply(main,alpha,dtype=cv2.CV_16U)
	main = cv2.convertScaleAbs(main_b,alpha=1.0/256)
	
	zoom_b = cv2.multiply(zoom,beta,dtype=cv2.CV_16U)
	zoom[:,:,:] = cv2.convertScaleAbs(zoom_b,alpha=1.0/256)[:,:,:]
	
	zoom += main
	
	#cv2.imwrite("test_output/debug.png", zoom)
	
	return src
	
def markov_add(old,next):
	return numpy.array(next)
	#return numpy.array(next) / 2 + numpy.array(old) / 2 
	
def translate(cord,scale,position):
	return position + cord - (scale / 2)

def blend(base, overlay):
	alpha = overlay[3] / 255.0
	return base * (1 - alpha) + overlay[:3] * alpha

def compute_scale(overlay, target):
	height, width = overlay.shape[:2]
	scale = 1.0 * target / width
	#print "scale: (",width,",",height,")[",target,"] --> ",scale
	return (int(width * scale),int(height * scale))

def scale_to(overlay, target):
	return cv2.resize(overlay,compute_scale(overlay,target),interpolation = cv2.INTER_AREA)

def process_frame(frame):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	#snose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	mustache = cv2.imread('CurlyMustache.png',-1)
	monocle = cv2.imread('monocle.png',-1)
	visor = cv2.imread('visor_b.png',-1)
	target = cv2.imread('target.png',-1)
	
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2,
                        minNeighbors=4,
                        flags=cv2.cv.CV_HAAR_SCALE_IMAGE |
                        cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
	for (x, y, w, h) in faces:
		#print "(",x,",",y,") [",w,",",h,"]"
		face_gray = gray[y:y+h, x:x+w]
		'''mouth = mouth_cascade.detectMultiScale(face_gray,
						flags=cv2.cv.CV_HAAR_SCALE_IMAGE |
                        cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
		(mx, my, mw, mh) = mouth[0]
		#print "(",mx,",",my,") [",mw,",",mh,"]"
		#print "<",(my + y + mh / 2),",",(mx + x + mw / 2),">"
		mustache = scale_to(mustache, w)
		frame = draw(frame, mustache, my + y + mh / 2, mx + x + mw / 2)'''
		'''eye = eye_cascade.detectMultiScale(face_gray,
						flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
		(ex, ey, ew, eh) = eye[1]
		print "(",ex,",",ey,") [",ew,",",eh,"]"
		print "<",(ey + y + eh / 2),",",(ex + x + ew / 2),">"
		monocle = scale_to(monocle, int(w/3.5))
		frame = draw(frame, monocle, ey + y + eh / 2, ex + x + ew / 2)'''
		eye = eye_cascade.detectMultiScale(face_gray,
						flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
		x1,y1 = box_center(*eye[0])
		x2,y2 = box_center(*eye[1])
		#print "(",x1,",",y1,") (",x2,",",y2,")"
		#print "<",((y1 + y2) / 2 + y),",",((x1 + x2) / 2 + x),">"
		visor = scale_to(visor,w)
		visor = align_slope(visor,x1,y1,x2,y2)
		frame = draw(frame,visor, (y1 + y2) / 2 + y, (x1 + x2) / 2 + x)

	return frame

def align_slope(overlay,x1,y1,x2,y2):
	height, width = overlay.shape[:2]
	#print "[",width,",",height,"]"
	theta = math.degrees(math.atan2((y1 - y2),abs(x1 - x2)))
	#print "theta: ",theta
	matrix = cv2.getRotationMatrix2D((height/2,width/2),-theta,1)
	img = cv2.warpAffine(overlay,matrix,(width, width))
	height, width = img.shape[:2]
	#print "[",width,",",height,"]"
	return img

def box_center(x,y,w,h):
	return (x + w / 2, y + h / 2)

def test():
	cv2.imwrite('test_output/new.png',process_frame(cv2.imread('frame_264.png')))
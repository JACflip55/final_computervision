import moustache
import tracking
import cv2
import Tkinter
import tkFileDialog

Tkinter.Tk().withdraw() # Close the root window
selected_video = tkFileDialog.askopenfilename(initialdir = 'test_data/', title = 'Select Video File')
#tracking.track_face(cv2.VideoCapture("test_data/face.mov"))
tracking.track_face(cv2.VideoCapture(selected_video))


#moustache.test();

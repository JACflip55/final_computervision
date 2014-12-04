import moustache
import tracking
import cv2
import Tkinter
import tkFileDialog
import ntpath

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

if __name__=='__main__':
    Tkinter.Tk().withdraw() # Close the root window
    selected_video = tkFileDialog.askopenfilename(initialdir = 'test_data/', title = 'Select Video File')
    #tracking.track_face(cv2.VideoCapture("test_data/face.mov"))
    filename = path_leaf(selected_video)
    tracking.track_face(cv2.VideoCapture(selected_video), filename)


#moustache.test();

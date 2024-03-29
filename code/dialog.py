import Tkinter, Tkconstants, tkFileDialog

fchosen_mustache = ""
fchosen_eyes = ""

class TkFileDialogExample(Tkinter.Frame):

    def __init__(self, root):

        Tkinter.Frame.__init__(self, root)
        # options for buttons
        button_opt = {'fill': Tkconstants.BOTH, 'padx': 3, 'pady': 3}

        # define buttons
        Tkinter.Button(self, text='Select a Mustache Overlay', command=self.set_mouth_piece).pack(**button_opt)
        Tkinter.Button(self, text='Select an Eye Overlay', command=self.set_eye_piece).pack(**button_opt)
        Tkinter.Button(self, text='Done', command=self.quit).pack(**button_opt)

    def set_mouth_piece(self):
        print "getting to mouth piece"
        global fchosen_mustache
        self.chosen_mustache = tkFileDialog.askopenfilename(initialdir = 'Assets/Mustaches/', title = 'Open Mustache Image')
        fchosen_mustache = self.chosen_mustache
        print fchosen_mustache

    def set_eye_piece(self):
        print "getting to eye piece"
        global fchosen_eyes
        self.chosen_eyes = tkFileDialog.askopenfilename(initialdir = 'Assets/Eyes/', title = 'Open Eye Image')
        fchosen_eyes = self.chosen_eyes
        print fchosen_eyes
    
def StartDialog():
    global fchosen_mustache
    global fchosen_eyes
    root = Tkinter.Tk()
    root.title("Select Assets")
    TkFileDialogExample(root).pack()
    root.mainloop()
    return (fchosen_mustache, fchosen_eyes)

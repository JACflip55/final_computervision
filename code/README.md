Simple desktop app to select a video and add a visor and mustache to a face


How it runs:
    In a terminal that has Python and OpenCV installed along with its attached libraries.
    To run the program type: "python test.py"
    
    
    This will bring up a File system window to let the user choose a video file from a directory. 
    In theory, you should be able to choose any video file, but it would be best if the video had a person in it
    and has only 1 person in it as well.
    
    After the video file is chosen, a empty window and a small box should appear. The empty window will be filled later
    The small box will let you click on a button to choose from the image files in the programs asset directory what mustache and eyewear 
    you wish to add to the face in the video. Both of these must be set for the program to run properly. Once selected press
    the "Done" button and it will continue to run. 
    
    As it runs you will see two windows pop up. One called "Overlayed Video" (OV) and the other called "Frame" (F)
    The OV window shows the video with the face overlaid with the choses images. The F window, is meant to add the 
    ability of object tracking. By first selecting that window then pressing "i", it will pause the video and allow
    you to click-out 4 points to surround an object with and are intended to be tracked. However it is very unlikely
    to do so accurately. 
    
    The video will finish playing and the overlaid video will be saved into a folder inside the programs directory.
    
    
Program Structure:
    The program may be put into any folder so long as it all stays together inside the code folder.
    
    The image assets to overlay images are kept in the asssets folder, but as part of the file system, the user
    may go to another directory for an image, but those in the assets are the ones garenteed to work right.
    
    After the program finishes executing, the overlaid video is saved in the Overlaid Videos folder under the same
    name that the video file originally had.
    

Things we imported:
    Tkinter - provided us with high level gui access to the file system and the ability to make the small button interface used in the program
    dialog - code borrowed from online and modified for our purposes, used in conjunction with Tkinter
    ntpath - separates the original file name from the path so it can be saved in the programs Overlaid Video folder after the program executes

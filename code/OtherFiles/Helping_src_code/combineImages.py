from operator import attrgetter
import math
import cv

def combineImages(*args):
 numImages = 0
 imageArray = []
 
 for img in args:
  if img != None and img.width > 0 and img.height > 0:
   imageArray.append(img)
   numImages += 1
 
 if numImages <= 0:
  return None

 # Find the largest x and y dimensions out of all the images
 # The resulting grid of images will all have this size, 
 # whether or not the images fit (though they won't be scaled).
 colWidth = max(imageArray, key=attrgetter('width')).width
 rowHeight = max(imageArray, key=attrgetter('height')).height
 
 # Square-root of the number of images will tell us how big the
 # sides of the square need to be. Ceiled to ensure they always
 # all fit.
 grid = int(math.ceil(math.sqrt(numImages)))
  
 combinedImage = cv.CreateImage((colWidth*grid, rowHeight*grid), 8, 3)
 
 cv.Set(combinedImage, cv.CV_RGB(50,50,50));
 
 for index, img in enumerate(imageArray):
  # Ensure all images are same type
  if img.nChannels == 1:
   colourImg = cv.CreateImage((img.width, img.height), 8,3)
   cv.CvtColor(img, colourImg, cv.CV_GRAY2RGB)
   img = colourImg
  
  # Which grid square are we up to?
  row = int(math.ceil(index / grid))
  column = index % grid
  
  cv.SetImageROI(combinedImage, (row*colWidth, column*rowHeight, img.width, img.height))
  cv.Copy(img, combinedImage)
  cv.ResetImageROI(combinedImage);
 
 return combinedImage

# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""
import cv2
#import matplotlib library
import matplotlib.pyplot as plt


# Get user supplied values
#imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image

image = cv2.imread('data/Wearing_Lipstick/000001.jpg',1)
r = 500.0 / image.shape[1]
dim = (500, int(image.shape[0] * r))
 
# perform the actual resizing of the image and show it
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    resized,
    scaleFactor=1.1,
    minNeighbors=5
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    print(x,y,w,h)
    cv2.rectangle(resized, (x-20, y-15), (x+w+10, y+h+10), (0, 255, 0), 2)
    
#image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)   
plt.imshow(image)    

#cv2.imshow("Faces found", image)
#cv2.waitKey(0)

#cropping face
# Convert the image to RGB colorspace
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
#gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
face_crop = []
for f in faces:
    x, y, w, h = [ v for v in f ]
  #  cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 3)
    # Define the region of interest in the image  
    face_crop.append(resized[y:y+h, x:x+w])

for face in face_crop:
    cv2.imwrite('test.jpg',face)
    # load the model 
    model = load_model('mode_mustache.h5')
    # load the image
    test_image = image.load_img('test.jpg', target_size = (32, 32))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    # make the prediction
    result = model.predict(test_image)
   
    if result[0][0] == 1:
        prediction = 'No M'
    else:
        prediction = 'M'
   # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    cv2.imshow('face',test_image)
    cv2.waitKey(0)


    # Write some Text

from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2

image = cv2.imread("test.jpg")

# Convert to PIL Image
cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_im = Image.fromarray(cv2_im_rgb)

draw = ImageDraw.Draw(pil_im)

# Choose a font
font = ImageFont.load_default().font

# Draw the text
draw.text((100, 200), "Your Text Here", font=font)

# Save the image
cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
cv2.imwrite("result.png", cv2_im_processed)
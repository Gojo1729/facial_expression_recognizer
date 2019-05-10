from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from datetime import datetime
import sys
import json
import collections
import json
import keras
import glob
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import model_from_json
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import threading
from multiprocessing.pool import ThreadPool
import win32com.client as wincl
from fastai.vision.image import open_image
from fastai.vision.learner import cnn_learner
from fastai.vision import models
from fastai.basic_train import load_learner
from fastai.vision import defaults
from fastai.vision.data import ImageDataBunch
from fastai.vision.transform import get_transforms
import torch
pool = ThreadPool(processes=1)
speak=wincl.Dispatch("SAPI.SpVoice")
torch.nn.Module.dump_patches = True



# folder where the captured images are stored.

di="C:\\Users\\shivu\\Anaconda2\\Desktop\\capture\\"
os.chdir("C:\\Users\\shivu\\Anaconda2\\Desktop\\capture\\")
result={"Afraid":0,"Angry":0,"Disgusted":0,"Happy":0,"Neutral":0,"Sad":0,"Surprised":0}
emotions=["Afraid","Angry","Disgusted","Happy","Neutral","Sad","Surprised"]
#keras model
'''un comment this if you want to use keras model, currently using fast.ai model
json_file = open('C:\\Users\\shivu\\bopro\\deeplearning\\fer\\aug_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_json = model_from_json(loaded_model_json)
adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#weights of keras model
loaded_model_json.load_weights("C:\\Users\\shivu\\bopro\\deeplearning\\fer\\aug_model.h5")
loaded_model_json.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

#clearing the folder before capturing video frames.
for file in os.listdir(di):
    if os.path.isfile(di+file):
        os.unlink(di+file)
def  predict(files):
	result={"afraid":0,"angry":0,"disgusted":0,"happy":0,"neutral":0,"sad":0,"surprised":0}
	for file in files:
		res=learn.predict(open_image(file,convert_mode="L"))
		
		result[str(res[0])]+=1
	v=list(result.values())
	s=np.sum(v)
	per=(v/s)*100
	print("percentage",per)
	return result
'''

#replace the path which I have provided and provide absolute path to the files
prototxt='C:\\Users\\shivu\\bopro\\deeplearning\\fer\\deploy.prototxt.txt'
model="C:\\Users\\shivu\\bopro\\deeplearning\\fer\\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt,model)
##net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
##net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# initialize the video stream and allow the cammera sensor to warmup
#print("[INFO] starting video stream...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
speak.Speak("Starting Video Capturing")
count=0

##--use this function for keras model
##def predictFile(file):
##    if os.path.isfile(file):
##        try:
##            img=image.load_img(file,color_mode="grayscale",target_size=(64,64))
##            img_tensor = image.img_to_array(img)
##            img_tensor=img_tensor/255
##            img_tensor = img_tensor.reshape((1,)+img_tensor.shape)
##            res=loaded_model_json.predict(img_tensor)
##            index=np.argmax(res)
##            return emotions[index]
##        except:
##            pass
##

#fasr ai model, folder containing export.pkl file
path='C:\\Users\\shivu\\bopro\\deeplearning\\fer\\resnet34-wfinetune\\'
defaults.device=torch.device("cpu")
learn=load_learner(path)

#for .pth file
##path=path+"resnetfinetrainedagain"
##data=ImageDataBunch.single_from_classes(path,emotions,ds_tfms=get_transforms(),size=224).normalize(([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
##learn=cnn_learner(data,models.resnet50)
##learn.load(path)

def predictFile(file):
    if os.path.isfile(file):
        try:
            img=open_image(file,convert_mode="L")
            r=learn.predict(img)
            res=str(r[0]).capitalize(),str(float(torch.max(r[2])*100))
            return res 
        except:
            pass
        


# loop over the frames from the video stream
while True:
    
   # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
 
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
 
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.9:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        

 
        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        file=frame[startY:endY,startX:endX]
        
        
##        cv2.putText(frame, text, (startX, y),
##            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255, 255))
        
        file_name=di+str(datetime.now().microsecond)+str(count)+".jpg"
        cv2.imwrite(file_name,file)
        #async_result = pool.apply_async(predictFile,(file_name,))
        #img_res=async_result.get()
        start = time.time()
        img_res=predictFile(file_name)
        end = time.time()
        print("elapsed time",end-start)
        #img_res=predictFile(file_name)
        if img_res is not None:
            result[img_res[0]]+=1
            
            if img_res[0] == "Afraid":
                color=(180,105,255)
            elif img_res[0] == "Angry":
                color=(0,0,255)
            elif img_res[0] == "Disgusted":
                color=(0,128,0)
            elif img_res[0] == "Happy":
                color=(0,255,255)
            elif img_res[0] == "Neutral":
                color=(255,255,255)
            elif img_res[0] == "Sad":
                color=(255,0,0)
            elif img_res[0] == "Surprised":
                color=(0,165,255) 
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                color,2)
            cv2.putText(frame,img_res[0]+" "+img_res[1][:5]+"%", (startX, y),
               cv2.FONT_HERSHEY_TRIPLEX,0.82,color)
        
      


    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(3) & 0xFF
    fps.update()
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
                WebcamVideoStream(src=0).stop()
                break
    
    
fps.stop()
        
cv2.destroyAllWindows()
vs.stop()



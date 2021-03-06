#sleeping packages
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
#mouth packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys
#sound packages
import pyaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import keras

import yamnet.params as params
import yamnet.yamnet as yamnet_model
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet/yamnet.h5')
yamnet_classes = yamnet_model.class_names('yamnet/yamnet_class_map.csv')
#facial packages
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
#stress and eye brows
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model
#eye gazing package
from gaze_tracking import GazeTracking

face_classifier = cv2.CascadeClassifier('/Users/Admin/Documents/GitHub/emotion_detection/haarcascade_frontalface_default.xml')
classifier =load_model('/Users/Admin/Documents/GitHub/emotion_detection/Emotion_little_vgg.h5')
class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)


frame_len = int(params.SAMPLE_RATE * 1) # 1sec

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=params.SAMPLE_RATE,
                input=True,
                frames_per_buffer=frame_len)

cnt = 0
plt.ion()
def eye_brow_distance(leye,reye):
    global points
    distq = dist.euclidean(leye,reye)
    points.append(int(distq))
    return distq

def emotion_finder(faces,frame):
    global emotion_classifier
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    if label in ['scared','sad']:
        label = 'stressed'
    else:
        label = 'not stressed'
    return label    

def normalize_values(points,disp):
    normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    stress_value = np.exp(-(normalized_value))
    # print(stress_value)
    if stress_value>=75:
        return stress_value,"High Stress"
    else:
        return stress_value,"low_stress"
#mouth
def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
	help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())

gaze = GazeTracking()

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.72
# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")

#mouth


#start web cam
print("[INFO] starting video stream thread...")
webcam = VideoStream(src=0).start()
# vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
detector = dlib.get_frontal_face_detector()
landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")#predictor
emotion_classifier = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
points = []
face_recognition = dlib.get_frontal_face_detector()#detector
time.sleep(1.0)
frame_width = 640
frame_height = 360
sleep_count = 0
max_sleep_count = 30
normal = False
normal_count = 0.0
normal_eye_ratio = 0
eyeflag=False
headflag=False

def eye_ratio(eye):
    avg_height = (abs(eye[1][1]-eye[5][1])+abs(eye[2][1]-eye[4][1]))/2
    width = abs(eye[0][0]-eye[3][0])

    return avg_height/width
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
time.sleep(1.0)
while True:
    #sound part
    # #get the image corresponding to a frame
    
    # data = stream.read(frame_len, exception_on_overflow=False)

    # # # byte --> float
    # frame_data = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)

    # # # model prediction
    # scores, melspec = yamnet.predict(np.reshape(frame_data, [1, -1]), steps=1)
    # prediction = np.mean(scores, axis=0)

    # # # visualize input audio
    # # # plt.imshow(melspec.T, cmap='jet', aspect='auto', origin='lower')
    # # # plt.pause(0.001)
    # # # plt.show()

    # top5_i = np.argsort(prediction)[::-1][:1]

    # # # print result
    # print('Current event:\n' +
    #     '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
    #             for i in top5_i))

    # # # print idx
    # print(cnt)
    # cnt += 1
    #end of sound part
    frame = webcam.read()
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=450)
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector(img,0)
    labels = []
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = ""
    elif gaze.is_right():
        text = ""
    elif gaze.is_left():
        text = ""
    elif gaze.is_center():
        text = "4-READING!!"
    cv2.putText(frame, text, (30, 120), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,255), 2)    
    faces = face_recognition(img, 0)
    facee = face_cascade.detectMultiScale(frame, 1.3, 5)
    faceee = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags=cv2.CV_HAAR_SCALE_IMAGE
    )
    if(not(normal) and normal_count<47):
        cv2.putText(frame, "FOCUS YOUR NORMAL EYES", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 255), 2)    
    for detection in detections:
        emotion = emotion_finder(detection,img)
        cv2.putText(frame, emotion, (170,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        shape = landmarks(frame,detection)#landmarks
        shape = face_utils.shape_to_np(shape)
           
        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]
            
        reyebrowhull = cv2.convexHull(reyebrow)
        leyebrowhull = cv2.convexHull(leyebrow)

        cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)

        distq = eye_brow_distance(leyebrow[-1],reyebrow[0])
        stress_value,stress_label = normalize_values(points,distq)
        cv2.putText(frame,"stress level:{}".format(str(int(stress_value*100))),(150,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for face in faces:
        #get the landmark data for the face as numpy array
        face_data = face_utils.shape_to_np(landmarks(img,face))
        #left eye positions are from 36th index to 41st index
        #right eye positions are from 42th index to 47st index
        
        # extract the mouth coordinates, then use the
		# coordinates to compute the mouth aspect ratio 
        mouth = face_data[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
    # compute the convex hull for the mouth, then
    # visualize the mouth
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #get eye data and show in the frame 
        left_eye = face_data[36:42]
        right_eye = face_data[42:48]
    
        leftEyeHull = cv2.convexHull(left_eye)
        rightEyeHull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    
        eye_avg_ratio = eye_ratio(left_eye)+eye_ratio(right_eye)/2.0
    # Draw text if mouth is open
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "2-TALKING!!", (30,85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            
        #print(eye_avg_ratio)
        if(not(normal)):
            if(normal_count<50):
               normal_eye_ratio = normal_eye_ratio+eye_avg_ratio
            else:
                normal_eye_ratio = normal_eye_ratio/normal_count
                normal = True
                cv2.putText(frame, "LETS START!", (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 255), 3)
                #print(normal_eye_ratio)
                
            normal_count=normal_count+1
            
        else:
            #print(normal_eye_ratio-eye_avg_ratio)
            if(normal_eye_ratio-eye_avg_ratio>0.05):
                sleep_count = sleep_count+1
                if(sleep_count>max_sleep_count):
                    cv2.putText(frame, "3-SLEEPING!!!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # print("Sleeping")
            else:
                # print("awake")
                sleep_count = 0
                #eye
                for (x,y,w,h) in facee:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = img[y:y+h, x:x+w]
                    # roi_color = frame[y:y+h, x:x+w]
                    roi_grayf = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                    cv2.putText(frame, "1-IS HERE", (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                    
                    if np.sum([roi_grayf])!=0:
                        roi = roi_grayf.astype('float')/255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi,axis=0)
                        
        # make a prediction on the ROI, then lookup the class

                        preds = classifier.predict(roi)[0]
                        label=class_labels[preds.argmax()]
                        label_position = (x,y)
                        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                    else:
                        cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
                    # eyes = eye_cascade.detectMultiScale(roi_gray)
                    # for (ex,ey,ew,eh) in eyes:
                    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    #     eyeflag=True
                    #     #end of eye
                    #     #head
                    # eyeflag=False 

                # for (x, y, w, h) in faceee:
                #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
                #     cv2.putText(frame, "Student is sitting", (30,85),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                #     headflag=True
                    #end of head
        if(headflag==True and eyeflag==False):
            cv2.putText(frame, "Student is writing", (30,120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    # Write the frame into the file 'output.avi'
    out.write(frame)    
    #show web cam frame 
    cv2.imshow("Frame", frame)
    if(normal_count==51):
        cv2.waitKey(1000)
        normal_count = 0
    else:
        wait = cv2.waitKey(1)
        if wait==ord("q"):
            cv2.destroyAllWindows()
            webcam.stop()
            break





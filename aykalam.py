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
landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")#predictor
face_recognition = dlib.get_frontal_face_detector()#detector
time.sleep(1.0)
frame_width = 640
frame_height = 360
sleep_count = 0
max_sleep_count = 30
normal = False
normal_count = 0.0
normal_eye_ratio = 0

def eye_ratio(eye):
    avg_height = (abs(eye[1][1]-eye[5][1])+abs(eye[2][1]-eye[4][1]))/2
    width = abs(eye[0][0]-eye[3][0])

    return avg_height/width
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
time.sleep(1.0)
while True:
    #get the image corresponding to a frame
    frame = webcam.read()
    frame = imutils.resize(frame, width=450)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_recognition(img, 0)
    if(not(normal) and normal_count<47):
        cv2.putText(frame, "FOCUS YOUR NORMAL EYES", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 255), 2)    

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
            cv2.putText(frame, "Mouth is Open!", (30,60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            print('mouth is open')
        else:
            print('mouth is closed')
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
                    cv2.putText(frame, "SLEEPING!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 255), 2)
                    print("Sleeping")
            else:
                print("awake")
                sleep_count = 0
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
#---------------------------------------------END OF SLEEPING CODE-------------------------------------------------------
#---------------------------------------------START OF MOUTH CODE--------------------------------------------------------
# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages














# loop over frames from the video stream

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	
	
	

	# detect faces in the grayscale frame


	# loop over the face detections
	
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		

		
		
        
	
	# show the frame
	
	# key = cv2.waitKey(1) & 0xFF

	# # if the `q` key was pressed, break from the loop
	# if key == ord("q"):
	# 	break


#--------------------------------------------END OF MOUTH CODE-----------------------------------------------------------
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import serial
import math

mixer.init()
sound = mixer.Sound('alarm.wav')
arduino=serial.Serial('COM3',9600)
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    image=frame
    height,width = frame.shape[:2] 
    cv2.rectangle(image,(100,100),(300,300),(0,255,0),0)
    crop_image = image[100:300, 100:300]
    arduino.write(str.encode("0"))

    blur = cv2.GaussianBlur(crop_image, (3,3), 0)


    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask2 = cv2.inRange(hsv, np.array([2,0,0]), np.array([20,255,255]))


    kernel = np.ones((5,5))
    
    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)    
       
    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3,3), 0)
    ret,thresh = cv2.threshold(filtered, 127, 255, 0)
    
    # Show threshold image
    cv2.imshow("Thresholded", thresh)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
    
    try:
        # Find contour with maximum area
        contour = max(contours, key = lambda x: cv2.contourArea(x))
        
        # Create bounding rectangle around the contour
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)
        
        # Find convex hull
        hull = cv2.convexHull(contour)
        
        # Draw contour
        drawing = np.zeros(crop_image.shape,np.uint8)
        cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
        cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
        
        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour,hull)
        
        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger 
        # tips) for all defects
        count_defects = 0
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
            
            # if angle > 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image,far,1,[0,0,255],-1)

            cv2.line(crop_image,start,end,[0,255,0],2)
        
        # Print number of fingers
        if count_defects == 0:
            cv2.putText(frame,"Acceleration", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,255), 2)
            arduino.write(str.encode("1"))
        elif count_defects == 1:
            cv2.putText(frame,"Brake", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            arduino.write(str.encode("1"))
        elif count_defects == 2:
            cv2.putText(frame, "Brake", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            arduino.write(str.encode("1"))
        elif count_defects == 3:
            cv2.putText(frame,"Brake", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            arduino.write(str.encode("0"))
        elif count_defects == 4:
            cv2.putText(frame,"Brake", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            arduino.write(str.encode("0"))
        elif count_defects == 5:
            cv2.putText(frame,"Brake", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            arduino.write(str.encode("0"))
        elif count_defects == 6:
            cv2.putText(frame,"Brake", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            arduino.write(str.encode("0"))
        else:
            pass
    except:
        pass
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    #arduino.write(str.encode("1"))
    print("Driving")
    #cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(180,height-20), font, 1,(50,255,0),1,cv2.LINE_AA)
        
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-2
        cv2.putText(frame,"Open",(180,height-20), font, 1,(0,255,0),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Timer::'+str(score),(300,height-20), font, 1,(255,0,0),1,cv2.LINE_AA)
        
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        #count_defects=5
        f=15
        for i in range(score):
            arduino.write(str.encode("0"))
        try:
             cv2.putText(frame,'ALERT!!!!',(200,height-100), font, 1,(0,0,255),1,cv2.LINE_AA)
             #count_defects=5
             arduino.write(str.encode("0"))
             sound.play()
             
        except:  # isplaying = False
            #pass
            cv2.putText(frame,'',(200,height-100), font, 1,(0,0,255),1,cv2.LINE_AA)
            
    #arduino.write(str.encode("1"))
        if(thicc<16):
            thicc= thicc+1
        else:
            thicc=thicc-1
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
    cv2.imshow('frame',frame)
    cv2.imshow("Gesture", frame)
    drawing = np.zeros(crop_image.shape,np.uint8)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

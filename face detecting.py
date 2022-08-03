import cv2
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeDetect = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
smileDetect = cv2.CascadeClassifier("haarcascade_smile.xml")
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detections = faceDetect.detectMultiScale(frame,1.1,4)
    for (x,y,w,h) in detections:
       # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        eye_gray = gray[y:y+h,x:x+w]
        eye_color = frame[y:y+h,x:x+w]
        eyeDetections =eyeDetect.detectMultiScale(eye_gray,1.1,4)
        for (ex,ey,ew,eh) in eyeDetections:
            hw = ew//2
            hh = eh//2
            cv2.circle(eye_color, (ex+hw, ey+hh), hh, (0, 0,0),3)
        smile_gray = gray[y:y+h,x:x+w]
        smile_color = frame[y:y+h,x:x+w]
        smileDetections = smileDetect.detectMultiScale(smile_gray,1.4,10)
        for(sx,sy,sw,sh) in smileDetections:
            cv2.rectangle(smile_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 5)
    cv2.imshow("my frame",frame)
    if cv2.waitKey(1)== 27:
        break
cap.release()
cv2.destroyAllWindows()


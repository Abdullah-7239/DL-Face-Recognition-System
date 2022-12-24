
import cv2
import numpy as np 
import face_recognition
import os
from datetime import datetime

teacher = input("Enter teacher name ")
path = 'images'
path2 = 'teacher images'
teacherName =[]
teacher_images = []
teacherPresent = []
images=[]
personName = []
myList = os.listdir(path)
myList2 = os.listdir(path2)
print(myList)
print(myList2)
x=1
for cu_img in myList :
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personName.append(os.path.splitext(cu_img)[0])


for cu_img in myList2 :
    current_Img = cv2.imread(f'{path2}/{cu_img}')
    teacher_images.append(current_Img)
    teacherName.append(os.path.splitext(cu_img)[0])


print(personName)
print(teacherName)

def faceEncodings(images) :
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open("attendance.csv",'r+') as f:
        myDataList= f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList :
            time_now = datetime.now()
            tStr = time_now.strftime("%H;%M;%S")
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tStr},{dStr}')

encodeListKnown = faceEncodings(images)
encodeTeacherListKnown = faceEncodings(teacher_images)
print("All encodings complete")

'''implmenting camera module'''

cap = cv2.VideoCapture(0)
while(True):
    ret, frame1 = cap.read()
    teacherfaces = cv2.resize(frame1, (0,0), None, 0.25, 0.25)
    teacherfaces = cv2.cvtColor(teacherfaces , cv2.COLOR_BGR2RGB)
    TeacherCurrentFrame = face_recognition.face_locations(teacherfaces)
    TeacherEncodeFrame = face_recognition.face_encodings(teacherfaces,TeacherCurrentFrame)

    for encodeteacherface, teacherloc in zip(TeacherEncodeFrame, TeacherCurrentFrame):
        match = face_recognition.compare_faces(encodeTeacherListKnown, encodeteacherface)
        faceDis = face_recognition.face_distance(encodeTeacherListKnown, encodeteacherface)

        matchIndex = int(np.average(faceDis))

        if match[matchIndex]:
            name = teacherName[matchIndex]
            y1,x2,y2,x1 = teacherloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame1, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.rectangle(frame1, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame1, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
            teacherPresent.append(name)
    cv2.imshow("Camera", frame1)
    if cv2.waitKey(1) == 13:
        break

if (teacher in teacherPresent) :
    while(True):
        ret, frame  = cap.read()
        faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        faces = cv2.cvtColor(faces , cv2.COLOR_BGR2RGB)
        facesCurrentFrame  = face_recognition.face_locations(faces)
        encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)


        for encodeFace, faceLoc in zip(encodesCurrentFrame,facesCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = int(np.average(faceDis))

            if matches[matchIndex]:
                name = personName[matchIndex].upper()
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
                cv2.rectangle(frame, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
                attendance(name)

        cv2.imshow("Camera",frame)
        if cv2.waitKey(1) == 13 :
            break
        
cap.release()
cv2.destroyAllWindows()
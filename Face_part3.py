# predict with model trained
import cv2
import datetime
import sys
import csv

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
studentToID = {}
# allId = set()
freq = {}

for i in range(1, len(sys.argv), 2):
    studentToID[sys.argv[i]] = sys.argv[i + 1]
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        name = "Unknown"
        # allId.add(Id)
        if Id is not None and not Id in freq:
            freq[Id] = 0
        freq[Id] += 1

        if str(Id) in studentToID:  # found student
            name = studentToID[str(Id)]

            # save data student
            day_now = datetime.date.today()
            DATA_FILE = "data_student/Student_Day_" + str(day_now) + ".csv"

            with open(DATA_FILE, 'a+') as csv_file:
                fieldnames = [name, 'Da diem danh']  # Định dạng cột
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

        cv2.putText(im, str(name), (x, y + h), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('im', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

print(max(freq, key=freq.get))  # tra ra id cua nguoi diem danh

cam.release()
cv2.destroyAllWindows()

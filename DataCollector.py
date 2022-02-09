from fileinput import filename
import cv2

#video = cv2.VideoCapture(2)
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

while True:
    ret, frame = video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        count = count + 1
        name = './Images/0/' + str(count) + '.jpg'
        # name = './Images/1/' + str(count) + '.jpg'
        print('Collecting Data.......!' + name)
        cv2.imwrite(name, frame[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow('Data Collector', frame)
    k = cv2.waitKey(1)

    if count >= 500:
        break

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
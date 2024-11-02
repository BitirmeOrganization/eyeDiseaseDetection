
from ultralytics import YOLO
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
kamera = cv2.VideoCapture(1)
confidence = 0.4

model = YOLO("models/4haziran.pt")
labels = ['drusen', 'normal']

while True:
    ret, frame = kamera.read()
    imgs = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(imgs, verbose = False)

    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i]
        score = results[0].boxes.conf[i]
        label = results[0].boxes.cls[i]
        x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
        name = labels[label]
        if score < confidence:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = name + ' ' + str(format(score, '.2f'))
        cv2.putText(frame, text, (x1, y1 - 10), font, 1.2, (255, 0, 255), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
kamera.release()
cv2.destroyAllWindows()
exit(0)

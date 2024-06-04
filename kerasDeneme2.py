import cv2
import numpy as np
import tensorflow as tf
from keras.src.layers import DepthwiseConv2D, BatchNormalization
from tensorflow.python.keras.models import load_model


# Modelinizi yüklerken custom_objects parametresini kullanın
custom_objects = {'DepthwiseConv2D': DepthwiseConv2D, 'BatchNormalization': BatchNormalization}
model = load_model('keras_model.h5', custom_objects=custom_objects)

# Kategori isimleri (modelinizin çıktısına göre ayarlayın)
class_names = ['DME', 'NORMAL']  # Burayı kendi modelinizin sınıflarıyla doldurun

def preprocess_frame(frame):
    # Modelinizin giriş boyutlarına göre çerçeveyi yeniden boyutlandırın
    img = cv2.resize(frame, (model.input_shape[1], model.input_shape[2]))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def draw_predictions(frame, predictions):
    for pred in predictions:
        x, y, w, h, class_id, confidence = pred
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def detect_objects(frame):
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)

    # Bu kısım, modelinize ve çıktısına göre değişir. Örneğin, YOLO gibi bir model kullanıyorsanız, özel bir post-processing gerekebilir.
    # predictions'ı (x, y, w, h, class_id, confidence) formatına dönüştürün

    return predictions

# Kamera akışını başlatın
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    predictions = detect_objects(frame)
    frame = draw_predictions(frame, predictions)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Gerekli kütüphanelerin yüklenmesi
# !pip install ultralytics
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Modelin yüklenmesi
model = YOLO('classification.pt')  # Yolov8 sınıflandırma modeli

# Görüntü verisinin yüklenmesi
image_path = 'C:/Users/Eren/Desktop/eyeDiseasesClassification/dataset/normal/1253_right.jpg'  # Görüntü dosya yolunu buraya ekleyin
image = Image.open(image_path)

# Tahmin yapma
results = model(image_path)

# Sonuçların yazdırılması ve görselleştirme
# YoloV8'de sonuçlar genellikle bir liste şeklinde döner
print(results)

# Tahmin edilen sınıf etiketlerini ve olasılıkları işleme
for result in results:
    label = result.names[result.top1]  # En yüksek olasılığa sahip sınıf etiketi
    confidence = result.top1conf.item()  # En yüksek olasılık değeri
    print(f'Label: {label}, Confidence: {confidence:.2f}')

# Görüntüyü ve tahmin edilen etiketleri görselleştirme
plt.imshow(image)
plt.title(f'Predicted: {label}, Confidence: {confidence:.2f}')
plt.show()

import cv2
from ultralytics import YOLO
import numpy as np
import tkinter as tk
from tkinter import filedialog, font, Toplevel
from ctypes import windll
from PIL import Image, ImageTk

windll.shcore.SetProcessDpiAwareness(1)


class MainInterface:
    def __init__(self, root):
        self.root = root

        # Properties of interface
        self.root.title("Göz Hastalıklarının Yapay Zeka ile Otomatik Tespiti")
        self.root.geometry("600x800")

        self.authorFont = tk.font.Font(root, family="Helvetica", size=12, weight="bold")
        self.authorText = tk.Label(self.root,
                                   text="Göz Hastalıklarının Yapay Zeka ile Otomatik Tespiti\nDr. Öğr. Üyesi Burak Yılmaz\nMehmet Ali Güven - 211229014\nEren GÜNER - 211229049",
                                   font=self.authorFont)
        self.authorText.pack(padx=10, pady=10)

        self.imgLabel = tk.Label(root)
        self.imgLabel.pack()

        self.uploadImageBtn = tk.Button(root, text="Görsel Yükle", command=self.upload_image)
        self.uploadImageBtn.pack(padx=10, pady=20)

        self.detecBtn = tk.Button(root, text="Hastalık Tespiti Yap", command=self.detect_disease)
        self.detecBtn.pack(padx=10, pady=20)

        self.image = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Resim Dosyalari", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = Image.open(file_path)
            self.display_image()

    def display_image(self):
        if self.image is not None:
            img = ImageTk.PhotoImage(self.image)
            self.imgLabel.config(image=img)
            self.imgLabel.image = img

    def detect_disease(self):
        if self.image is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            confidence = 0.4
            model = YOLO("4haziran.pt")
            labels = ["drusen", "normal"]
            arraySrc = np.array(self.image)
            srcH, srcW = arraySrc.shape[:2]
            srcRGB = cv2.cvtColor(arraySrc, cv2.COLOR_BGR2RGB)
            results = model(srcRGB, verbose=False)

            for i in range(len(results[0].boxes)):
                x1, y1, x2, y2 = results[0].boxes.xyxy[i]
                score = results[0].boxes.conf[i]
                label = results[0].boxes.cls[i]
                x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
                name = labels[label]
                if score < confidence:
                    continue

                cv2.rectangle(srcRGB, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = name + ' ' + str(format(score, '.2f'))
                cv2.putText(srcRGB, text, (x1, y1 - 10), font, 1.2, (255, 0, 255), 2)

            # Görüntü işleme işlemleri tamamlandıktan sonra sonucu göster
            self.result = Image.fromarray(srcRGB)
            img_tk = ImageTk.PhotoImage(self.result)

            new_window = Toplevel(self.root)
            new_window.title("Sonuç")

            label = tk.Label(new_window, image=img_tk)
            label.image = img_tk  # Görüntü referansını sakla
            label.pack()


root = tk.Tk()
MainInterface(root)
root.mainloop()

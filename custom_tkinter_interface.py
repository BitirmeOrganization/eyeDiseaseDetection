import tkinter as tk
import customtkinter as ctk

import cv2
from ultralytics import YOLO
import numpy as np
from tkinter import filedialog, font, Toplevel
from ctypes import windll
from PIL import Image
from customtkinter import CTkImage, CTkToplevel

windll.shcore.SetProcessDpiAwareness(1)

class MainInterface:
    def __init__(self, root):
        self.root = root

        # Properties of interface
        self.root.title("Göz Hastalıklarının Yapay Zeka ile Otomatik Tespiti")
        self.root.geometry("600x800")

        self.authorText = ctk.CTkLabel(self.root,
                                       text="Göz Hastalıklarının Yapay Zeka ile Otomatik Tespiti\n\nDr. Öğr. Üyesi Burak Yılmaz\nMehmet Ali Güven - 211229014\nEren Güner - 211229049")
        self.authorText.pack(padx=10, pady=10)

        self.imgLabel = ctk.CTkLabel(root, text="")
        self.imgLabel.pack()

        self.uploadImageBtn = ctk.CTkButton(root, text="Görsel Yükle", command=self.upload_image)
        self.uploadImageBtn.pack(padx=10, pady=20)

        self.detecBtn = ctk.CTkButton(root, text="Hastalık Tespiti Yap", command=self.detect_disease)
        self.detecBtn.pack(padx=10, pady=20)

        self.image = None
        self.warning_label = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Resim Dosyalari", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = Image.open(file_path)
            self.display_image()

            if self.warning_label:
                self.warning_label.destroy()
                self.warning_label = None

    def display_image(self):
        if self.image is not None:
            img = CTkImage(self.image, size=(500, 500))  # Görüntüyü ölçeklendirerek CTkImage oluşturun
            self.imgLabel.configure(image=img)
            self.imgLabel.image = img  # Görüntü referansını saklayın

    def detect_disease(self):
        if self.image is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            confidence = 0.4
            model = YOLO("4haziran.pt")
            labels = ["drusen", "normal"]
            arraySrc = np.array(self.image)
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

            self.result = Image.fromarray(srcRGB)
            img_tk = CTkImage(self.result, size=(500, 500))  # Sonuç görüntüsünü ölçekleyerek CTkImage oluşturun

            new_window = CTkToplevel(self.root)
            new_window.title("Sonuç")

            label = ctk.CTkLabel(new_window, image=img_tk, text="")
            label.image = img_tk  # Görüntü referansını saklayın
            label.pack()

        else:
            if not self.warning_label:  # warning_label daha önce oluşturulmadıysa
                self.warning_label = ctk.CTkLabel(self.root, text="Hastalık tespiti yapabilmek için görsel yükleyin!",
                                                  fg_color="red")
                self.warning_label.pack(pady=10)


root = ctk.CTk()
MainInterface(root)
root.mainloop()
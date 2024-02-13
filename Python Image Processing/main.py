import sys
import face_recognition
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QImageReader
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from matplotlib import patches, pyplot as plt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Yapay Zeka Görsel İşlemler Uygulaması")
        self.setGeometry(100, 100, 1200, 600)

        main_layout = QHBoxLayout()

        button_widget = QWidget()

        self.central_widget = QLabel(self)
        self.central_widget.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-weight: bold; color: green;")

        self.add_image_button = QPushButton("Resim Ekle", self)
        self.add_image_button.clicked.connect(self.load_image)

        self.detect_face_button = QPushButton("Yüzleri Tanı", self)
        self.detect_face_button.clicked.connect(self.detect_faces)

        self.process_button = QPushButton("Watershap", self)
        self.process_button.clicked.connect(self.watershap)

        self.extra_button = QPushButton("Yüz Bulanıklaştıma", self)
        self.extra_button.clicked.connect(self.Bulanik)

        self.harris_button = QPushButton("Harris Köşe Tespiti", self)
        self.harris_button.clicked.connect(self.harris)

        self.custom_button_1 = QPushButton("Resim Dışına Kenarlık Ekleme", self)
        self.custom_button_1.clicked.connect(self.Kenar)

        self.custom_button_2 = QPushButton("Gamma", self)
        self.custom_button_2.clicked.connect(self.gamma)

        self.custom_button_3 = QPushButton("Siyah Beyaz", self)
        self.custom_button_3.clicked.connect(self.blackwhite)

        self.custom_button_4 = QPushButton("Histogram", self)
        self.custom_button_4.clicked.connect(self.histogram)

        self.custom_button_5 = QPushButton("Otsu Eşitleme", self)
        self.custom_button_5.clicked.connect(self.otsu)

        self.custom_button_6 = QPushButton("Adaptive Threshold", self)
        self.custom_button_6.clicked.connect(self.adaptivethres)

        self.custom_button_7 = QPushButton("Contours", self)
        self.custom_button_7.clicked.connect(self.contours)

        self.custom_button_8 = QPushButton("Sobel", self)
        self.custom_button_8.clicked.connect(self.sobel)

        self.detect_face_webcam_button = QPushButton("Webcam ile Yüzleri Tanı", self)
        self.detect_face_webcam_button.clicked.connect(self.detect_faces_webcam)

        button_layout = QVBoxLayout(button_widget)
        button_layout.addWidget(self.add_image_button)
        button_layout.addWidget(self.detect_face_button)
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.extra_button)
        button_layout.addWidget(self.harris_button)
        button_layout.addWidget(self.custom_button_1)
        button_layout.addWidget(self.custom_button_2)
        button_layout.addWidget(self.custom_button_3)
        button_layout.addWidget(self.custom_button_4)
        button_layout.addWidget(self.custom_button_5)
        button_layout.addWidget(self.custom_button_6)
        button_layout.addWidget(self.custom_button_7)
        button_layout.addWidget(self.custom_button_8)
        button_layout.addWidget(self.detect_face_webcam_button)




        main_layout.addWidget(button_widget)
        main_layout.addWidget(self.central_widget)
        main_layout.addWidget(self.result_label)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.image_path = None
        self.image_pixmap = None
        self.face_locations = []

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        file_dialog.setOptions(options)

        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            self.image_path = selected_file
            self.image_pixmap = self.get_scaled_pixmap(self.image_path)
            self.central_widget.setPixmap(self.image_pixmap)

    def detect_faces(self):
        if self.image_path is None:
            self.result_label.setText("Lütfen önce bir resim ekleyin.")
            return

        resim = face_recognition.load_image_file(self.image_path)
        self.face_locations = face_recognition.face_locations(resim)

        fig, ax = plt.subplots(figsize=(self.image_pixmap.width() / 100, self.image_pixmap.height() / 100))
        ax.imshow(resim)

        for resim_yeri in self.face_locations:
            ust, sag, alt, sol = resim_yeri
            kare = patches.Rectangle((sol, ust), (sag - sol), (alt - ust), linewidth=3, edgecolor='y', facecolor='none')
            ax.add_patch(kare)

        plt.axis('off')
        plt.tight_layout()

        plt.savefig("result_image.png", bbox_inches="tight", pad_inches=0, transparent=True)
        result_pixmap = self.get_scaled_pixmap("result_image.png")
        self.result_label.setPixmap(result_pixmap)

    def watershap(self):
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(thresh, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg, labels=5)

        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(img, markers)
        img[markers == -1] = [0, 0, 255]

        result_image_path = "result_image.png"
        cv2.imwrite(result_image_path, img)

        result_pixmap = self.get_scaled_pixmap(result_image_path)
        self.result_label.setPixmap(result_pixmap)

    def Bulanik(self):
        if self.image_path is None:
            self.result_label.setText("Lütfen önce bir resim ekleyin.")
            return

        original_image = cv2.imread(self.image_path)
        blurred_image = original_image.copy()

        for face_location in self.face_locations:
            top, right, bottom, left = face_location
            face = original_image[top:bottom, left:right]
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            blurred_image[top:bottom, left:right] = blurred_face

        result_image_path = "blurred_result_image.png"
        cv2.imwrite(result_image_path, blurred_image)

        result_pixmap = self.get_scaled_pixmap(result_image_path)
        self.result_label.setPixmap(result_pixmap)

    def harris(self):
        if self.image_path is None:
            self.result_label.setText("Lütfen önce bir resim ekleyin.")
            return

        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Harris köşe tespiti parametreleri
        block_size = 2
        corner_quality = 0.04
        min_distance = 10

        # Harris köşe tespiti uygula
        corners = cv2.cornerHarris(gray, block_size, 3, corner_quality)

        # Köşeleri belirli bir eşik değerine göre seç
        corners = cv2.dilate(corners, None)
        img[corners > 0.01 * corners.max()] = [0, 0, 255]

        # Sonucu göster
        cv2.imshow('Harris Corner Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Kenar(self):
        if self.image_path is None:
            self.result_label.setText("Lütfen önce bir resim ekleyin.")
            return

        original_image = cv2.imread(self.image_path)
        bordered_image = cv2.copyMakeBorder(original_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 255, 0])

        result_image_path = "bordered_result_image.png"
        cv2.imwrite(result_image_path, bordered_image)

        result_pixmap = self.get_scaled_pixmap(result_image_path)
        self.result_label.setPixmap(result_pixmap)
        pass

    def gamma(self):
        if self.image_path is None:
            self.result_label.setText("Lütfen önce bir resim ekleyin.")
            return

        original_image = cv2.imread(self.image_path)
        gamma_value = 1.5  # Gamma değerini istediğiniz şekilde ayarlayın
        gamma_corrected = np.array(255 * (original_image / 255) ** gamma_value, dtype='uint8')

        result_image_path = "gamma_corrected_result_image.png"
        cv2.imwrite(result_image_path, gamma_corrected)

        result_pixmap = self.get_scaled_pixmap(result_image_path)
        self.result_label.setPixmap(result_pixmap)
        pass

    def blackwhite(self):
        if self.image_path is None:
            self.result_label.setText("Lütfen önce bir resim ekleyin.")
            return

        original_image = cv2.imread(self.image_path)
        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        result_image_path = "grayscale_result_image.png"
        cv2.imwrite(result_image_path, grayscale_image)

        result_pixmap = self.get_scaled_pixmap(result_image_path)
        self.result_label.setPixmap(result_pixmap)
        pass

    def histogram(self):
        if self.image_path is None:
            self.result_label.setText("Lütfen önce bir resim ekleyin.")
            return

        original_image = cv2.imread(self.image_path)

        # Histogram eşitleme işlemi
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)

        # Histogramı göster
        self.plot_histogram(equalized_image)

        result_image_path = "adjusted_pixel_values_result_image.png"
        cv2.imwrite(result_image_path, equalized_image)

        result_pixmap = self.get_scaled_pixmap(result_image_path)
        self.result_label.setPixmap(result_pixmap)

    def plot_histogram(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        plt.plot(hist, color='gray')
        plt.xlabel('Piksel Değeri')
        plt.ylabel('Frekans')
        plt.title('Histogram')
        plt.show()
    def otsu(self):
        if self.image_path is None:
            self.result_label.setText("Lütfen önce bir resim ekleyin.")
            return

        original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Otsu eşitleme işlemi
        _, otsu_equalized_image = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



        result_image_path = "otsu_equalization_result_image.png"
        cv2.imwrite(result_image_path, otsu_equalized_image)

        result_pixmap = self.get_scaled_pixmap(result_image_path)
        self.result_label.setPixmap(result_pixmap)
        pass

    def adaptivethres(self):
        if self.image_path is None:
            self.result_label.setText("Lütfen önce bir resim ekleyin.")
            return

        original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Adaptive thresholding işlemi
        adaptive_thresh = cv2.adaptiveThreshold(original_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,
                                                2)

        result_image_path = "adaptive_threshold_result_image.png"
        cv2.imwrite(result_image_path, adaptive_thresh)

        result_pixmap = self.get_scaled_pixmap(result_image_path)
        self.result_label.setPixmap(result_pixmap)
        pass

    def contours(self):
        if self.image_path is None:
            self.result_label.setText("Lütfen önce bir resim ekleyin.")
            return

        original_image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_image = original_image.copy()
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

        result_image_path = "contours_result_image.png"
        cv2.imwrite(result_image_path, result_image)

        result_pixmap = self.get_scaled_pixmap(result_image_path)
        self.result_label.setPixmap(result_pixmap)

    def sobel(self):
        if self.image_path is None:
            self.result_label.setText("Lütfen önce bir resim ekleyin.")
            return

        original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Sobel kenar tespiti işlemi
        sobel_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        sobel_edges = cv2.convertScaleAbs(magnitude)

        result_image_path = "sobel_edge_result_image.png"
        cv2.imwrite(result_image_path, sobel_edges)

        result_pixmap = self.get_scaled_pixmap(result_image_path)
        self.result_label.setPixmap(result_pixmap)
        pass

    def get_scaled_pixmap(self, image_path, target_width=400):
        image_reader = QImageReader(image_path)
        image_reader.setAutoTransform(True)
        size = image_reader.size()
        ratio = size.width() / size.height()

        target_height = int(target_width / ratio)
        scaled_pixmap = QPixmap(image_reader.read().scaled(target_width, target_height, Qt.KeepAspectRatio))

        return scaled_pixmap

    def detect_faces_webcam(self):
        # Kamera açma işlemi
        cap = cv2.VideoCapture(0)

        while True:
            # Kameradan bir kare al
            ret, frame = cap.read()

            if ret:
                # Yüz tespiti
                face_locations = face_recognition.face_locations(frame)

                # Yüzleri kare çerçeve içine al
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Yüz tespiti sonucunu göster
                cv2.imshow('Webcam ile yuz Tespiti', frame)

                # Çıkış için 'q' tuşuna basma kontrolü
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Kamera kapatma
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

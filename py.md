Artificial Intelligence Image Processing Application(Yapay Zeka Görsel İşleme Uygulaması)

This Python application leverages OpenCV, face_recognition, and PyQt5 to offer various image processing and computer vision functionalities.

**Key Features:**

- **Image Loading and Display:** Load and display images from files.
- **Face Detection:** Detect faces in loaded images or frames from a webcam.
- **Face Blurring:** Anonymize faces in images for privacy protection.
- **Watershape Segmentation:** Extract foreground objects based on intensity and connectedness.
- **Harris Corner Detection:** Identify prominent corners in images for feature tracking.
- **Image Bordering:** Add decorative borders to images.
- **Gamma Correction:** Adjust image contrast for improved visibility.
- **Grayscale Conversion:** Convert images to grayscale for artistic effects or analysis.
- **Histogram Adjustment:** Enhance image contrast and reduce noise using equalization.
- **Otsu Thresholding:** Automatically determine optimal intensity thresholds for segmentation.
- **Adaptive Thresholding:** Apply variable thresholds based on local image properties.
- **Contour Detection:** Find and highlight object outlines in images.
- **Sobel Edge Detection:** Extract edges and boundaries in images.

**Getting Started:**

1. Install required libraries: OpenCV, face_recognition, PyQt5.
2. Run the application (python main.py).
3. Use buttons to interact with features:
   - Load images using the "Resim Ekle" button.
   - Perform face detection (image or webcam) using the "Yüzleri Tanı" or "Webcam ile Yüzleri Tanı" buttons.
   - Experiment with other image processing operations using their respective buttons.

**Additional Notes:**

- The application uses external tools (OpenCV, face_recognition) for specific functionalities.
- Some features (like face detection) require specific image formats or webcam access.

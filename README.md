# Vision Pro Surveillance: Smart Lock with Liveness Detection

Vision Pro Surveillance is a high-speed, secure smart lock system designed for edge computing. By leveraging **Stereo Vision** and **Deep Learning**, the system prevents 2D spoofing attacks (photos/videos) through real-time depth estimation while maintaining high-speed face recognition on a Raspberry Pi 5.

## 🚀 Features

* **Liveness Detection:** Uses **Epipolar Geometry** and facial landmarks (eyes/nose) to calculate depth, ensuring the subject is a 3D human and not a 2D photo.
* **High-Speed Recognition:** Employs **YuNet** for millisecond-level face detection and **SFace** for robust identity verification.
* **Edge Optimized:** Specifically designed to run locally on a Raspberry Pi 5, eliminating cloud latency and privacy concerns.
* **Web Dashboard:** An interactive interface for remote user registration, live monitoring, and admin approval logs.
* **Dual-Camera Setup:** Utilizes two ESP32-CAM modules for stereo image acquisition.

## 🛠️ Hardware Requirements

* **Processing:** Raspberry Pi 5 (4GB or 8GB)
* **Imaging:** 2x ESP32-CAM Modules
* **Actuation:** Arduino Uno/Nano + 5V Relay + 12V Solenoid Lock
* **Power:** 12V DC Power Supply (for the lock) and 5V USB-C (for the Pi)

## 💻 Tech Stack

* **Language:** Python 3.9+
* **Vision:** OpenCV (Open Source Computer Vision Library)
* **Models:** YuNet (Detection), SFace (Recognition)
* **Frontend:** Flask / Streamlit (for the Web Dashboard)
* **Communication:** Serial (Pi to Arduino), HTTP/RTSP (ESP32 to Pi)


## 🔧 Installation & Setup

1. **Clone the Repository:**
```bash
git clone https://github.com/YourUsername/Vision-Pro-Surveillance.git
cd Vision-Pro-Surveillance

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Camera Calibration:**
Capture a series of checkerboard images using both ESP32-CAMs and run the calibration script to generate the `calibration.xml` file.
```bash
python calibrate_stereo.py

```


4. **Run the Application:**
```bash
python main.py

```



## 📊 Performance

| Metric | Result |
| --- | --- |
| **Detection Latency** | ~15ms (YuNet) |
| **Accuracy** | 100% (Tested on internal dataset) |
| **False Acceptance Rate (FAR)** | 0.0% |
| **Depth Sensitivity** | 0.015m - 0.080m |

## 🤝 Acknowledgments

* **Prof. Dr. Andreas Lehrmann** – For guidance and technical feedback.
* **THWS (Technical University of Applied Sciences Würzburg-Schweinfurt)** – For providing resources and infrastructure.

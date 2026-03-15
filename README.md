# 🎯 FocusLens AI: Intelligent Study Monitoring System

A real-time, computer-vision-based study monitoring dashboard designed to track focus, detect distractions, and calculate productivity using advanced AI models. Built with a sleek, modern glassmorphism UI, this full-stack application helps students optimize their study habits through data-driven insights.

## ✨ Key Features

* **Biometric Authentication:** Passwordless, highly secure student registration and login using MediaPipe Face Mesh (mapping 468 3D facial landmarks).
* **Real-Time Object Detection:** Integrates Ultralytics YOLOv8 Nano to instantly detect smartphone usage and discourage digital distractions.
* **Posture & Fatigue Tracking:** Calculates Eye Aspect Ratio (EAR) to detect drowsiness/sleeping, and monitors Head Pose (Yaw/Pitch/Roll) to ensure the user is looking at their workstation.
* **Smart Analytics Engine:** Dynamically calculates a Focus-to-Break ratio and an overall productivity score based on active session data.
* **Chronotype AI Coaching:** Analyzes historical focus scores to determine the student's peak cognitive performance window (Morning, Afternoon, or Evening/Night) and provides randomized, dynamic study tips.
* **Hardware-Safe Architecture:** Implements threading locks and JavaScript beacons to ensure the webcam hardware is safely released when navigating between pages or closing the browser.

## 🛠️ Technology Stack

**Backend & AI**
* **Python 3.11+**
* **Flask** (Web Framework, Routing, API Endpoints)
* **SQLite** (Lightweight Database for Session & User Management)
* **OpenCV** (Image processing & Camera I/O)
* **MediaPipe** (Facial recognition & geometry tracking)
* **Ultralytics YOLOv8** (Computer Vision & Object Detection)

**Frontend**
* **HTML5 / CSS3** (Custom Glassmorphism Dark Theme)
* **JavaScript** (Async fetch APIs, Client-side routing)
* **Chart.js** (Dynamic Data Visualization)

## 🚀 Installation & Setup

To run this project locally, follow these steps.

**1. Clone the repository**
```bash
git clone [https://github.com/Ana40705/Ai-Study-monitoring-system.git](https://github.com/Ana40705/Ai-Study-monitoring-system.git)
cd Ai-Study-monitoring-system
```
**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```
**3. Install Dependencies**
```bash
pip install -r requirements.txt
```
**4. Run the Application**
```bash
python app.py
```
**5. Access the Dashboard**
Open your web browser and navigate to http://127.0.0.1:5000.
Note: Because the database initializes automatically, you must click "New Student? Register here" to scan your face and create an account before your first login.

📂 Project Structure
```bash
├── app.py                 # Main Flask server and routing logic
├── camera.py              # AI inference, OpenCV feed, and state management
├── database.py            # SQLite schema and query functions
├── yolov8n.pt             # Pre-trained YOLOv8 weights (downloads automatically)
├── requirements.txt       # Project dependencies
└── templates/
   ├── index.html         # Landing page
   ├── login.html         # Biometric authentication portal
   ├── register.html      # New student onboarding
   ├── dashboard.html     # Live monitoring and metrics view
   └── report.html        # Post-session analytics and AI tips

```

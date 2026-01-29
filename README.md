**AI Based Student Study Real-Time Monitoring System**
This project is a computer vision-based application designed to monitor student focus during study sessions. It uses Artificial Intelligence to detect if a student is sleeping, distracted, or using a mobile phone, and provides a real-time focus score.

**🌟 What This System Does**
The monitor tracks four main behaviors to ensure a student stays focused:

1. Sleep Detection: Monitors eye-closing patterns (EAR) to detect drowsiness.

2. Phone Detection: Uses YOLOv8 to identify if a mobile phone appears in the camera frame.

3. Distraction Tracking: Detects when a student looks away from the screen for too long.

4. No Face Detection: Alerts the system if the student leaves their seat.

**🛠️ How to Set It Up**
Since we are using specific AI libraries, everyone needs to set up a local environment on their laptop. Follow these steps exactly:

1. Clone the Project
Open your terminal and run:
    git clone https://github.com/Ana40705/Ai-Study-monitoring-system.git
    cd Ai-Study-monitoring-system

2. Create a Virtual Environment
We must use a virtual environment so our libraries don't clash.
    python -m venv venv

3. Activate the Environment Windows: venv\Scripts\activate

4. Install Requirements
This will install Flask, OpenCV, MediaPipe, and Ultralytics (YOLO).

pip install -r requirements.txt


**🚀 How to Run the Project**
The Easy Way (Windows Only)
Simply double-click the run_project.bat file.

It activates the environment for you.

It starts the AI models (takes about 10-15 seconds).

It will automatically open your browser to http://localhost:5000.

The Manual Way
If you prefer using the terminal:

python app.py
Once it says "Running on...", open your browser and go to http://127.0.0.1:5000.

**📁 Project Structure**
app.py: The main Flask server that handles the web pages.

camera.py: The "brain" of the project where all AI detection logic happens.

database.py: Handles user registration and saves study session reports.

yolov8n.pt: The pre-trained AI weights used for object detection.

templates/: Contains the HTML files for the Dashboard, Login, and Reports.

requirements.txt: The list of all libraries needed to run the app.

**📊 Viewing Reports**
After you finish a study session, the data is saved to student_monitor.db. You can view your focus percentage and distraction counts in the Reports section of the web interface.

**⚠️ Troubleshooting**
Camera Not Opening: Make sure no other app (like Zoom, Teams, or Chrome) is using your webcam.

Missing venv: If the .bat file fails, ensure you created the venv folder in the first step.

Slow Performance: AI models require some processing power. Close heavy apps while running the monitor for the best experience.

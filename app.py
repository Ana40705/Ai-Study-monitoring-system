from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
import database
import time
import json

app = Flask(__name__)
database.init_db()

camera = VideoCamera()
current_user = "Guest"

@app.route('/')
def home():
    camera.start_stream()
    return render_template('login.html')

@app.route('/register_page')
def register_page():
    camera.start_stream()
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    student_id = request.form.get('student_id')
    email = request.form.get('email')
    
    metrics = camera.capture_metrics_snapshot()
    if metrics:
        if database.register_user(username, student_id, email, metrics):
            return jsonify({"status": "success", "message": "Registered Successfully!"})
        else:
            return jsonify({"status": "error", "message": "Username already exists."})
    else:
        return jsonify({"status": "error", "message": "No face detected."})

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    saved_metrics = database.get_user_face_data(username)
    if not saved_metrics: return jsonify({"status": "error", "message": "User not found."})
    
    current_metrics = camera.capture_metrics_snapshot()
    if not current_metrics: return jsonify({"status": "error", "message": "No face detected."})
    
    match_score = camera.compare_faces(saved_metrics, current_metrics)
    if match_score > 75:
        global current_user
        current_user = username
        # REMOVED: camera.set_reference_metrics(saved_metrics)
        return jsonify({"status": "success", "redirect": "/dashboard"})
    else:
        return jsonify({"status": "error", "message": f"Identity Mismatch ({int(match_score)}%)."})

@app.route('/dashboard')
def dashboard():
    camera.reset_state()
    camera.start_stream()
    return render_template('dashboard.html', user=current_user)

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else: time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(camera.get_stats())

@app.route('/stop_session', methods=['POST'])
def stop_session():
    stats = camera.get_stats()
    # Save the specific breakdown dictionary
    database.save_session(current_user, stats['focus_score'], camera.distracted_frames // 20, stats['details'])
    camera.stop_stream()
    return jsonify({"status": "saved"})

@app.route('/report')
def report():
    camera.stop_stream()
    raw_sessions = database.get_all_sessions()
    
    # Process sessions to include detailed breakdown
    processed_sessions = []
    
    # Global totals for the Pie Chart (Proxy Removed)
    global_breakdown = {"phone": 0, "sleep": 0, "look_away": 0}
    
    for row in raw_sessions:
        # row structure: (id, name, start, end, score, total_count, details_json)
        details = {"phone": 0, "sleep": 0, "look_away": 0}
        
        # Try to parse the details from the database
        if len(row) > 6 and row[6]:
            try:
                details = json.loads(row[6])
            except:
                pass # Keep defaults if error
        
        # Add to global totals
        global_breakdown["phone"] += details.get("phone", 0)
        global_breakdown["sleep"] += details.get("sleep", 0)
        global_breakdown["look_away"] += details.get("look_away", 0)

        # Create a clean object for the HTML
        session_data = {
            "id": row[0],
            "name": row[1],
            "date": row[2],
            "score": row[4],
            "total": row[5],
            "phone": details.get("phone", 0),
            "sleep": details.get("sleep", 0),
            "look_away": details.get("look_away", 0)
        }
        processed_sessions.append(session_data)

    # Prepare Graph Data (Last 15 sessions)
    recent = processed_sessions[:15]
    dates = [s["date"] for s in recent][::-1]
    scores = [s["score"] for s in recent][::-1]

    return render_template('report.html', 
                        sessions=processed_sessions, 
                        dates=dates, 
                        scores=scores, 
                        breakdown=global_breakdown)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
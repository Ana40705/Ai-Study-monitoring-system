from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
import database
import json
from datetime import datetime
import random

app = Flask(__name__)
database.init_db()
camera = VideoCamera()
current_user = None

@app.route('/')
def home(): 
    camera.stop_stream() 
    return render_template('index.html')

@app.route('/login_page')
def login_page(): 
    camera.stop_stream() 
    return render_template('login.html')

@app.route('/register_page')
def register_page(): 
    return render_template('register.html')

@app.route('/kill_camera', methods=['POST'])
def kill_camera():
    camera.stop_stream()
    return '', 204

@app.route('/login', methods=['POST'])
def login():
    global current_user
    user = request.form.get('username')
    sid = request.form.get('student_id')
    email = request.form.get('email')
    
    saved_metrics = database.get_user_face_data(user, sid, email)
    if not saved_metrics: 
        return jsonify({"status": "error", "message": "Invalid Credentials. User not found."})
    
    current_metrics = camera.capture_metrics_snapshot()
    if not current_metrics: 
        camera.stop_stream() 
        return jsonify({"status": "error", "message": "No face detected. Look directly at the camera."})
    
    if camera.compare_faces(saved_metrics, current_metrics) > 75:
        current_user = user
        camera.set_reference_metrics(saved_metrics)
        return jsonify({"status": "success", "redirect": "/dashboard"})
    
    camera.stop_stream() 
    return jsonify({"status": "error", "message": "Biometric Identity Mismatch"})

@app.route('/register', methods=['POST'])
def register():
    user = request.form.get('username')
    sid = request.form.get('student_id')
    email = request.form.get('email')
    
    metrics = camera.capture_metrics_snapshot()
    if metrics and database.register_user(user, sid, email, metrics):
        return jsonify({"status": "success", "message": "Registered Successfully!"})
    return jsonify({"status": "error", "message": "Registration Failed - Keep face steady"})

@app.route('/video_feed')
def video_feed():
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame: 
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/status')
def status(): 
    return jsonify(camera.get_stats())

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    paused = camera.toggle_pause()
    return jsonify({"status": "success", "is_paused": paused})

@app.route('/dashboard')
def dashboard():
    if not current_user: 
        return render_template('login.html')
    
    # FIX: Guarantees a fresh, active camera state every time a session begins
    camera.reset_state()
    
    return render_template('dashboard.html', user=current_user)

@app.route('/stop_session', methods=['POST'])
def stop_session():
    stats = camera.get_stats()
    database.save_session(
        current_user, 
        stats['focus_score'], 
        sum(stats['details'].values()), 
        stats['details'],
        stats['focus_min'],
        stats['break_min'],
        stats['productivity']
    )
    camera.is_paused = True 
    camera.stop_stream()
    return jsonify({"status": "success"})

@app.route('/report')
def report():
    if not current_user: 
        return render_template('login.html')
    
    raw_sessions = database.get_user_sessions(current_user)
    if not raw_sessions:
        return "No session data found. Please complete a session first."

    processed = []
    time_slots = {
        "Morning (6am-12pm)": {"total": 0, "cnt": 0}, 
        "Afternoon (12pm-6pm)": {"total": 0, "cnt": 0}, 
        "Evening/Night (6pm-12am)": {"total": 0, "cnt": 0}
    }

    for row in raw_sessions:
        details = json.loads(row[6]) if row[6] else {}
        processed.append({
            "date": row[2], "score": row[4], "focus_t": row[7], 
            "break_t": row[8], "prod": row[9], **details
        })
        try:
            hr = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S").hour
            slot = "Morning (6am-12pm)" if 6<=hr<12 else "Afternoon (12pm-6pm)" if 12<=hr<18 else "Evening/Night (6pm-12am)"
            time_slots[slot]["total"] += row[4]
            time_slots[slot]["cnt"] += 1
        except Exception:
            pass

    chronotype = {k: round(v["total"]/v["cnt"], 1) if v["cnt"]>0 else 0 for k,v in time_slots.items()}
    peak = max(chronotype, key=chronotype.get) if any(v["cnt"]>0 for v in time_slots.values()) else "No Data"

    tips = []
    latest = processed[0]
    previous = processed[1] if len(processed) > 1 else None

    if previous:
        prod_diff = round(latest["prod"] - previous["prod"], 1)
        if prod_diff > 5: tips.append({"title": "📈 Upward Trend", "text": f"Awesome! Productivity increased by {prod_diff}%."})
        elif prod_diff < -5: tips.append({"title": "📉 Slight Dip", "text": "Productivity dropped. Ensure you are taking restful breaks."})
            
    phone_advice = ["Phone detected multiple times. Try putting it in another room.", "Turn on 'Do Not Disturb' while studying.", "Keep your phone face-down and away from your desk."]
    sleep_advice = ["You seemed tired. Sleep consolidates memory—don't skimp on it!", "Drowsiness detected. Try doing a 5-minute stretch.", "Your eyes are heavy. It might be time to rest."]
    focus_advice = ["Try the Pomodoro technique (25m study, 5m break) to boost efficiency.", "Your focus-to-break ratio is low. Break your tasks into smaller chunks.", "Change your study environment for a quick reset."]

    if latest.get("phone", 0) > 2: tips.append({"title": "Digital Distraction", "text": random.choice(phone_advice)})
    if latest.get("sleep", 0) > 1: tips.append({"title": "Fatigue Warning", "text": random.choice(sleep_advice)})
    if latest["prod"] < 70: tips.append({"title": "Focus Strategy", "text": random.choice(focus_advice)})
        
    if len(tips) == 0 or (len(tips) == 1 and "Trend" in tips[0]["title"]):
        tips.append({"title": "Legendary Focus 🏆", "text": "Zero major distractions. You are locked in. Keep up this elite routine!"})
    
    if peak != "No Data":
        tips.insert(0, {"title": "Peak Performance Window", "text": f"You are a '{peak}' studier. Try to schedule your most difficult subjects during this time."})

    return render_template('report.html', sessions=processed, ai_tips=tips, chronotype=chronotype, peak_window=peak)

if __name__ == '__main__': 
    app.run(debug=True, port=5000)
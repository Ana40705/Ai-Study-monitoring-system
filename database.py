import sqlite3
import datetime
import json

DB_NAME = "student_monitor.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            student_id TEXT,
            email TEXT,
            face_data TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            start_time TEXT,
            end_time TEXT,
            focus_score INTEGER,
            distraction_count INTEGER,
            distraction_details TEXT,
            total_focus_time FLOAT,
            total_break_time FLOAT,
            overall_productivity FLOAT
        )
    ''')
    conn.commit()
    conn.close()

def get_user_face_data(username, student_id, email):
    """Updated to require all three credentials for higher security."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT face_data FROM users WHERE username=? AND student_id=? AND email=?", (username, student_id, email))
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

def register_user(username, student_id, email, face_metrics):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?)", 
                  (username, student_id, email, json.dumps(face_metrics)))
        conn.commit()
        return True
    except: return False
    finally: conn.close()

def get_user_sessions(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE user_name=? ORDER BY id DESC", (username,))
    data = c.fetchall()
    conn.close()
    return data

def save_session(user_name, focus_score, distractions, details, f_time, b_time, prod):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO sessions (user_name, start_time, end_time, focus_score, 
                            distraction_count, distraction_details, 
                            total_focus_time, total_break_time, overall_productivity)
        VALUES (?, datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?, ?)
    ''', (user_name, end_time, focus_score, distractions, json.dumps(details), f_time, b_time, prod))
    conn.commit()
    conn.close()
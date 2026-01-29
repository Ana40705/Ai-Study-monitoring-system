import sqlite3
import datetime
import json

DB_NAME = "student_monitor.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Updated Sessions Table: Now stores specific distraction counts
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            start_time TEXT,
            end_time TEXT,
            focus_score INTEGER,
            distraction_count INTEGER,
            distraction_details TEXT
        )
    ''')
    
    # Updated Users Table: Adds Student ID and Email
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            student_id TEXT,
            email TEXT,
            face_data TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database Initialized with New Fields.")

def register_user(username, student_id, email, face_metrics):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    metrics_json = json.dumps(face_metrics)
    try:
        c.execute("INSERT OR REPLACE INTO users (username, student_id, email, face_data) VALUES (?, ?, ?, ?)", 
                  (username, student_id, email, metrics_json))
        conn.commit()
        return True
    except Exception as e:
        print(f"Registration Error: {e}")
        return False
    finally:
        conn.close()

def get_user_face_data(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT face_data FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row: return json.loads(row[0])
    return None

def save_session(user_name, focus_score, total_distractions, details_dict):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save the dictionary {"phone": 5, "sleep": 2...} as a string
    details_json = json.dumps(details_dict)
    
    c.execute('''
        INSERT INTO sessions (user_name, start_time, end_time, focus_score, distraction_count, distraction_details)
        VALUES (?, datetime('now'), ?, ?, ?, ?)
    ''', (user_name, end_time, focus_score, total_distractions, details_json))
    conn.commit()
    conn.close()

def get_all_sessions():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM sessions ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return data
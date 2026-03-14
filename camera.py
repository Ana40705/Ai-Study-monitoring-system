import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time
import math
import winsound
import threading 

class VideoCamera:
    def __init__(self):
        self.video = None 
        print("Loading YOLO...")
        self.yolo = YOLO("yolov8n.pt") 
        print("Loading MediaPipe...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.EAR_CLOSED_THRESHOLD = 0.19    
        self.BLINK_MAX_SECONDS = 0.4        
        self.DISTRACTION_SECONDS_REQUIRED = 1.0 
        self.YAW_THRESH = 35    
        self.PITCH_THRESH = 30
        self.FPS_EST = 20.0  
        self.BLINK_MAX_FRAMES = int(self.BLINK_MAX_SECONDS * self.FPS_EST)
        self.DISTRACTION_FRAMES_REQUIRED = int(self.DISTRACTION_SECONDS_REQUIRED * self.FPS_EST)

        self.reference_face_metrics = None 
        
        # FIX: Hardware lock to prevent OpenCV crashes from rapid switching
        self.camera_lock = threading.Lock() 
        self.is_camera_active = False
        
        self.reset_state()
        self.MODEL_POINTS_3D = np.array([(0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], dtype=np.float64)

    def reset_state(self):
        self.is_paused = False
        self.total_focus_seconds = 0
        self.total_break_seconds = 0
        self.distracted = False
        self.phone_detected = False
        self.focus_score = 100.0
        self.status_text = "Active"
        self.frame_count = 0
        self.distracted_frames = 0
        self.consec_not_focused_frames = 0
        self.closed_eyes_frames = 0
        self.phone_consec_frames = 0
        self.last_phone_boxes = []
        self.count_phone = self.count_sleep = self.count_look_away = 0
        self.last_state_change = time.time()

    def play_alert(self):
        winsound.Beep(1000, 300) 

    def toggle_pause(self):
        now = time.time()
        duration = now - self.last_state_change
        if self.is_paused:
            self.total_break_seconds += duration
            self.is_paused = False
            self.start_stream() 
            self.status_text = "Active"
        else:
            self.total_focus_seconds += duration
            self.is_paused = True
            self.stop_stream() 
            self.status_text = "On Break"
        self.last_state_change = now
        return self.is_paused

    def start_stream(self):
        """Thread-safe camera startup"""
        with self.camera_lock:
            if not self.is_camera_active:
                try:
                    self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.is_camera_active = True
                except Exception as e:
                    print(f"Camera Initialization Warning: {e}")

    def stop_stream(self):
        """Thread-safe camera shutdown"""
        with self.camera_lock:
            if self.is_camera_active and self.video:
                try:
                    self.video.release()
                except Exception as e:
                    print(f"Ignored release error: {e}")
                self.video = None
                self.is_camera_active = False

    def set_reference_metrics(self, metrics):
        self.reference_face_metrics = metrics

    def get_distance(self, p1, p2): return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    def euclidean(self, p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))

    def get_face_metrics(self, landmarks):
        try:
            base_w = self.get_distance(landmarks[33], landmarks[263])
            if base_w == 0: return None
            return [self.get_distance(landmarks[1], landmarks[152])/base_w, 
                    self.get_distance(landmarks[33], landmarks[1])/base_w, 
                    self.get_distance(landmarks[263], landmarks[1])/base_w, 
                    self.get_distance(landmarks[61], landmarks[291])/base_w]
        except: return None

    def capture_metrics_snapshot(self):
        if not self.is_camera_active:
            self.start_stream()
        
        # Safely try to read frames
        if self.video:
            for _ in range(15): self.video.read() 
            success, frame = self.video.read()
            if not success: return None
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)
            if res.multi_face_landmarks:
                return self.get_face_metrics(res.multi_face_landmarks[0].landmark)
        return None

    def compare_faces(self, saved, current):
        if not saved or not current: return 0.0
        error = sum([abs(saved[i] - current[i]) for i in range(len(saved))])
        return max(0, 100 - (error * 115))

    def get_ear(self, lm, idx, w, h):
        pts = [np.array([lm[i].x * w, lm[i].y * h]) for i in idx]
        return (self.euclidean(pts[1], pts[5]) + self.euclidean(pts[2], pts[4])) / (2.0 * (self.euclidean(pts[0], pts[3]) + 1e-6))

    def solve_head_pose(self, lm, w, h):
        img_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in [1, 152, 33, 263, 61, 291]], dtype=np.float64)
        focal = w; center = (w/2, h/2)
        cam_mat = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64)
        success, rot, trans = cv2.solvePnP(self.MODEL_POINTS_3D, img_pts, cam_mat, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return 0, 0, 0
        rmat, _ = cv2.Rodrigues(rot)
        pmat = np.hstack((rmat, trans))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pmat)
        return euler[0][0], euler[1][0], euler[2][0]

    def get_frame(self):
        if self.is_paused:
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, "CAMERA OFF", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            return cv2.imencode('.jpg', black_frame)[1].tobytes()

        if not self.is_camera_active:
            self.start_stream()
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, "STARTING CAMERA...", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            return cv2.imencode('.jpg', black_frame)[1].tobytes()

        # Added safety check before reading
        if self.video is None:
            return None

        success, frame = self.video.read()
        if not success: return None
        h, w, _ = frame.shape
        self.frame_count += 1
        
        if self.frame_count % 3 == 0:
            res = self.yolo.predict(frame, verbose=False, classes=[67], conf=0.25) 
            if len(res[0].boxes) > 0:
                self.phone_consec_frames += 1
                self.last_phone_boxes = [list(map(int, box.xyxy[0])) for box in res[0].boxes]
            else: 
                self.phone_consec_frames = 0
                self.last_phone_boxes = []
            
            if self.phone_consec_frames == 2: 
                self.count_phone += 1
                self.play_alert()
                
            self.phone_detected = self.phone_consec_frames > 1

        if self.phone_detected:
            for box in self.last_phone_boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                cv2.putText(frame, "PHONE DETECTED", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        focused = True
        reason = "Focused"

        if not res.multi_face_landmarks:
            focused = False; reason = "No Face"
            self.closed_eyes_frames = 0
        else:
            lm = res.multi_face_landmarks[0].landmark
            
            pitch_raw, yaw, roll = self.solve_head_pose(lm, w, h)
            pitch = pitch_raw - 180 if pitch_raw > 90 else (pitch_raw + 180 if pitch_raw < -90 else pitch_raw)

            left_ear = self.get_ear(lm, [33, 160, 158, 133, 153, 144], w, h)
            right_ear = self.get_ear(lm, [263, 387, 385, 362, 380, 373], w, h)
            avg_ear = (left_ear + right_ear)/2.0
            
            if avg_ear < self.EAR_CLOSED_THRESHOLD: self.closed_eyes_frames += 1
            else: self.closed_eyes_frames = 0
            
            is_sleeping = self.closed_eyes_frames > self.BLINK_MAX_FRAMES
            looking_away = abs(yaw) > self.YAW_THRESH or abs(pitch) > self.PITCH_THRESH

            if is_sleeping:
                focused = False; reason = "Sleeping"
                cv2.putText(frame, "SLEEPING!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                if self.closed_eyes_frames == self.BLINK_MAX_FRAMES + 1: 
                    self.count_sleep += 1
                    self.play_alert() 
            elif looking_away:
                focused = False; reason = "Looking Away"
                cv2.putText(frame, "LOOKING AWAY", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                if self.consec_not_focused_frames == 1: self.count_look_away += 1

            cv2.putText(frame, "TRACKING ACTIVE", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            self.mp_drawing.draw_landmarks(frame, res.multi_face_landmarks[0], self.mp_face_mesh.FACEMESH_TESSELATION, None, self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

        if self.phone_detected: self.distracted = True; self.status_text = "Phone Detected"
        elif not focused:
            self.consec_not_focused_frames += 1
            if self.consec_not_focused_frames >= self.DISTRACTION_FRAMES_REQUIRED:
                self.distracted = True; self.status_text = reason
        else:
            self.consec_not_focused_frames = 0; self.distracted = False; self.status_text = "Focused"

        if self.distracted: self.distracted_frames += 1
        if self.frame_count > 0: self.focus_score = 100 - (self.distracted_frames / self.frame_count * 100)

        return cv2.imencode('.jpg', frame)[1].tobytes()

    def get_stats(self):
        now = time.time()
        delta = now - self.last_state_change
        f_sec = self.total_focus_seconds + (delta if not self.is_paused else 0)
        b_sec = self.total_break_seconds + (delta if self.is_paused else 0)
        total = f_sec + b_sec
        prod = (f_sec / total * self.focus_score) if total > 0 else 0
        
        return {
            "distracted": self.distracted, "focus_score": int(self.focus_score),
            "status_text": self.status_text, "is_paused": self.is_paused,
            "focus_min": round(f_sec/60, 2), "break_min": round(b_sec/60, 2),
            "productivity": round(prod, 1),
            "details": {"phone": self.count_phone, "sleep": self.count_sleep, "look_away": self.count_look_away}
        }
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time
import math

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

        # --- TUNING PARAMETERS ---
        self.EAR_NORMAL_THRESHOLD = 0.16      # Standard eye openness for "Sleeping"
        self.EAR_LOOKING_DOWN_THRESHOLD = 0.12 # [NEW] Stricter threshold when looking down (allows squinting/reading)
        
        self.BLINK_MAX_SECONDS = 2.0          # [FIX] Delayed alarm for 2 seconds
        self.DISTRACTION_SECONDS_REQUIRED = 1.0 
        
        self.YAW_THRESH = 35    
        self.PITCH_THRESH = 35                # Increased slightly to allow more head movement
        self.FPS_EST = 20.0  
        
        self.BLINK_MAX_FRAMES = int(self.BLINK_MAX_SECONDS * self.FPS_EST)
        self.DISTRACTION_FRAMES_REQUIRED = int(self.DISTRACTION_SECONDS_REQUIRED * self.FPS_EST)

        self.reset_state()
        
        # 3D Points for Pose Estimation
        self.MODEL_POINTS_3D = np.array([(0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], dtype=np.float64)

    def reset_state(self):
        self.distracted = False
        self.phone_detected = False
        self.focus_score = 100.0
        self.status_text = "Active"
        self.frame_count = 0
        self.distracted_frames = 0
        self.consec_not_focused_frames = 0
        self.closed_eyes_frames = 0
        self.phone_consec_frames = 0
        self.count_phone = 0
        self.count_sleep = 0
        self.count_look_away = 0

    def start_stream(self):
        if self.video is None or not self.video.isOpened(): self.video = cv2.VideoCapture(0)

    def stop_stream(self):
        if self.video and self.video.isOpened(): self.video.release()
        self.video = None

    # --- MATH HELPERS ---
    def get_distance(self, p1, p2): return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    def euclidean(self, p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))

    def get_face_metrics(self, landmarks):
        try:
            base_w = self.get_distance(landmarks[33], landmarks[263])
            if base_w == 0: return None
            return [self.get_distance(landmarks[1], landmarks[152])/base_w, self.get_distance(landmarks[33], landmarks[1])/base_w, self.get_distance(landmarks[263], landmarks[1])/base_w, self.get_distance(landmarks[61], landmarks[291])/base_w]
        except: return None

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
        if self.video is None: return None
        success, frame = self.video.read()
        if not success: return None
        h, w, _ = frame.shape
        self.frame_count += 1
        
        # 1. Phone Detection (with higher confidence to avoid chargers)
        if self.frame_count % 3 == 0:
            res = self.yolo.predict(frame, verbose=False, classes=[67], conf=0.5) 
            if len(res[0].boxes) > 0:
                self.phone_consec_frames += 1
                for box in res[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "PHONE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            else: self.phone_consec_frames = max(0, self.phone_consec_frames - 1)
            
            if self.phone_consec_frames == 5: self.count_phone += 1
            self.phone_detected = self.phone_consec_frames > 4

        # 2. Face Analysis
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        focused = True
        reason = "Focused"

        if not res.multi_face_landmarks:
            focused = False; reason = "No Face"
            self.closed_eyes_frames = 0
        else:
            lm = res.multi_face_landmarks[0].landmark
            
            # --- ANGLES ---
            pitch_raw, yaw, roll = self.solve_head_pose(lm, w, h)
            pitch = pitch_raw - 180 if pitch_raw > 90 else (pitch_raw + 180 if pitch_raw < -90 else pitch_raw)
            
            # --- EYES & SLEEP LOGIC ---
            left_ear = self.get_ear(lm, [33, 160, 158, 133, 153, 144], w, h)
            right_ear = self.get_ear(lm, [263, 387, 385, 362, 380, 373], w, h)
            avg_ear = (left_ear + right_ear)/2.0
            
            # [FIX] DYNAMIC THRESHOLD
            # If pitch > 15 (Looking Down), use stricter threshold (0.12) because eyes naturally narrow.
            # If pitch < 15 (Looking Forward), use normal threshold (0.16).
            is_looking_down = pitch > 15
            current_ear_thresh = self.EAR_LOOKING_DOWN_THRESHOLD if is_looking_down else self.EAR_NORMAL_THRESHOLD

            if avg_ear < current_ear_thresh:
                self.closed_eyes_frames += 1
            else:
                self.closed_eyes_frames = 0
            
            # Only trigger sleep after 2.0 seconds (BLINK_MAX_FRAMES)
            is_sleeping = self.closed_eyes_frames > self.BLINK_MAX_FRAMES
            looking_away = abs(yaw) > self.YAW_THRESH or abs(pitch) > self.PITCH_THRESH

            # Status Priority
            if self.phone_detected:
                focused = False; reason = "Phone" 
            elif is_sleeping:
                focused = False; reason = "Sleeping"
                cv2.putText(frame, "SLEEPING!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                if self.closed_eyes_frames == self.BLINK_MAX_FRAMES + 1: self.count_sleep += 1
            elif looking_away:
                focused = False; reason = "Looking Away"
                cv2.putText(frame, "LOOKING AWAY", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                if self.consec_not_focused_frames == 1: self.count_look_away += 1

            self.mp_drawing.draw_landmarks(image=frame, landmark_list=res.multi_face_landmarks[0], connections=self.mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # Status Update
        if self.phone_detected: self.distracted = True; self.status_text = "Phone Detected"
        elif not focused:
            self.consec_not_focused_frames += 1
            if self.consec_not_focused_frames >= self.DISTRACTION_FRAMES_REQUIRED:
                self.distracted = True; self.status_text = reason
        else:
            self.consec_not_focused_frames = 0; self.distracted = False; self.status_text = "Focused"

        if self.distracted: self.distracted_frames += 1
        if self.frame_count > 0: self.focus_score = 100 - (self.distracted_frames / self.frame_count * 100)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_stats(self):
        return {
            "distracted": self.distracted,
            "focus_score": int(self.focus_score),
            "status_text": self.status_text,
            "details": {
                "phone": self.count_phone,
                "sleep": self.count_sleep,
                "look_away": self.count_look_away
            }
        }
        
    def capture_metrics_snapshot(self):
        was_closed = self.video is None or not self.video.isOpened()
        if was_closed: 
            self.start_stream()
            for _ in range(5): self.video.read()
        success, frame = self.video.read()
        if not success: return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if res.multi_face_landmarks: return self.get_face_metrics(res.multi_face_landmarks[0].landmark)
        return None
# pose_checker.py
import cv2
import numpy as np
import mediapipe as mp

# We use this specific way to access the solutions
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)



def calculate_angle(a, b, c):
    """Calculate the 2D angle at point B"""
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_landmark_coords(landmarks, index):
    """Extracts x, y, z as a list"""
    l = landmarks[index]
    if isinstance(l, dict):
        return [l['x'], l['y'], l.get('z', 0)]
    return [l.x, l.y, l.z]

def get_dist(p1, p2):
    """Calculates Euclidean distance between two points in 2D"""
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# --- EXACT POSE CHECK FUNCTIONS ---

def check_hands_up(landmarks):
    # Wrist, Elbow, Shoulder for both sides
    lw, le, ls = [get_landmark_coords(landmarks, i) for i in [16, 14, 12]]
    rw, re, rs = [get_landmark_coords(landmarks, i) for i in [15, 13, 11]]
    nose = get_landmark_coords(landmarks, 0)

    # 1. Height Check: Wrists must be significantly above the nose
    above_head = lw[1] < (nose[1] - 0.05) and rw[1] < (nose[1] - 0.05)
    
    # 2. Straightness: Angles should be near 180
    l_angle = calculate_angle(ls, le, lw)
    r_angle = calculate_angle(rs, re, rw)

    # 3. Verticality: Wrists should be roughly over shoulders (not a 'V' shape)
    l_vertical = abs(lw[0] - ls[0]) < 0.12
    r_vertical = abs(rw[0] - rs[0]) < 0.12

    if not above_head: return False, "Reach higher! Hands above your head"
    if l_angle < 165 or r_angle < 165: return False, "Straighten your elbows fully"
    if not (l_vertical and r_vertical): return False, "Keep your arms straight up, not wide"
    
    return True, "✅ Perfect vertical reach!"

def check_toe_touch(landmarks):
    # Shoulder, Hip, Knee, Ankle (using left side as reference)
    ls, lh, lk, la = [get_landmark_coords(landmarks, i) for i in [12, 24, 26, 28]]
    lw, rw = [get_landmark_coords(landmarks, i) for i in [16, 15]]
    
    # 1. Knee Straightness: Crucial to prevent 'cheating' by squatting
    knee_angle = calculate_angle(lh, lk, la)
    # 2. Hip Hinge: Torso relative to legs
    hip_angle = calculate_angle(ls, lh, lk)
    # 3. Reach: Distance between hands and ankles
    hand_to_foot = min(get_dist(lw, la), get_dist(rw, la))

    if knee_angle < 160: return False, "Keep your knees straight!"
    if hip_angle > 75: return False, "Bend more at the waist"
    if hand_to_foot > 0.25: return False, "Reach closer to your toes"
    
    return True, "✅ Great flexibility and form!"

def check_cross_arm_left(landmarks):
    # Stretching Left: Wrist(16), Elbow(14), Shoulder(12). Checked against Right Shoulder(11)
    lw, le, ls = [get_landmark_coords(landmarks, i) for i in [16, 14, 12]]
    rs = get_landmark_coords(landmarks, 11)
    
    # Arm straightness
    arm_straight = calculate_angle(ls, le, lw) > 160
    # Cross distance: Wrist must be well past the opposite shoulder
    shoulder_width = get_dist(ls, rs)
    crossed = lw[0] > (rs[0] + shoulder_width * 0.1)
    # Height: Elbow should stay at shoulder height
    at_height = abs(le[1] - ls[1]) < 0.1

    if not arm_straight: return False, "Keep your left arm straight"
    if not at_height: return False, "Keep your arm at shoulder height"
    if not crossed: return False, "Pull the arm tighter across your chest"
    return True, "✅ Excellent shoulder stretch"

def check_cross_arm_right(landmarks):
    rw, re, rs = [get_landmark_coords(landmarks, i) for i in [15, 13, 11]]
    ls = get_landmark_coords(landmarks, 12)
    
    arm_straight = calculate_angle(rs, re, rw) > 160
    shoulder_width = get_dist(ls, rs)
    crossed = rw[0] < (ls[0] - shoulder_width * 0.1)
    at_height = abs(re[1] - rs[1]) < 0.1

    if not arm_straight: return False, "Keep your right arm straight"
    if not at_height: return False, "Keep your arm at shoulder height"
    if not crossed: return False, "Pull the arm tighter across your chest"
    return True, "✅ Excellent shoulder stretch"

def check_tricep_left(landmarks):
    # Left Arm: Shoulder(12), Elbow(14), Wrist(16)
    ls, le, lw = [get_landmark_coords(landmarks, i) for i in [12, 14, 16]]
    nose = get_landmark_coords(landmarks, 0)
    
    # 1. Elbow height: Elbow should be high above shoulder
    elbow_high = le[1] < (ls[1] - 0.1)
    # 2. Elbow angle: Arm must be bent (hand reaching down back)
    elbow_angle = calculate_angle(ls, le, lw)
    # 3. Proximity: Elbow should be close to the head (X-axis)
    elbow_near_head = abs(le[0] - nose[0]) < 0.15

    if not elbow_high: return False, "Point your left elbow straight up"
    if elbow_angle > 70: return False, "Reach your hand further down your back"
    if not elbow_near_head: return False, "Tuck your elbow closer to your head"
    return True, "✅ Tricep stretch is perfect"

def check_tricep_right(landmarks):
    rs, re, rw = [get_landmark_coords(landmarks, i) for i in [11, 13, 15]]
    nose = get_landmark_coords(landmarks, 0)
    
    elbow_high = re[1] < (rs[1] - 0.1)
    elbow_angle = calculate_angle(rs, re, rw)
    elbow_near_head = abs(re[0] - nose[0]) < 0.15

    if not elbow_high: return False, "Point your right elbow straight up"
    if elbow_angle > 70: return False, "Reach your hand further down your back"
    if not elbow_near_head: return False, "Tuck your elbow closer to your head"
    return True, "✅ Tricep stretch is perfect"


def check_pose_logic(landmarks, stretch_name):
    """Route to the correct pose check function"""
    checks = {
        "hands_up": check_hands_up,
        "toe_touch": check_toe_touch,
        "cross_arm_left": check_cross_arm_left,
        "cross_arm_right": check_cross_arm_right,
        "tricep_left": check_tricep_left,
        "tricep_right": check_tricep_right,
    }
    
    if stretch_name in checks:
        return checks[stretch_name](landmarks)
    return False, "Unknown stretch"

def analyze_pose(image_bytes, stretch_name):
    """
    Analyze pose from image bytes and return results
    """
    try:
        # Convert bytes to image
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"detected": False, "correct": False, "message": "Invalid image", "landmarks": []}
        
        # Process with MediaPipe
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({"x": landmark.x, "y": landmark.y, "z": landmark.z})
            
            # Run specific logic
            is_correct, message = check_pose_logic(landmarks, stretch_name)
            
            return {
                "detected": True,
                "correct": is_correct,
                "message": message,
                "landmarks": landmarks
            }
        else:
            return {
                "detected": False,
                "correct": False,
                "message": "🔍 No person detected",
                "landmarks": []
            }
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"detected": False, "correct": False, "message": f"Error: {str(e)}", "landmarks": []}

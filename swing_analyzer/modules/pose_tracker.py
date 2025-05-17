import cv2
import mediapipe as mp
import numpy as np

class PoseTracker:
    def __init__(self):
        """Initialize the MediaPipe pose estimation model."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define key golf swing landmarks of interest
        self.key_landmarks = {
            'head': [self.mp_pose.PoseLandmark.NOSE],
            'shoulders': [self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                          self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'elbows': [self.mp_pose.PoseLandmark.LEFT_ELBOW, 
                       self.mp_pose.PoseLandmark.RIGHT_ELBOW],
            'wrists': [self.mp_pose.PoseLandmark.LEFT_WRIST, 
                       self.mp_pose.PoseLandmark.RIGHT_WRIST],
            'hips': [self.mp_pose.PoseLandmark.LEFT_HIP, 
                     self.mp_pose.PoseLandmark.RIGHT_HIP],
            'knees': [self.mp_pose.PoseLandmark.LEFT_KNEE, 
                      self.mp_pose.PoseLandmark.RIGHT_KNEE],
            'ankles': [self.mp_pose.PoseLandmark.LEFT_ANKLE, 
                       self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        }
        
        # History of landmarks to track movement
        self.landmarks_history = []
        self.max_history_frames = 60  # Store about 2 seconds at 30fps
        
    def process_frame(self, frame):
        """
        Process a video frame to detect pose landmarks.
        
        Args:
            frame: BGR video frame from OpenCV
            
        Returns:
            processed_frame: Frame with pose landmarks drawn
            landmarks: Normalized pose landmarks if detected, None otherwise
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks on the frame
        processed_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                processed_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Store landmarks in history
            self.landmarks_history.append(results.pose_landmarks)
            if len(self.landmarks_history) > self.max_history_frames:
                self.landmarks_history.pop(0)
                
        return processed_frame, results.pose_landmarks
    
    def get_landmark_position(self, landmarks, landmark_id):
        """
        Get the pixel coordinates of a specific landmark.
        
        Args:
            landmarks: Pose landmarks from MediaPipe
            landmark_id: ID of the landmark to get
            
        Returns:
            (x, y): Tuple of coordinates or None if landmark not detected
        """
        if not landmarks:
            return None
        
        landmark = landmarks.landmark[landmark_id]
        return (landmark.x, landmark.y, landmark.z, landmark.visibility)
    
    def get_angle(self, landmarks, point1, point2, point3):
        """
        Calculate the angle between three points.
        
        Args:
            landmarks: Pose landmarks from MediaPipe
            point1, point2, point3: Landmark IDs forming the angle, where point2 is the vertex
            
        Returns:
            angle: Angle in degrees or None if landmarks not detected
        """
        if not landmarks:
            return None
        
        # Get coordinates
        p1 = np.array([landmarks.landmark[point1].x, landmarks.landmark[point1].y])
        p2 = np.array([landmarks.landmark[point2].x, landmarks.landmark[point2].y])
        p3 = np.array([landmarks.landmark[point3].x, landmarks.landmark[point3].y])
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def get_distance(self, landmarks, point1, point2):
        """
        Calculate the distance between two landmarks.
        
        Args:
            landmarks: Pose landmarks from MediaPipe
            point1, point2: Landmark IDs to measure distance between
            
        Returns:
            distance: Normalized distance or None if landmarks not detected
        """
        if not landmarks:
            return None
        
        # Get coordinates
        p1 = np.array([landmarks.landmark[point1].x, landmarks.landmark[point1].y, landmarks.landmark[point1].z])
        p2 = np.array([landmarks.landmark[point2].x, landmarks.landmark[point2].y, landmarks.landmark[point2].z])
        
        # Calculate distance
        return np.linalg.norm(p1 - p2)
    
    def calculate_body_metrics(self, landmarks):
        """
        Calculate key metrics about body positioning from landmarks.
        
        Args:
            landmarks: Pose landmarks from MediaPipe
            
        Returns:
            metrics: Dictionary of calculated metrics
        """
        if not landmarks:
            return {}
        
        metrics = {}
        
        # Head position
        if all(self.get_landmark_position(landmarks, lm) for lm in self.key_landmarks['head']):
            metrics['head_position'] = self.get_landmark_position(landmarks, self.mp_pose.PoseLandmark.NOSE)
        
        # Shoulder angle (important for swing plane)
        if all(self.get_landmark_position(landmarks, lm) for lm in self.key_landmarks['shoulders']):
            metrics['shoulder_angle'] = self.get_angle(
                landmarks,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_HIP
            )
        
        # Hip rotation (critical for power generation)
        if all(self.get_landmark_position(landmarks, lm) for lm in self.key_landmarks['hips']):
            metrics['hip_angle'] = self.get_angle(
                landmarks,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER
            )
        
        # Knee flex (for stability)
        if all(self.get_landmark_position(landmarks, lm) for lm in self.key_landmarks['knees'] + self.key_landmarks['hips'] + self.key_landmarks['ankles']):
            metrics['left_knee_angle'] = self.get_angle(
                landmarks,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE
            )
            metrics['right_knee_angle'] = self.get_angle(
                landmarks,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            )
        
        # Arm extension (for swing radius control)
        if all(self.get_landmark_position(landmarks, lm) for lm in self.key_landmarks['shoulders'] + self.key_landmarks['elbows'] + self.key_landmarks['wrists']):
            metrics['left_arm_angle'] = self.get_angle(
                landmarks,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST
            )
            metrics['right_arm_angle'] = self.get_angle(
                landmarks,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_WRIST
            )
            
            # Calculate wrist-to-shoulder distances to detect arm overextension
            metrics['left_arm_extension'] = (
                self.get_distance(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_WRIST)
            )
            metrics['right_arm_extension'] = (
                self.get_distance(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_WRIST)
            )
        
        return metrics
    
    def reset(self):
        """Reset the landmark history."""
        self.landmarks_history = [] 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from datetime import datetime

class VisualFeedback:
    def __init__(self, pose_tracker, swing_analyzer):
        """
        Initialize the visual feedback generator.
        
        Args:
            pose_tracker: Instance of PoseTracker class
            swing_analyzer: Instance of SwingAnalyzer class
        """
        self.pose_tracker = pose_tracker
        self.swing_analyzer = swing_analyzer
        
        # Define colors for different visualizations
        self.colors = {
            'backswing': (0, 255, 0),  # Green
            'downswing': (0, 0, 255),  # Red
            'follow_through': (255, 165, 0),  # Orange
            'text': (255, 255, 255),  # White
            'angle': (255, 255, 0),  # Yellow
            'error': (0, 0, 255),  # Red
            'good': (0, 255, 0),  # Green
            'trace': (0, 255, 255)  # Cyan
        }
        
        # Trail of key points to visualize swing path
        self.trail_points = {
            'right_wrist': [],
            'right_elbow': [],
            'right_shoulder': [],
            'left_wrist': [],
            'head': []
        }
        
        self.max_trail_length = 30  # Number of frames to keep in trail
    
    def add_trail_points(self, landmarks):
        """
        Add current landmark positions to the trail history.
        
        Args:
            landmarks: MediaPipe pose landmarks
        """
        if not landmarks:
            return
            
        # Map of point names to MediaPipe landmarks
        landmark_map = {
            'right_wrist': self.pose_tracker.mp_pose.PoseLandmark.RIGHT_WRIST,
            'right_elbow': self.pose_tracker.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'right_shoulder': self.pose_tracker.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_wrist': self.pose_tracker.mp_pose.PoseLandmark.LEFT_WRIST,
            'head': self.pose_tracker.mp_pose.PoseLandmark.NOSE
        }
        
        # Get position for each tracked point
        for point_name, landmark_id in landmark_map.items():
            position = self.pose_tracker.get_landmark_position(landmarks, landmark_id)
            if position:
                # Convert normalized coordinates to pixel values
                self.trail_points[point_name].append(position)
                
                # Keep trail at max length
                if len(self.trail_points[point_name]) > self.max_trail_length:
                    self.trail_points[point_name].pop(0)
    
    def draw_pose_trails(self, frame, frame_dims):
        """
        Draw trails of key points to visualize the swing path.
        
        Args:
            frame: The video frame to draw on
            frame_dims: (height, width) of the frame
            
        Returns:
            frame: The frame with trails drawn
        """
        height, width, _ = frame_dims
        
        for point_name, trail in self.trail_points.items():
            if len(trail) < 2:
                continue
                
            # Draw lines connecting trail points
            for i in range(1, len(trail)):
                start_pos = (int(trail[i-1][0] * width), int(trail[i-1][1] * height))
                end_pos = (int(trail[i][0] * width), int(trail[i][1] * height))
                
                # Make the trail fade out for older points
                alpha = i / len(trail)  # 0 to 1
                color = self.colors['trace']
                
                cv2.line(frame, start_pos, end_pos, color, thickness=2)
        
        return frame
    
    def draw_angle_measurements(self, frame, landmarks, frame_dims):
        """
        Draw angle measurements on the frame at key joints.
        
        Args:
            frame: The video frame to draw on
            landmarks: MediaPipe pose landmarks
            frame_dims: (height, width) of the frame
            
        Returns:
            frame: The frame with angle measurements drawn
        """
        if not landmarks:
            return frame
            
        height, width, _ = frame_dims
        
        # Define angles to visualize
        angles = [
            # Shoulder angle
            {
                'points': [
                    self.pose_tracker.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.pose_tracker.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.pose_tracker.mp_pose.PoseLandmark.RIGHT_HIP
                ],
                'label': 'Shoulder Angle'
            },
            # Hip angle
            {
                'points': [
                    self.pose_tracker.mp_pose.PoseLandmark.LEFT_HIP,
                    self.pose_tracker.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.pose_tracker.mp_pose.PoseLandmark.RIGHT_SHOULDER
                ],
                'label': 'Hip Angle'
            },
            # Right arm angle
            {
                'points': [
                    self.pose_tracker.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.pose_tracker.mp_pose.PoseLandmark.RIGHT_ELBOW,
                    self.pose_tracker.mp_pose.PoseLandmark.RIGHT_WRIST
                ],
                'label': 'R Arm Angle'
            }
        ]
        
        for angle_def in angles:
            # Calculate the angle
            angle_value = self.pose_tracker.get_angle(landmarks, *angle_def['points'])
            
            if angle_value is None:
                continue
                
            # Get the middle point (vertex of the angle)
            vertex = landmarks.landmark[angle_def['points'][1]]
            vertex_px = (int(vertex.x * width), int(vertex.y * height))
            
            # Draw the angle text
            text = f"{angle_def['label']}: {angle_value:.1f}°"
            cv2.putText(
                frame, text, 
                (vertex_px[0] + 10, vertex_px[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                self.colors['angle'], 2
            )
        
        return frame
    
    def draw_swing_phase(self, frame, current_phase, frame_dims):
        """
        Draw the current swing phase on the frame.
        
        Args:
            frame: The video frame to draw on
            current_phase: Current SwingPhase from the analyzer
            frame_dims: (height, width) of the frame
            
        Returns:
            frame: The frame with swing phase drawn
        """
        height, width, _ = frame_dims
        
        # Draw phase name at the top of the frame
        phase_name = self.swing_analyzer.get_phase_name(current_phase)
        cv2.putText(
            frame, f"Phase: {phase_name}", 
            (int(width * 0.05), int(height * 0.1)), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
            self.colors['text'], 2
        )
        
        return frame
    
    def draw_feedback(self, frame, issues, issues_details):
        """
        Draw feedback about detected issues on the frame.
        
        Args:
            frame: The video frame to draw on
            issues: Dictionary of detected issues (bool values)
            issues_details: Dictionary with details about each issue
            
        Returns:
            frame: The frame with feedback drawn
        """
        height, width, _ = frame.shape
        
        # Draw a transparent box at the bottom of the frame
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (0, int(height * 0.7)), 
            (width, height), 
            (0, 0, 0), 
            -1
        )
        
        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw title
        cv2.putText(
            frame, "Swing Feedback:", 
            (20, int(height * 0.75)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
            self.colors['text'], 2
        )
        
        # Draw issues (up to 3)
        y_offset = int(height * 0.8)
        issues_shown = 0
        
        for issue_name, is_detected in issues.items():
            if is_detected and issue_name in issues_details:
                display_name = issue_name.replace('_', ' ').title()
                
                if 'recommendation' in issues_details[issue_name]:
                    text = f"• {display_name}: {issues_details[issue_name]['recommendation']}"
                else:
                    text = f"• {display_name}"
                
                cv2.putText(
                    frame, text, 
                    (40, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    self.colors['error'], 2
                )
                
                y_offset += 30
                issues_shown += 1
                
                if issues_shown >= 3:
                    break
        
        # If no issues, show positive feedback
        if issues_shown == 0:
            cv2.putText(
                frame, "• Good swing! Keep it up.", 
                (40, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                self.colors['good'], 2
            )
        
        return frame
    
    def draw_comparison(self, frame, pro_landmarks, frame_dims, side_by_side=True):
        """
        Draw comparison between user's pose and professional golfer's pose.
        
        Args:
            frame: The video frame to draw on
            pro_landmarks: MediaPipe pose landmarks of professional golfer
            frame_dims: (height, width) of the frame
            side_by_side: Whether to draw side by side or overlay
            
        Returns:
            frame: The frame with comparison drawn
        """
        height, width, _ = frame_dims
        
        if side_by_side:
            # Create side-by-side comparison
            blank_side = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw pro pose on blank side
            blank_side = self.pose_tracker.draw_landmarks(blank_side, pro_landmarks)
            
            # Combine both images
            combined = np.hstack((frame, blank_side))
            
            # Add labels
            cv2.putText(
                combined, "Your Swing", 
                (int(width * 0.1), 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                self.colors['text'], 2
            )
            
            cv2.putText(
                combined, "Pro Reference", 
                (int(width * 1.1), 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                self.colors['text'], 2
            )
            
            # Resize to fit original dimensions
            combined = cv2.resize(combined, (width, height))
            return combined
        else:
            # Overlay mode
            # Draw pro pose with different color
            original_color = self.pose_tracker.landmark_color
            self.pose_tracker.landmark_color = (0, 255, 0)  # Green for pro
            
            # Create a copy to draw pro pose
            overlay = frame.copy()
            overlay = self.pose_tracker.draw_landmarks(overlay, pro_landmarks)
            
            # Apply transparency
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Add label
            cv2.putText(
                frame, "Pro Reference Overlay", 
                (int(width * 0.6), 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 255, 0), 2
            )
            
            # Reset color
            self.pose_tracker.landmark_color = original_color
            
            return frame
    
    def generate_report_image(self, issues, issues_details, swing_metrics=None):
        """
        Generate a summary image for the swing analysis report.
        
        Args:
            issues: Dictionary of detected issues (bool values)
            issues_details: Dictionary with details about each issue
            swing_metrics: Optional dictionary of swing metrics
            
        Returns:
            report_img: Image summarizing the report (numpy array)
        """
        # Create a blank white image
        width, height = 1000, 800
        report_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add title
        cv2.putText(
            report_img, "Golf Swing Analysis Summary", 
            (int(width * 0.2), 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
            (0, 0, 0), 3
        )
        
        # Draw line under title
        cv2.line(report_img, (50, 100), (width - 50, 100), (0, 0, 0), 2)
        
        # Add current date
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            report_img, f"Date: {timestamp}", 
            (50, 140), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
            (100, 100, 100), 2
        )
        
        # Add issues section title
        cv2.putText(
            report_img, "Detected Issues:", 
            (50, 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
            (0, 0, 0), 2
        )
        
        # Add issues and recommendations
        y_offset = 250
        issues_found = False
        
        for issue_name, is_detected in issues.items():
            if is_detected and issue_name in issues_details:
                issues_found = True
                display_name = issue_name.replace('_', ' ').title()
                
                # Draw issue name in red
                cv2.putText(
                    report_img, f"• {display_name}:", 
                    (70, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (0, 0, 255), 2
                )
                
                y_offset += 40
                
                # Draw recommendation
                if 'recommendation' in issues_details[issue_name]:
                    recommendation = issues_details[issue_name]['recommendation']
                    
                    # Split long text into multiple lines
                    words = recommendation.split()
                    lines = []
                    current_line = "  - "
                    
                    for word in words:
                        if len(current_line + word) < 80:
                            current_line += word + " "
                        else:
                            lines.append(current_line)
                            current_line = "    " + word + " "
                    
                    lines.append(current_line)
                    
                    for line in lines:
                        cv2.putText(
                            report_img, line, 
                            (90, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                            (0, 0, 0), 2
                        )
                        y_offset += 30
                    
                y_offset += 20
        
        # If no issues found
        if not issues_found:
            cv2.putText(
                report_img, "• No significant issues detected!", 
                (70, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                (0, 150, 0), 2
            )
            y_offset += 40
            
            cv2.putText(
                report_img, "  - Your swing demonstrates good fundamentals.", 
                (90, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 0, 0), 2
            )
            y_offset += 40
        
        # Add a line separator
        y_offset += 20
        cv2.line(report_img, (50, y_offset), (width - 50, y_offset), (200, 200, 200), 2)
        y_offset += 40
        
        # Add recommendations section
        cv2.putText(
            report_img, "Recommended Next Steps:", 
            (50, y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
            (0, 0, 0), 2
        )
        y_offset += 50
        
        # Add generic recommendations
        recommendations = [
            "Practice with a focus on the specific issues highlighted above",
            "Record your swing from multiple angles for more complete feedback",
            "Consider working with a coach to address specific technical issues",
            "Use alignment aids during practice to reinforce proper positions"
        ]
        
        for rec in recommendations:
            cv2.putText(
                report_img, f"• {rec}", 
                (70, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 0, 0), 2
            )
            y_offset += 40
        
        return report_img
    
    def reset(self):
        """
        Reset the visual feedback state.
        """
        # Clear all trails
        for key in self.trail_points:
            self.trail_points[key] = [] 
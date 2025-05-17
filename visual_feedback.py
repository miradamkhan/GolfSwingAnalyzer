import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

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
        
        # Position for the text
        y_offset = int(height * 0.2)
        line_height = 30
        
        # Check each issue and display it if detected
        for issue_name, is_detected in issues.items():
            if issue_name in issues_details:
                color = self.colors['error'] if is_detected else self.colors['good']
                
                status = "✗" if is_detected else "✓"
                
                # Format the issue name for display
                display_name = issue_name.replace('_', ' ').title()
                
                cv2.putText(
                    frame, f"{status} {display_name}", 
                    (int(width * 0.05), y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    color, 2
                )
                
                # If issue is detected, show recommendation
                if is_detected and 'recommendation' in issues_details[issue_name]:
                    cv2.putText(
                        frame, f"  {issues_details[issue_name]['recommendation']}", 
                        (int(width * 0.05), y_offset + line_height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        self.colors['text'], 1
                    )
                    y_offset += line_height * 2
                else:
                    y_offset += line_height
        
        return frame
    
    def draw_comparison(self, frame, pro_landmarks, frame_dims, side_by_side=True):
        """
        Draw a comparison between the user's pose and a professional golfer's pose.
        
        Args:
            frame: The user's video frame
            pro_landmarks: Pose landmarks from a professional golfer's swing
            frame_dims: (height, width) of the frame
            side_by_side: Whether to display pro pose side-by-side or overlay
            
        Returns:
            frame: The frame with the comparison visualization
        """
        if not pro_landmarks:
            return frame
            
        height, width, _ = frame_dims
        
        if side_by_side:
            # Create a side-by-side comparison
            comparison_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
            comparison_frame[:, :width] = frame
            
            # Create a blank frame for the pro pose
            pro_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw the pro pose landmarks
            self.pose_tracker.mp_drawing.draw_landmarks(
                pro_frame,
                pro_landmarks,
                self.pose_tracker.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.pose_tracker.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Add the pro frame to the right side
            comparison_frame[:, width:] = pro_frame
            
            # Add labels
            cv2.putText(
                comparison_frame, "Your Swing", 
                (int(width * 0.4), 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                self.colors['text'], 2
            )
            
            cv2.putText(
                comparison_frame, "Pro Swing", 
                (int(width * 1.4), 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                self.colors['text'], 2
            )
            
            return comparison_frame
        else:
            # Overlay the pro pose on the user's frame with transparency
            pro_frame = frame.copy()
            
            # Draw the pro pose landmarks
            self.pose_tracker.mp_drawing.draw_landmarks(
                pro_frame,
                pro_landmarks,
                self.pose_tracker.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.pose_tracker.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Blend the frames
            alpha = 0.6  # Transparency factor
            overlay_frame = cv2.addWeighted(frame, alpha, pro_frame, 1 - alpha, 0)
            
            # Add label
            cv2.putText(
                overlay_frame, "Pro Overlay", 
                (int(width * 0.7), 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                self.colors['text'], 2
            )
            
            return overlay_frame
    
    def generate_report_image(self, issues, issues_details, swing_metrics=None):
        """
        Generate a summary report image with swing metrics and improvement suggestions.
        
        Args:
            issues: Dictionary of detected issues
            issues_details: Dictionary with details about each issue
            swing_metrics: Optional additional metrics to display
            
        Returns:
            report_image: NumPy array containing the report visualization
        """
        # Create a figure and axis
        fig = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        
        # Turn off axis
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "Golf Swing Analysis Report", horizontalalignment='center',
                fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        # Detected issues and recommendations
        y_pos = 0.85
        ax.text(0.1, y_pos, "Swing Issues Detected:", fontsize=14, fontweight='bold')
        y_pos -= 0.05
        
        # Display issues
        has_issues = False
        for issue_name, is_detected in issues.items():
            if is_detected and issue_name in issues_details:
                has_issues = True
                display_name = issue_name.replace('_', ' ').title()
                
                # Issue name
                ax.text(0.1, y_pos, f"• {display_name}:", fontsize=12, fontweight='bold', color='red')
                y_pos -= 0.04
                
                # Recommendation
                if 'recommendation' in issues_details[issue_name]:
                    ax.text(0.15, y_pos, issues_details[issue_name]['recommendation'], fontsize=11)
                    y_pos -= 0.05
        
        if not has_issues:
            ax.text(0.1, y_pos, "No significant issues detected. Good job!", fontsize=12, color='green')
            y_pos -= 0.05
        
        # Add additional metrics if provided
        if swing_metrics:
            y_pos -= 0.05
            ax.text(0.1, y_pos, "Swing Metrics:", fontsize=14, fontweight='bold')
            y_pos -= 0.05
            
            for metric_name, value in swing_metrics.items():
                display_name = metric_name.replace('_', ' ').title()
                ax.text(0.1, y_pos, f"• {display_name}: {value}", fontsize=11)
                y_pos -= 0.04
        
        # Tips for improvement
        y_pos -= 0.05
        ax.text(0.1, y_pos, "Next Steps:", fontsize=14, fontweight='bold')
        y_pos -= 0.05
        
        # General tips
        general_tips = [
            "Practice the specific corrections mentioned above",
            "Record your swing from different angles for more complete feedback",
            "Consider using alignment aids during practice sessions",
            "Practice with a slower swing speed to reinforce proper mechanics"
        ]
        
        for tip in general_tips:
            ax.text(0.1, y_pos, f"• {tip}", fontsize=11)
            y_pos -= 0.04
        
        # Draw the canvas
        canvas.draw()
        
        # Convert to NumPy array
        report_image = np.array(canvas.buffer_rgba())
        
        # Convert RGBA to RGB (remove alpha channel)
        report_image = cv2.cvtColor(report_image, cv2.COLOR_RGBA2BGR)
        
        return report_image
    
    def reset(self):
        """Reset the visual feedback state."""
        self.trail_points = {key: [] for key in self.trail_points} 
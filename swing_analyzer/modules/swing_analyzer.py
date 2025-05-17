import numpy as np
from enum import Enum

class SwingPhase(Enum):
    """Enum representing different phases of a golf swing."""
    SETUP = 0
    BACKSWING = 1
    TOP_OF_SWING = 2
    DOWNSWING = 3
    IMPACT = 4
    FOLLOW_THROUGH = 5
    FINISH = 6
    UNKNOWN = 7

class SwingAnalyzer:
    def __init__(self, pose_tracker):
        """
        Initialize the swing analyzer.
        
        Args:
            pose_tracker: An instance of PoseTracker class
        """
        self.pose_tracker = pose_tracker
        self.metrics_history = []
        self.detected_phases = []
        self.current_phase = SwingPhase.UNKNOWN
        self.phase_frame_indices = {phase: [] for phase in SwingPhase}
        self.issues = {
            'poor_hip_rotation': False,
            'arm_overextension': False,
            'head_movement': False,
            'asymmetrical_shoulders': False
        }
        self.issues_details = {}
        
        # Reference values for ideal swing (these would be calibrated based on pro swings)
        self.reference_values = {
            'min_hip_rotation': 45.0,  # Minimum hip rotation angle in degrees
            'max_arm_extension': 0.4,  # Maximum normalized distance
            'max_head_movement': 0.03,  # Maximum normalized movement
            'max_shoulder_asymmetry': 10.0  # Maximum angle difference in degrees
        }
    
    def reset(self):
        """Reset the analyzer state for a new swing."""
        self.metrics_history = []
        self.detected_phases = []
        self.current_phase = SwingPhase.UNKNOWN
        self.phase_frame_indices = {phase: [] for phase in SwingPhase}
        self.issues = {
            'poor_hip_rotation': False,
            'arm_overextension': False,
            'head_movement': False,
            'asymmetrical_shoulders': False
        }
        self.issues_details = {}
    
    def update(self, landmarks, frame_index):
        """
        Update the analyzer with new pose landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            frame_index: Current frame index
            
        Returns:
            current_phase: The detected swing phase for this frame
        """
        if not landmarks:
            return SwingPhase.UNKNOWN
        
        # Calculate body metrics for this frame
        metrics = self.pose_tracker.calculate_body_metrics(landmarks)
        if not metrics:
            return SwingPhase.UNKNOWN
            
        self.metrics_history.append(metrics)
        
        # Detect swing phase
        self.current_phase = self._detect_swing_phase(metrics, frame_index)
        self.detected_phases.append(self.current_phase)
        
        # Record frame index for this phase
        self.phase_frame_indices[self.current_phase].append(frame_index)
        
        return self.current_phase
    
    def _detect_swing_phase(self, metrics, frame_index):
        """
        Detect the current swing phase based on pose metrics.
        This is a simplified implementation - a real system would use more robust detection.
        
        Args:
            metrics: Body metrics from pose_tracker
            frame_index: Current frame index
            
        Returns:
            phase: Detected SwingPhase
        """
        # Need at least a few frames of history to detect phases
        if len(self.metrics_history) < 5:
            return SwingPhase.SETUP
        
        # Get wrist positions (assuming right-handed golfer)
        # For left-handed, we would need to swap these
        if 'right_arm_extension' not in metrics or 'left_arm_extension' not in metrics:
            return SwingPhase.UNKNOWN
            
        # Simple phase detection based on right wrist height and movement
        # Getting the current and previous positions
        right_wrist_current = None
        right_wrist_prev = None
        
        for landmarks in self.pose_tracker.landmarks_history[-2:]:
            pos = self.pose_tracker.get_landmark_position(landmarks, self.pose_tracker.mp_pose.PoseLandmark.RIGHT_WRIST)
            if pos:
                if right_wrist_current is None:
                    right_wrist_current = pos
                else:
                    right_wrist_prev = pos
        
        if right_wrist_current is None or right_wrist_prev is None:
            return SwingPhase.UNKNOWN
        
        # Calculate vertical movement of wrist
        wrist_y_movement = right_wrist_prev[1] - right_wrist_current[1]
        
        # Simple heuristics for phase detection
        # In a real implementation, this would be more sophisticated using machine learning
        
        # Phase detection based on wrist movement and position
        if len(self.detected_phases) < 3:
            # Initial frames
            return SwingPhase.SETUP
            
        # Check previous phases to determine current phase
        prev_phases = self.detected_phases[-3:]
        
        if SwingPhase.FINISH in prev_phases:
            # Once we reach finish, stay there
            return SwingPhase.FINISH
            
        if SwingPhase.FOLLOW_THROUGH in prev_phases:
            # After follow through comes finish
            if right_wrist_current[1] < 0.4:  # Wrist is high
                return SwingPhase.FINISH
            return SwingPhase.FOLLOW_THROUGH
            
        if SwingPhase.IMPACT in prev_phases:
            # After impact comes follow through
            if wrist_y_movement < -0.01:  # Wrist moving up
                return SwingPhase.FOLLOW_THROUGH
            return SwingPhase.IMPACT
            
        if SwingPhase.DOWNSWING in prev_phases:
            # After downswing comes impact
            if abs(right_wrist_current[0] - 0.5) < 0.1:  # Wrist close to center x-position
                return SwingPhase.IMPACT
            return SwingPhase.DOWNSWING
            
        if SwingPhase.TOP_OF_SWING in prev_phases:
            # After top of swing comes downswing
            if wrist_y_movement > 0.01:  # Wrist moving down
                return SwingPhase.DOWNSWING
            return SwingPhase.TOP_OF_SWING
            
        if SwingPhase.BACKSWING in prev_phases:
            # After backswing comes top of swing
            if right_wrist_current[1] < 0.3 and abs(wrist_y_movement) < 0.01:  # Wrist high and relatively still
                return SwingPhase.TOP_OF_SWING
            if wrist_y_movement < -0.01:  # Wrist still moving up
                return SwingPhase.BACKSWING
            return SwingPhase.TOP_OF_SWING
            
        if SwingPhase.SETUP in prev_phases:
            # After setup comes backswing
            if wrist_y_movement < -0.01:  # Wrist moving up
                return SwingPhase.BACKSWING
            return SwingPhase.SETUP
            
        # Default case
        return SwingPhase.UNKNOWN
    
    def analyze_swing(self):
        """
        Analyze the completed swing to detect issues.
        
        Returns:
            issues: Dictionary of detected issues
            details: Dictionary with details about each issue
        """
        if not self.metrics_history or SwingPhase.IMPACT not in self.detected_phases:
            return self.issues, {"error": "Complete swing not detected"}
        
        # Find key phase indices
        backswing_indices = self.phase_frame_indices[SwingPhase.BACKSWING]
        top_indices = self.phase_frame_indices[SwingPhase.TOP_OF_SWING]
        downswing_indices = self.phase_frame_indices[SwingPhase.DOWNSWING]
        impact_indices = self.phase_frame_indices[SwingPhase.IMPACT]
        
        if not (backswing_indices and top_indices and downswing_indices and impact_indices):
            return self.issues, {"error": "One or more swing phases not detected"}
        
        # Get metrics at key phases
        backswing_metrics = self.metrics_history[backswing_indices[len(backswing_indices)//2]] if backswing_indices else {}
        top_metrics = self.metrics_history[top_indices[0]] if top_indices else {}
        downswing_metrics = self.metrics_history[downswing_indices[len(downswing_indices)//2]] if downswing_indices else {}
        impact_metrics = self.metrics_history[impact_indices[0]] if impact_indices else {}
        
        # Check for hip rotation issues
        if 'hip_angle' in top_metrics and 'hip_angle' in impact_metrics:
            hip_rotation_range = abs(top_metrics['hip_angle'] - impact_metrics['hip_angle'])
            self.issues['poor_hip_rotation'] = hip_rotation_range < self.reference_values['min_hip_rotation']
            self.issues_details['poor_hip_rotation'] = {
                'detected': hip_rotation_range,
                'recommendation': f"Increase hip rotation (current: {hip_rotation_range:.1f}°, target: >{self.reference_values['min_hip_rotation']}°)"
            }
        
        # Check for arm overextension
        if 'right_arm_extension' in downswing_metrics:
            arm_extension = downswing_metrics['right_arm_extension']
            self.issues['arm_overextension'] = arm_extension > self.reference_values['max_arm_extension']
            self.issues_details['arm_overextension'] = {
                'detected': arm_extension,
                'recommendation': f"Maintain better control of arm extension (current: {arm_extension:.2f}, target: <{self.reference_values['max_arm_extension']:.2f})"
            }
        
        # Check for head movement
        head_positions = [m.get('head_position') for m in self.metrics_history if 'head_position' in m]
        if len(head_positions) > 5:
            head_x_positions = [pos[0] for pos in head_positions if pos]
            head_y_positions = [pos[1] for pos in head_positions if pos]
            
            head_x_movement = max(head_x_positions) - min(head_x_positions)
            head_y_movement = max(head_y_positions) - min(head_y_positions)
            total_head_movement = np.sqrt(head_x_movement**2 + head_y_movement**2)
            
            self.issues['head_movement'] = total_head_movement > self.reference_values['max_head_movement']
            self.issues_details['head_movement'] = {
                'detected': total_head_movement,
                'recommendation': f"Keep your head more stable during the swing (movement: {total_head_movement:.3f}, target: <{self.reference_values['max_head_movement']:.3f})"
            }
        
        # Check for asymmetrical shoulders
        if 'shoulder_angle' in backswing_metrics and 'shoulder_angle' in impact_metrics:
            shoulder_angle_diff = abs(backswing_metrics['shoulder_angle'] - impact_metrics['shoulder_angle'])
            self.issues['asymmetrical_shoulders'] = shoulder_angle_diff > self.reference_values['max_shoulder_asymmetry']
            self.issues_details['asymmetrical_shoulders'] = {
                'detected': shoulder_angle_diff,
                'recommendation': f"Work on shoulder symmetry through the swing (current difference: {shoulder_angle_diff:.1f}°, target: <{self.reference_values['max_shoulder_asymmetry']}°)"
            }
        
        return self.issues, self.issues_details
    
    def get_phase_name(self, phase):
        """Convert a SwingPhase enum to a readable string."""
        phase_names = {
            SwingPhase.SETUP: "Setup",
            SwingPhase.BACKSWING: "Backswing",
            SwingPhase.TOP_OF_SWING: "Top of Swing",
            SwingPhase.DOWNSWING: "Downswing",
            SwingPhase.IMPACT: "Impact",
            SwingPhase.FOLLOW_THROUGH: "Follow Through",
            SwingPhase.FINISH: "Finish",
            SwingPhase.UNKNOWN: "Unknown"
        }
        return phase_names.get(phase, "Unknown") 
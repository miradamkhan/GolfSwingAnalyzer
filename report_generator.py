import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class ReportGenerator:
    def __init__(self, pose_tracker, swing_analyzer):
        """
        Initialize the report generator.
        
        Args:
            pose_tracker: An instance of PoseTracker class
            swing_analyzer: An instance of SwingAnalyzer class
        """
        self.pose_tracker = pose_tracker
        self.swing_analyzer = swing_analyzer
        
        # Define standard improvement suggestions for common issues
        self.improvement_suggestions = {
            'poor_hip_rotation': [
                "Focus on rotating your hips more during the backswing",
                "Practice with alignment sticks to visualize proper hip turn",
                "Work on hip flexibility exercises to increase rotation capacity"
            ],
            'arm_overextension': [
                "Maintain better arm connection with your body throughout the swing",
                "Practice with a towel under your armpits to prevent overextension",
                "Focus on keeping a consistent swing radius"
            ],
            'head_movement': [
                "Practice keeping your head still and focused on the ball",
                "Try the drill of placing a ball under your chin during practice swings",
                "Maintain your spine angle throughout the swing"
            ],
            'asymmetrical_shoulders': [
                "Work on proper shoulder turn during the backswing",
                "Practice with alignment aids to get feedback on shoulder position",
                "Focus on maintaining equal shoulder tilt throughout the swing"
            ]
        }
    
    def generate_text_report(self, issues, issues_details, video_path=None):
        """
        Generate a text-based performance report.
        
        Args:
            issues: Dictionary of detected issues (bool values)
            issues_details: Dictionary with details about each issue
            video_path: Optional path to the analyzed video
            
        Returns:
            report_text: String containing the full report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            "==================================",
            "       GOLF SWING ANALYSIS REPORT",
            "==================================",
            f"Date: {timestamp}",
        ]
        
        if video_path:
            report_lines.append(f"Video analyzed: {os.path.basename(video_path)}")
            
        report_lines.extend([
            "==================================",
            "\nSWING ISSUES SUMMARY:",
            "----------------------------------"
        ])
        
        # Add detected issues and recommendations
        detected_count = 0
        for issue_name, is_detected in issues.items():
            if is_detected and issue_name in issues_details:
                detected_count += 1
                display_name = issue_name.replace('_', ' ').title()
                
                report_lines.append(f"\n{detected_count}. {display_name}:")
                
                # Add the detected value and recommendation
                if 'detected' in issues_details[issue_name] and 'recommendation' in issues_details[issue_name]:
                    report_lines.append(f"   - {issues_details[issue_name]['recommendation']}")
                
                # Add specific improvement suggestions
                if issue_name in self.improvement_suggestions:
                    report_lines.append("   - Drills to improve:")
                    for i, suggestion in enumerate(self.improvement_suggestions[issue_name][:2], 1):
                        report_lines.append(f"     {i}. {suggestion}")
        
        if detected_count == 0:
            report_lines.append("\nNo significant issues detected. Your swing looks good!")
            
        # Add a general improvement section
        report_lines.extend([
            "\n==================================",
            "GENERAL IMPROVEMENT SUGGESTIONS:",
            "----------------------------------",
            "• Record your swing from multiple angles for more complete feedback",
            "• Practice with a focus on one specific element at a time",
            "• Consider working with a coach to address specific technical issues",
            "• Use alignment aids during practice to reinforce proper positions",
            "==================================",
        ])
        
        # Compile the full report
        report_text = "\n".join(report_lines)
        return report_text
    
    def generate_metric_chart(self, metrics_history, metric_name, ideal_range=None):
        """
        Generate a chart visualizing a specific metric over time.
        
        Args:
            metrics_history: List of metrics dictionaries from each frame
            metric_name: Name of the metric to chart
            ideal_range: Optional (min, max) tuple indicating ideal range
            
        Returns:
            chart_image: NumPy array containing the chart visualization
        """
        # Extract the metric values from history
        values = []
        for metrics in metrics_history:
            if metric_name in metrics:
                values.append(metrics[metric_name])
        
        if not values:
            return None
            
        # Create the figure and plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the metric values
        frames = range(len(values))
        ax.plot(frames, values, marker='o', linestyle='-', linewidth=2, markersize=4)
        
        # Add ideal range if provided
        if ideal_range:
            min_val, max_val = ideal_range
            ax.axhspan(min_val, max_val, alpha=0.2, color='green')
            
        # Add labels and title
        ax.set_xlabel('Frame')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} Throughout Swing')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout
        fig.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        chart_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        chart_image = chart_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close the figure to free memory
        plt.close(fig)
        
        return chart_image
    
    def save_report(self, report_text, output_dir, video_path=None, charts=None):
        """
        Save the report to a file.
        
        Args:
            report_text: Text report to save
            output_dir: Directory to save the report in
            video_path: Optional path to the analyzed video
            charts: Optional list of chart images to save
            
        Returns:
            report_path: Path to the saved report
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if video_path:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            report_filename = f"swing_report_{video_name}_{timestamp}.txt"
        else:
            report_filename = f"swing_report_{timestamp}.txt"
            
        report_path = os.path.join(output_dir, report_filename)
        
        # Save the text report
        with open(report_path, 'w') as f:
            f.write(report_text)
            
        # Save charts if provided
        if charts:
            for i, chart in enumerate(charts):
                chart_filename = f"chart_{i+1}_{timestamp}.png"
                chart_path = os.path.join(output_dir, chart_filename)
                cv2.imwrite(chart_path, cv2.cvtColor(chart, cv2.COLOR_RGB2BGR))
                
        return report_path
    
    def extract_key_metrics(self, metrics_history):
        """
        Extract key metrics to include in the report.
        
        Args:
            metrics_history: List of metrics dictionaries from each frame
            
        Returns:
            key_metrics: Dictionary of key metrics
        """
        if not metrics_history:
            return {}
            
        key_metrics = {}
        
        # Extract hip angle at key points
        hip_angles = [m.get('hip_angle') for m in metrics_history if 'hip_angle' in m]
        if hip_angles:
            key_metrics['max_hip_angle'] = max(hip_angles)
            key_metrics['hip_rotation_range'] = max(hip_angles) - min(hip_angles)
            
        # Extract shoulder angle at key points
        shoulder_angles = [m.get('shoulder_angle') for m in metrics_history if 'shoulder_angle' in m]
        if shoulder_angles:
            key_metrics['max_shoulder_angle'] = max(shoulder_angles)
            key_metrics['shoulder_rotation_range'] = max(shoulder_angles) - min(shoulder_angles)
            
        # Extract arm extension metrics
        right_arm_extensions = [m.get('right_arm_extension') for m in metrics_history if 'right_arm_extension' in m]
        if right_arm_extensions:
            key_metrics['max_arm_extension'] = max(right_arm_extensions)
            
        # Calculate swing tempo (ratio of backswing to downswing time)
        # This requires frame counts for different phases
        backswing_frames = len(self.swing_analyzer.phase_frame_indices[self.swing_analyzer.SwingPhase.BACKSWING])
        downswing_frames = len(self.swing_analyzer.phase_frame_indices[self.swing_analyzer.SwingPhase.DOWNSWING])
        
        if backswing_frames > 0 and downswing_frames > 0:
            key_metrics['swing_tempo'] = round(backswing_frames / downswing_frames, 1)
            
        return key_metrics
    
    def generate_full_report(self, issues, issues_details, metrics_history, video_path=None, output_dir="reports"):
        """
        Generate and save a complete report with text and visualizations.
        
        Args:
            issues: Dictionary of detected issues
            issues_details: Dictionary with details about each issue
            metrics_history: List of metrics dictionaries from each frame
            video_path: Optional path to the analyzed video
            output_dir: Directory to save the report in
            
        Returns:
            report_path: Path to the saved report
        """
        # Extract key metrics for the report
        key_metrics = self.extract_key_metrics(metrics_history)
        
        # Generate text report
        report_text = self.generate_text_report(issues, issues_details, video_path)
        
        # Add key metrics section to the report
        if key_metrics:
            metrics_section = [
                "\n==================================",
                "KEY SWING METRICS:",
                "----------------------------------"
            ]
            
            for name, value in key_metrics.items():
                display_name = name.replace('_', ' ').title()
                metrics_section.append(f"• {display_name}: {value:.1f}")
                
            # Add information about swing tempo
            if 'swing_tempo' in key_metrics:
                tempo = key_metrics['swing_tempo']
                if tempo > 3.5:
                    metrics_section.append("  - Your swing tempo is relatively slow")
                elif tempo < 2.5:
                    metrics_section.append("  - Your swing tempo is relatively fast")
                else:
                    metrics_section.append("  - Your swing tempo is balanced (ideal is around 3:1)")
                    
            report_text += "\n" + "\n".join(metrics_section)
        
        # Generate charts
        charts = []
        
        if metrics_history:
            # Hip angle chart
            if any('hip_angle' in m for m in metrics_history):
                hip_chart = self.generate_metric_chart(metrics_history, 'hip_angle', ideal_range=(40, 90))
                if hip_chart is not None:
                    charts.append(hip_chart)
            
            # Arm extension chart
            if any('right_arm_extension' in m for m in metrics_history):
                arm_chart = self.generate_metric_chart(metrics_history, 'right_arm_extension', ideal_range=(0.3, 0.4))
                if arm_chart is not None:
                    charts.append(arm_chart)
        
        # Save the report
        return self.save_report(report_text, output_dir, video_path, charts) 
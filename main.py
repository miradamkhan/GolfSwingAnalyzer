import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm

from swing_analyzer.modules.pose_tracker import PoseTracker
from swing_analyzer.modules.swing_analyzer import SwingAnalyzer, SwingPhase
from swing_analyzer.modules.visual_feedback import VisualFeedback
from swing_analyzer.modules.report_generator import ReportGenerator

def process_video(video_path, output_path=None, show_preview=True, comparison_video=None):
    """
    Process a golf swing video to analyze the form.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the output video (optional)
        show_preview: Whether to show a preview window during processing
        comparison_video: Path to a professional swing video for comparison (optional)
    """
    # Create the output directory if needed
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Initialize components
    pose_tracker = PoseTracker()
    swing_analyzer = SwingAnalyzer(pose_tracker)
    visual_feedback = VisualFeedback(pose_tracker, swing_analyzer)
    report_generator = ReportGenerator(pose_tracker, swing_analyzer)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load comparison video if provided
    pro_pose_frames = []
    if comparison_video:
        comp_video = cv2.VideoCapture(comparison_video)
        while True:
            ret, frame = comp_video.read()
            if not ret:
                break
                
            # Process frame with pose tracker to get landmarks
            _, landmarks = pose_tracker.process_frame(frame)
            if landmarks:
                pro_pose_frames.append(landmarks)
        
        comp_video.release()
        
        if not pro_pose_frames:
            print("Warning: Could not extract pose landmarks from comparison video.")
    
    # Create video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process the video frame by frame
    frame_index = 0
    
    # Progress bar
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Process frame with pose tracker
        processed_frame, landmarks = pose_tracker.process_frame(frame)
        
        # Update swing analyzer with new landmarks
        current_phase = swing_analyzer.update(landmarks, frame_index)
        
        # Add current frame's landmarks to trail
        visual_feedback.add_trail_points(landmarks)
        
        # Create a clean copy for visualization
        display_frame = processed_frame.copy()
        
        # Draw swing trails
        display_frame = visual_feedback.draw_pose_trails(display_frame, (height, width, 3))
        
        # Draw angle measurements
        display_frame = visual_feedback.draw_angle_measurements(display_frame, landmarks, (height, width, 3))
        
        # Draw current swing phase
        display_frame = visual_feedback.draw_swing_phase(display_frame, current_phase, (height, width, 3))
        
        # If we're in analysis phase, show feedback
        if frame_index > 10:  # Skip first few frames to get enough data
            issues, issues_details = swing_analyzer.analyze_swing()
            display_frame = visual_feedback.draw_feedback(display_frame, issues, issues_details)
        
        # Compare with pro pose if available
        if pro_pose_frames and len(pro_pose_frames) > 0:
            # Select a pro pose frame based on current phase
            # This is a simplified matching - a real system would need more sophisticated alignment
            pro_index = min(len(pro_pose_frames) - 1, 
                           int(frame_index / total_frames * len(pro_pose_frames)))
            
            pro_landmarks = pro_pose_frames[pro_index]
            display_frame = visual_feedback.draw_comparison(display_frame, pro_landmarks, 
                                                          (height, width, 3), side_by_side=False)
        
        # Write the frame to output video
        if output_path:
            out.write(display_frame)
        
        # Show preview if enabled
        if show_preview:
            cv2.imshow('Golf Swing Analysis', display_frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_index += 1
        pbar.update(1)
    
    pbar.close()
    
    # Release video resources
    video.release()
    if output_path:
        out.release()
    
    if show_preview:
        cv2.destroyAllWindows()
    
    # Generate final report
    if frame_index > 10:  # Only if we have processed enough frames
        issues, issues_details = swing_analyzer.analyze_swing()
        report_path = report_generator.generate_full_report(
            issues, 
            issues_details, 
            swing_analyzer.metrics_history,
            video_path=video_path
        )
        print(f"Analysis report saved to: {report_path}")
        
        # Generate and display a report image
        report_image = visual_feedback.generate_report_image(issues, issues_details)
        if report_image is not None:
            cv2.imshow('Swing Analysis Report', report_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save the report image
            report_image_path = os.path.join(os.path.dirname(report_path), 
                                           os.path.basename(report_path).replace('.txt', '.png'))
            cv2.imwrite(report_image_path, report_image)
            print(f"Report image saved to: {report_image_path}")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Golf Swing Analyzer')
    parser.add_argument('--video', '-v', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='Path to save output video file')
    parser.add_argument('--no-preview', action='store_true',
                      help='Disable preview window during processing')
    parser.add_argument('--comparison', '-c', type=str, default=None,
                      help='Path to professional swing video for comparison')
    
    args = parser.parse_args()
    
    # Process the video
    process_video(
        args.video, 
        output_path=args.output,
        show_preview=not args.no_preview,
        comparison_video=args.comparison
    )

if __name__ == "__main__":
    main() 
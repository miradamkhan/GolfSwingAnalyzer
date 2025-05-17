import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from PIL import Image
import io
import base64
import mimetypes
import pathlib

from swing_analyzer.modules.pose_tracker import PoseTracker
from swing_analyzer.modules.swing_analyzer import SwingAnalyzer, SwingPhase
from swing_analyzer.modules.visual_feedback import VisualFeedback
from swing_analyzer.modules.report_generator import ReportGenerator

def process_uploaded_video(video_file, comparison_file=None):
    """
    Process an uploaded video file.
    
    Args:
        video_file: The uploaded video file
        comparison_file: Optional professional swing video for comparison
        
    Returns:
        output_video_path: Path to processed video
        comparison_video_path: Path to side-by-side comparison (if comparison provided)
        report_path: Path to generated report
        report_image: Generated report image
    """
    # Create temporary files for processing
    temp_dir = tempfile.mkdtemp()
    
    # Save uploaded files to temporary location
    input_path = os.path.join(temp_dir, "input_video.mp4")
    with open(input_path, "wb") as f:
        f.write(video_file.read())
    
    comparison_path = None
    if comparison_file:
        comparison_path = os.path.join(temp_dir, "comparison_video.mp4")
        with open(comparison_path, "wb") as f:
            f.write(comparison_file.read())
    
    # Set output paths
    output_video_path = os.path.join(temp_dir, "output_video.mp4")
    side_by_side_path = os.path.join(temp_dir, "side_by_side.mp4") if comparison_path else None
    
    # Initialize components
    pose_tracker = PoseTracker()
    swing_analyzer = SwingAnalyzer(pose_tracker)
    visual_feedback = VisualFeedback(pose_tracker, swing_analyzer)
    report_generator = ReportGenerator(pose_tracker, swing_analyzer)
    
    # Open the video file
    video = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load comparison video if provided
    pro_pose_frames = []
    if comparison_path:
        comp_video = cv2.VideoCapture(comparison_path)
        while True:
            ret, frame = comp_video.read()
            if not ret:
                break
                
            # Process frame with pose tracker to get landmarks
            _, landmarks = pose_tracker.process_frame(frame)
            if landmarks:
                pro_pose_frames.append(landmarks)
        
        comp_video.release()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process the video frame by frame
    frame_index = 0
    
    # Create a progress bar in Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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
            pro_index = min(len(pro_pose_frames) - 1, 
                           int(frame_index / total_frames * len(pro_pose_frames)))
            
            pro_landmarks = pro_pose_frames[pro_index]
            display_frame = visual_feedback.draw_comparison(display_frame, pro_landmarks, 
                                                          (height, width, 3), side_by_side=False)
        
        # Write the frame to output video
        out.write(display_frame)
        
        frame_index += 1
        
        # Update progress
        progress = int(frame_index / total_frames * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_index}/{total_frames} ({progress}%)")
    
    # Release video resources
    video.release()
    out.release()
    
    # Generate final report
    report_path = None
    report_image = None
    
    if frame_index > 10:  # Only if we have processed enough frames
        issues, issues_details = swing_analyzer.analyze_swing()
        report_path = report_generator.generate_full_report(
            issues, 
            issues_details, 
            swing_analyzer.metrics_history,
            video_path=input_path,
            output_dir=temp_dir
        )
        
        # Generate a report image
        report_image = visual_feedback.generate_report_image(issues, issues_details)
        
        if report_image is not None:
            # Save the report image
            report_image_path = os.path.join(temp_dir, "report_image.png")
            cv2.imwrite(report_image_path, report_image)
    
    status_text.text("Processing complete!")
    
    # If we have a comparison video, create a side-by-side version too
    if comparison_path and pro_pose_frames:
        # Create video writer for side-by-side comparison
        side_by_side_writer = cv2.VideoWriter(side_by_side_path, fourcc, fps, (width*2, height))
        
        # Reopen the input video
        input_video = cv2.VideoCapture(input_path)
        comp_video = cv2.VideoCapture(comparison_path)
        
        # Process frames for side-by-side
        while True:
            ret1, frame1 = input_video.read()
            ret2, frame2 = comp_video.read()
            
            if not ret1 or not ret2:
                break
                
            # Resize frames if they're different sizes
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
                
            # Combine frames side by side
            combined = np.hstack((frame1, frame2))
            
            # Add labels
            cv2.putText(combined, "Your Swing", (int(width * 0.1), 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(combined, "Pro Reference", (int(width * 1.1), 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Write to video
            side_by_side_writer.write(combined)
            
        # Release resources
        input_video.release()
        comp_video.release()
        side_by_side_writer.release()
    
    return output_video_path, side_by_side_path, report_path, report_image

def create_video_preview(video_path, preview_path=None, frames_to_extract=10):
    """
    Create a video preview GIF from the first few frames of a video.
    
    Args:
        video_path: Path to the video file
        preview_path: Path to save the preview image (if None, a temp path is created)
        frames_to_extract: Number of frames to extract for the preview
        
    Returns:
        preview_path: Path to the created preview image
    """
    if preview_path is None:
        preview_path = os.path.splitext(video_path)[0] + "_preview.gif"
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frames to skip to get a representative sample
    frame_skip = max(1, total_frames // frames_to_extract)
    
    # Collect frames
    frames = []
    for i in range(0, total_frames, frame_skip):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if ret:
            # Resize frame to make GIF smaller
            frame = cv2.resize(frame, (width // 2, height // 2))
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            
        if len(frames) >= frames_to_extract:
            break
    
    # Save as GIF
    if frames:
        frames[0].save(
            preview_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=200,  # milliseconds per frame
            loop=0
        )
    
    # Release video
    video.release()
    
    return preview_path

def display_video_player(video_path, width=700):
    """
    Display a video player in Streamlit with a fallback to download if playback fails.
    
    Args:
        video_path: Path to the video file
        width: Width of the video player
    """
    # Create a preview GIF
    preview_path = create_video_preview(video_path)
    
    # Get the video file size in MB
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    
    # Display the video using Streamlit's native video player
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    
    st.video(video_bytes)
    
    # Display a preview GIF as a fallback
    st.markdown("### Video Preview (GIF)")
    st.markdown("If the video above doesn't play correctly, you can view this animated preview or download the video.")
    st.image(preview_path, width=width)
    
    # Add download button
    st.download_button(
        label=f"Download Video ({file_size_mb:.1f} MB)",
        data=video_bytes,
        file_name=os.path.basename(video_path),
        mime="video/mp4"
    )
    
    video_file.close()

def main():
    st.set_page_config(
        page_title="Golf Swing Analyzer",
        page_icon="🏌️",
        layout="wide",
    )
    
    st.title("Golf Swing Analyzer")
    st.write("Upload a video of your golf swing for analysis")
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Your Swing Video")
        video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    
    with col2:
        st.subheader("Upload Pro Swing (Optional)")
        st.write("For side-by-side comparison")
        comparison_file = st.file_uploader("Choose a video file for comparison", type=["mp4", "mov", "avi"])
    
    if video_file:
        st.write("Video uploaded successfully!")
        
        # Process button
        if st.button("Analyze Swing"):
            with st.spinner("Processing video..."):
                output_video, side_by_side, report_path, report_image = process_uploaded_video(
                    video_file, 
                    comparison_file
                )
                
                # Display results
                if output_video and os.path.exists(output_video):
                    st.subheader("Analysis Results")
                    
                    # Create tabs for different views
                    result_tabs = st.tabs(["Analyzed Swing", "Side-by-Side Comparison" if side_by_side else "Detailed Report", "Summary Report"])
                    
                    with result_tabs[0]:
                        # Display the processed video using the enhanced video player
                        st.markdown("### Your Swing with Analysis")
                        display_video_player(output_video)
                    
                    with result_tabs[1]:
                        if side_by_side and os.path.exists(side_by_side):
                            st.markdown("### Your Swing vs Pro Reference")
                            display_video_player(side_by_side, width=900)
                        else:
                            # Display text report if available
                            if report_path and os.path.exists(report_path):
                                st.markdown("### Detailed Analysis Report")
                                
                                with open(report_path, 'r') as f:
                                    report_text = f.read()
                                    
                                st.text_area("Full Report", report_text, height=400)
                    
                    with result_tabs[2 if side_by_side else 1]:
                        # Display report image if available
                        if report_image is not None:
                            st.markdown("### Swing Analysis Summary")
                            
                            # Convert OpenCV image to format Streamlit can display
                            report_image_rgb = cv2.cvtColor(report_image, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(report_image_rgb)
                            
                            # Display image
                            st.image(pil_image, use_column_width=True)
                    
                    # Add download section for all results
                    st.subheader("Download Results")
                    col1, col2 = st.columns(2)
                    
                    # Only show report download button since video downloads are now included with each video
                    if report_path and os.path.exists(report_path):
                        with col1:
                            with open(report_path, "rb") as file:
                                st.download_button(
                                    label="Download Analysis Report",
                                    data=file,
                                    file_name="golf_swing_analysis_report.txt",
                                    mime="text/plain"
                                )
                    
                    # Add a button to save all results to a zip file for easy downloading
                    with col2:
                        st.info("Individual video download options are available in each tab above.")
    
    # Additional information
    with st.expander("How to use this tool"):
        st.write("""
        1. Upload a video of your golf swing (side view preferred)
        2. Optionally upload a professional swing video for comparison
        3. Click 'Analyze Swing' to process the video
        4. Review the analysis results and suggestions for improvement
        5. Download the report and analyzed video
        """)
    
    with st.expander("About"):
        st.write("""
        This tool uses computer vision and pose estimation to analyze golf swings.
        
        Features:
        - Pose tracking with MediaPipe
        - Swing phase detection (setup, backswing, downswing, etc.)
        - Detection of common swing issues
        - Visual feedback with joint angles and pose trails
        - Comparison with professional swing (if provided)
        - Detailed performance report with suggestions
        """)
    
    st.sidebar.title("Golf Swing Analyzer")
    st.sidebar.image("https://i.imgur.com/8FOcmwL.png", use_column_width=True)
    st.sidebar.markdown("---")
    st.sidebar.info("Upload a video of your golf swing to receive instant feedback and analysis.")
    st.sidebar.markdown("---")
    st.sidebar.success("Golf swing technology - Helping improve your game through data-driven feedback.")

if __name__ == "__main__":
    main() 
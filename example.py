#!/usr/bin/env python
"""
Example script for Golf Swing Analyzer
This script demonstrates how to use the Golf Swing Analyzer with a sample video.
"""
import os
import sys
import argparse

# Add the current directory to the path to ensure modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from swing_analyzer.modules.pose_tracker import PoseTracker
from swing_analyzer.modules.swing_analyzer import SwingAnalyzer
from swing_analyzer.modules.visual_feedback import VisualFeedback
from swing_analyzer.modules.report_generator import ReportGenerator

def download_sample_video():
    """
    Download a sample golf swing video if one doesn't exist.
    This function will create a 'samples' directory and download a sample video.
    
    Returns:
        str: Path to the sample video
    """
    import urllib.request
    
    # Create samples directory if it doesn't exist
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Path to save the sample video
    sample_path = os.path.join(samples_dir, 'sample_golf_swing.mp4')
    
    # Only download if the file doesn't exist
    if not os.path.exists(sample_path):
        print("Downloading sample golf swing video...")
        # You would need to replace this URL with a real golf swing video URL
        sample_video_url = "https://example.com/sample_golf_swing.mp4"
        
        try:
            urllib.request.urlretrieve(sample_video_url, sample_path)
            print(f"Sample video downloaded to {sample_path}")
        except Exception as e:
            print(f"Failed to download sample video: {e}")
            print("Please provide your own golf swing video using the --video argument.")
            return None
    else:
        print(f"Using existing sample video at {sample_path}")
        
    return sample_path

def run_example(video_path=None, output_dir=None):
    """
    Run the golf swing analyzer on a sample or provided video.
    
    Args:
        video_path: Path to the input video file, or None to use sample
        output_dir: Directory to save output files, or None for default
    """
    # Use sample video if none provided
    if video_path is None:
        video_path = download_sample_video()
        
        # If we still don't have a video, exit
        if video_path is None:
            print("No video available for analysis. Exiting.")
            return
    
    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)
    
    # Output video path
    output_video_path = os.path.join(output_dir, f"analyzed_{os.path.basename(video_path)}")
    
    print(f"Analyzing golf swing from {video_path}")
    print(f"Results will be saved to {output_dir}")
    
    # Run the analysis using main.py
    from main import process_video
    
    process_video(
        video_path=video_path,
        output_path=output_video_path,
        show_preview=True,
        comparison_video=None  # No comparison video in this example
    )
    
    print("\nAnalysis complete!")
    print(f"Analyzed video saved to: {output_video_path}")
    print(f"Check the 'reports' directory for the analysis report.")

def main():
    """Parse arguments and run the example."""
    parser = argparse.ArgumentParser(description='Golf Swing Analyzer Example')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to input video file (optional, uses sample if not provided)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output files (optional)')
    
    args = parser.parse_args()
    
    run_example(args.video, args.output_dir)

if __name__ == "__main__":
    main() 
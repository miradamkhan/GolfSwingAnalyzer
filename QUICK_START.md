# Golf Swing Analyzer Quick Start Guide

This guide will help you get up and running with the Golf Swing Analyzer quickly.

## Prerequisites

- Python 3.8+ installed on your system
- A video of your golf swing (preferably from a side angle)
- Basic familiarity with the command line

## Installation

1. Clone or download this repository
2. Open a terminal or command prompt
3. Navigate to the project directory
4. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```
5. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
6. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Analysis

For the quickest analysis of your golf swing:

1. Make sure your golf swing video is accessible on your computer
2. Run the example script with your video file:
   ```bash
   python example.py --video path/to/your/swing.mp4
   ```
3. View the real-time analysis (press 'q' to quit the preview)
4. Check the `reports` directory for your analysis report
5. Review the annotated video in the `output` directory

## Using the Web Interface

For a more user-friendly experience:

1. Start the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser to http://localhost:8501
3. Upload your golf swing video using the file uploader
4. Optionally upload a professional swing video for comparison
5. Click "Analyze Swing" and wait for the processing to complete
6. Review the analysis results and download the report

## Getting Better Results

For the best analysis results:

1. Record your swing from a side angle, with your entire body visible
2. Ensure good lighting conditions
3. Wear clothing that contrasts with the background
4. Position the camera at hip height for the best angle
5. Include your entire swing, from setup through follow-through
6. Try to minimize background distractions or other people in the frame

## Troubleshooting

If you encounter issues:

- **No pose detection**: Try recording in better lighting or with clearer contrast between you and the background
- **Incorrect swing phase detection**: Ensure your entire swing is captured from setup to finish
- **Application crashes**: Check that your video format is supported (MP4, MOV, or AVI recommended)
- **Missing dependencies**: Run `pip install -r requirements.txt` again to ensure all dependencies are installed

## Next Steps

- Try recording from different angles for more comprehensive feedback
- Compare your swing with professional examples
- Track your progress over time by saving reports and videos
- Adjust your practice routine based on the identified issues

For more detailed information, refer to the full [README.md](README.md). 
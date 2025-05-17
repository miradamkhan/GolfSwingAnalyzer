# Golf Swing Analyzer

A Python-based tool that uses pose estimation to analyze golf swings and provide feedback for improvement.

## Features

- Video capture and processing with OpenCV
- Pose estimation using MediaPipe
- Swing phase detection (backswing, downswing, follow-through)
- Swing analysis for common issues:
  - Poor hip rotation
  - Overextension of arms
  - Head movement
  - Asymmetrical shoulder angles
- Visual feedback with overlay of key measurements
- Performance reports with suggestions for improvement
- Side-by-side comparison with professional swing examples
- Streamlit web interface for easy use

## Project Structure

```
golf_swing_analyzer/
├── main.py              # Command line application
├── app.py               # Streamlit web application
├── requirements.txt     # Required dependencies
├── reports/             # Generated reports directory
└── swing_analyzer/      # Main package
    ├── __init__.py
    └── modules/
        ├── __init__.py
        ├── pose_tracker.py       # Pose estimation module
        ├── swing_analyzer.py     # Swing phase detection and analysis
        ├── visual_feedback.py    # Visual overlay and feedback
        └── report_generator.py   # Performance report generation
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/golf-swing-analyzer.git
cd golf-swing-analyzer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
python main.py --video path/to/your/swing.mp4 --output path/to/save/analysis.mp4
```

Options:
- `--video, -v`: Path to input video file (required)
- `--output, -o`: Path to save output video file (optional)
- `--no-preview`: Disable preview window during processing
- `--comparison, -c`: Path to professional swing video for comparison (optional)

### Web Interface

```bash
streamlit run app.py
```

Then open your browser to http://localhost:8501 and use the web interface to:
1. Upload your golf swing video
2. Optionally upload a professional swing video for comparison
3. Process the video and view analysis results
4. Download the annotated video and analysis report

## Example

![Example Analysis](https://i.imgur.com/8FOcmwL.png)

## Customization

### Adjusting Reference Values

You can adjust the reference values for swing analysis in `swing_analyzer/modules/swing_analyzer.py`:

```python
self.reference_values = {
    'min_hip_rotation': 45.0,      # Minimum hip rotation angle in degrees
    'max_arm_extension': 0.4,      # Maximum normalized distance
    'max_head_movement': 0.03,     # Maximum normalized movement
    'max_shoulder_asymmetry': 10.0 # Maximum angle difference in degrees
}
```

## Limitations and Future Work

- Currently optimized for side-view swing analysis
- Assumes right-handed golfer (can be adapted for left-handed players)
- Future work could include:
  - Machine learning for more accurate swing phase detection
  - Additional metrics for club path analysis
  - Face angle and impact position detection
  - Multi-angle analysis with synchronized cameras
  - Automatic club detection
  - Personalized improvement drills based on detected issues

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for the pose estimation model
- OpenCV for computer vision capabilities 
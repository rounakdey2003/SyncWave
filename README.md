# Brain-Computer Interface (BCI) - Motor Imagery Analysis & Real-time Movement Detection
# 🔗 Link - https://syncwave.streamlit.app

A comprehensive Streamlit application for analyzing motor imagery EEG data from the BCI Competition IV 2a dataset and real-time movement detection using computer vision.

![BCI Application](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🧠 Overview

This application provides an interactive platform for exploring brain-computer interface concepts through:

1. **Motor Imagery Data Analysis**: Visualization and exploration of EEG signals from the BCI Competition IV 2a dataset
2. **Brain Activity Mapping**: Real-time heatmaps showing brain activity patterns
3. **Movement Detection**: Computer vision-based movement detection that simulates brain activity responses

The application is designed to be educational and demonstrates how different brain regions (C3, Cz, C4) activate during different types of motor imagery and physical movements.

## 📊 Dataset Information

The application uses the **BCI Competition IV 2a** dataset, which contains:

- **Subjects**: 9 subjects (A01-A09)
- **Sessions**: Training (T) and Evaluation (E) sessions
- **Channels**: 22 EEG channels + 3 EOG channels
- **Sampling Rate**: 250 Hz
- **Motor Imagery Tasks**:
  - Left hand movement (Class 1)
  - Right hand movement (Class 2)
  - Foot movement (Class 3)
  - Tongue movement (Class 4)

### Brain Regions Mapping

| Channel | Brain Region | Controls | Motor Imagery Task |
|---------|--------------|----------|-------------------|
| C3 | Left Motor Cortex | Right side of body | Right hand movement |
| Cz | Central Motor Cortex | Feet and tongue | Foot/tongue movement |
| C4 | Right Motor Cortex | Left side of body | Left hand movement |

## 🚀 Features

### 1. Brain Activity Map
- **Interactive heatmaps** of brain activity for different motor cortex regions
- **Real-time visualization** of EEG signal patterns
- **Color-coded activity levels**: Red (high activity) to Blue (low activity)

### 2. Brain Signals Explorer
- **Detailed signal analysis** for individual brain regions
- **Interactive channel selection** (C3, Cz, C4)
- **Time-series visualization** of EEG signals

### 3. Movement Detection
- **Real-time webcam integration** for movement detection
- **Body region tracking**: Left hand, right hand, and head movements
- **Simulated brain activity** based on detected movements
- **Live signal generation** showing how brain regions would respond

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for movement detection feature)
- macOS/Linux/Windows

### Setup Instructions

1. **Clone or download the repository**:
   ```bash
   git clone <repository-url>
   cd BCI_IV2a
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv bci_env
   source bci_env/bin/activate  # On Windows: bci_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset files** (if not included):
   - The application expects `.npz` files for subjects A01-A09
   - Files should be named: `A01T.npz`, `A01E.npz`, etc.
   - These should be placed in the root directory

## 🎯 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Application Interface

#### Control Panel (Sidebar)
- **Subject Selection**: Choose from 9 test subjects (A01T-A09T)
- **Download Feature**: Download raw dataset files
- **Navigation**: Links to project resources

#### Main Interface Tabs

1. **Brain Activity Map**
   - View heatmap representations of brain activity
   - Observe patterns across different motor cortex regions
   - Understand activity levels through color coding

2. **Brain Signals Explorer**
   - Select specific brain regions to analyze
   - Examine detailed EEG signal patterns
   - Understand temporal dynamics of brain activity

3. **Movement Detection**
   - Start/stop webcam for real-time movement detection
   - See how physical movements correlate with brain activity
   - Observe live brain signal simulation

### Movement Detection Guide

1. **Click "Start Camera"** to begin webcam capture
2. **Position yourself** in front of the camera
3. **Perform movements**:
   - **Left hand**: Activates right brain (C4)
   - **Right hand**: Activates left brain (C3)
   - **Head movement**: Activates central brain (Cz)
4. **Observe real-time**:
   - Brain activity visualization
   - Simulated EEG signals
   - Movement detection feedback

## 🔧 Technical Implementation

### Core Components

#### 1. MotorImageryDataset Class
```python
class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz')
    def get_trials_from_channel(self, channel=7)
    def get_trials_from_channels(self, channels=[7, 9, 11])
```
Handles loading and processing of BCI Competition IV 2a dataset files.

#### 2. MotionDetector Class
```python
class MotionDetector:
    def __init__(self)
    def start_camera(self)
    def detect_motion(self, frame)
    def get_active_brain_regions(self)
    def generate_real_time_signal(self, region, base_signal)
```
Manages computer vision-based movement detection and brain activity simulation.

### Key Technologies

- **Streamlit**: Web application framework
- **NumPy**: Numerical computing and data manipulation
- **Plotly**: Interactive plotting and visualization
- **OpenCV**: Computer vision and image processing
- **SciPy**: Scientific computing utilities

### Data Processing Pipeline

1. **Data Loading**: Load `.npz` files containing EEG data
2. **Event Extraction**: Extract motor imagery events and trials
3. **Signal Processing**: Process EEG signals for visualization
4. **Movement Detection**: Real-time motion analysis via webcam
5. **Brain Mapping**: Map movements to corresponding brain regions
6. **Visualization**: Generate interactive plots and heatmaps

## 📁 Project Structure

```
BCI_IV2a/
├── app.py                  # Main Streamlit application
├── camera_utils.py         # Motion detection utilities
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── A01T.npz - A09T.npz    # Training dataset files
├── A01E.npz - A09E.npz    # Evaluation dataset files
└── __pycache__/           # Python cache files
```

## 🎨 User Interface Features

### Visual Design
- **Modern, clean interface** with intuitive navigation
- **Color-coded visualizations** for easy interpretation
- **Responsive layout** that works on different screen sizes
- **Real-time updates** for dynamic content

### Interactive Elements
- **Subject selection dropdown** for dataset exploration
- **Tab-based navigation** for different analysis modes
- **Camera controls** for movement detection
- **Downloadable dataset files**

## 🔬 Scientific Background

### Motor Imagery
Motor imagery is the mental rehearsal of motor acts without overt body movement. It activates similar brain regions as actual movement execution, making it valuable for BCI applications.

### EEG Signal Characteristics
- **Mu rhythms (8-12 Hz)**: Decrease during motor imagery
- **Beta rhythms (13-30 Hz)**: Desynchronize during movement
- **Event-related desynchronization (ERD)**: Power decrease in specific frequency bands

### Brain-Computer Interface Applications
- **Assistive technology**: Help paralyzed individuals control devices
- **Rehabilitation**: Motor recovery after stroke
- **Gaming and entertainment**: Brain-controlled interfaces
- **Research**: Understanding brain function and plasticity

## 🚨 Troubleshooting

### Common Issues

1. **Camera not working**:
   - Ensure webcam is connected and not used by other applications
   - Check camera permissions in system settings
   - Try restarting the application

2. **Dataset files not found**:
   - Verify `.npz` files are in the root directory
   - Check file naming convention (A01T.npz, A02T.npz, etc.)
   - Ensure files are not corrupted

3. **Performance issues**:
   - Close other resource-intensive applications
   - Reduce camera resolution if needed
   - Check system requirements

### Error Messages

- **"Could not open video device"**: Camera access issue
- **"Failed to capture frame"**: Camera connection problem
- **"File not found"**: Missing dataset files

## 🎓 Educational Use

This application is ideal for:

- **Students** learning about brain-computer interfaces
- **Researchers** exploring motor imagery data
- **Educators** demonstrating BCI concepts
- **Developers** understanding real-time signal processing

### Learning Objectives
- Understand EEG signal characteristics
- Explore motor cortex brain mapping
- Learn about movement detection algorithms
- Experience real-time data visualization

## 🔮 Future Enhancements

### Planned Features
- **Machine learning classification** of motor imagery tasks
- **Signal preprocessing options** (filtering, artifact removal)
- **Advanced movement detection** with pose estimation
- **Data export capabilities** for further analysis
- **Multi-subject comparison** tools

### Technical Improvements
- **Real EEG device integration** (OpenBCI, Emotiv)
- **Advanced signal processing** algorithms
- **Cloud deployment** options
- **Mobile app** compatibility

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BCI Competition IV organizers** for providing the dataset
- **Streamlit team** for the excellent web framework
- **OpenCV community** for computer vision tools
- **Scientific Python ecosystem** for numerical computing libraries

---

**Note**: This application is for educational and research purposes. It simulates brain activity patterns and should not be used for medical diagnosis or treatment.

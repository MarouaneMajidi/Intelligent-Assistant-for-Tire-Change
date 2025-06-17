# Intelligent-Assistant-for-Tire-Change
[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

## **Documentation:** 
The full project documentation is hosted [here](https://rtd-cvproject.readthedocs.io/en/latest/).

## Overview
This assistant leverages real-time computer vision and NLP to guide users through changing a flat tire using an egocentric (chest-mounted) camera.  

It detects tools, tracks task progress, and provides interactive, step-by-step visual and voice instructions for changing a vehicle tire.

---

## System Features
- **Real-Time Tool Detection**  
  Identifies car jack, wheel wrench, spare tire, and more using YOLOv8.

- **Action Recognition**  
  Tracks the task progression: loosening bolts, jacking the car, removing the tire, etc.

- **Voice Assistant**  
  Responds to voice queries like “What’s next?” using speech recognition and TTS.

- **Edge-Friendly Pipeline**  
  Designed for mobile or embedded deployment with minimal latency.

---

## Models Used
- **Object Detection**  
  YOLOv11, fine-tuned on tire-change-specific tools and flat tire.

- **Action Recognition**  
  TimeDistributed EfficientNetB0 and a streaming version of MoViNet, fine-tuned on the tire change dataset.

- **Voice Assistant**  
  - **Speech Recognition**: [Vosk](https://alphacephei.com/vosk/)
  - **Text-to-Speech**: [pyttsx3](https://pypi.org/project/pyttsx3/)

---

## Dataset

### Data Collection Methodology
- **Primary Source**  
  Self-collected footage of two tire changes on a *Renault Megane 2*, recorded using Samsung A50 smartphones from chest-mounted perspectives.

- **Secondary Source**  
  Curated YouTube videos demonstrating tire changes on various vehicle types.

- **Annotation**  
  Manual labeling of action segments and bounding boxes for object detection.

### Dataset Structure
```
data/
├── lower_car/
├── lift_car_with_jack/
├── tighten_bolts/
├── initial_wrench_tighten/
├── place_spare_tire/
├── remove_tire/
├── hand_tighten_bolts/
├── loosen_bolts/
├── remove_bolts/
├── labels.csv                 # Timestamps and action class labels
└── README.txt                 # Dataset documentation
```
Each action class folder contains video clips used for model training, validation, and testing.

![resized](https://github.com/user-attachments/assets/b3e77c3a-f267-4877-b48a-37b0ff49b208)

---

## Usage
First, ensure you have a **CUDA/cuDNN version compatible with your GPU** installed.

### 1. Clone the Repository
```bash
git clone https://github.com/sohaibdaoudi/ChangingTireAssistant_CV_NLP_Project.git
```

### 2. Create and Activate Virtual Environment (Recommended)

#### For `venv` users:
```bash
# Create the virtual environment with Python 3.9
python3.9 -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Change to project directory after activating environment
cd C:\path\to\folder\ChangingTireAssistant_CV_NLP_Project\Algorithm_V4
```

#### For `conda` users:
```bash
# Create the conda environment with Python 3.9
conda create --name tire-assistant python=3.9

# Activate the environment
conda activate tire-assistant

# Change to project directory after activating environment
cd C:\path\to\folder\ChangingTireAssistant_CV_NLP_Project\Algorithm_V4
```

### 3. Install Dependencies
```bash
pip install -r Algorithm_V4/requirements.txt
```

### 4. Additional Setup
Before running the system, complete the following setup steps:

- **Clone TensorFlow Models Repository:**  
  Inside the `Algorithm_V4` folder, clone the official TensorFlow models repo:
  ```bash
  git clone https://github.com/tensorflow/models
  ```

- **Download Vosk Voice Assistant Model:**  
  Download the lightweight Vosk model `vosk-model-small-en-us-0.15` (or a different version depending on your system performance) from:  
  https://alphacephei.com/vosk/models

  Extract the model folder into the directory:
  ```bash
  Algorithm_V4/vosk-model-small-en-us-0.15
  ```

### 5. Run the System
Use the following command to start the full real-time Changing Tire Assistant:
```bash
python Algorithm_V4/AlgoV4.py
```
---

## Authors
- **Sohaib Daoudi** – [soh.daoudi@gmail.com](mailto:soh.daoudi@gmail.com)  
- **Marouane Majidi** – [majidi.marouane0@gmail.com](mailto:majidi.marouane0@gmail.com)

---

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for more details.

# Intelligent-Assistant-for-Tire-Change
[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)


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

## Project Structure
```              
├── action_recognition/        # Trained models for action recognition & voice assistant
├── object_detection/          # YOLOv8 models for tool detection
├── requirements.txt           # Python dependencies
```

---

## Models Used
- **Object Detection**  
  YOLOv8, fine-tuned on tire-change-specific tools and vehicle parts.

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

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Intelligent-Assistant-for-Tire-Change.git
cd Intelligent-Assistant-for-Tire-Change
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the System
Run action recognition or object detection modules as follows (example):
```bash
python action_recognition/infer.py      # For recognizing user actions
python object_detection/detect.py       # For detecting tools in real-time
```

> Make sure to update paths to your video or webcam stream, and provide access to model weights.

---

## Authors
- **Sohaib Daoudi** – [soh.daoudi@gmail.com](mailto:soh.daoudi@gmail.com)  
- **Marouane Majidi** – [majidi.marouane0@gmail.com](mailto:majidi.marouane0@gmail.com)

---

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for more details.

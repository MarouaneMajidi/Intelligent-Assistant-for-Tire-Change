# Intelligent-Assistant-for-Tire-Change
[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**Project Status: Under Development**

---

## Overview
This assistant leverages real-time computer vision and NLP to guide users through changing a flat tire using an egocentric (chest-mounted) camera. 

The assistant detects tools, tracks task progress, and provides interactive, step-by-step visual and voice instructions for changing a vehicle tire.

---

## System Features
- **Real-Time Tool Detection**:
  - Identifies car jack, wheel wrench, etc.
- **Action Recognition**:
  - Tracks task progression like loosening nuts, jacking the car, replacing the wheel, etc.
- **Voice Assistant**:
  - Responds to user queries such as "What's next?"
- **Edge-Friendly Pipeline**:
  - Designed for deployment on mobile or embedded systems with minimal latency.

---

## Project Structure
```              
├── action_recognition/                 # Trained YOLO/Action Recognition/Voice Assistant models
├── object_detection/      
├── requirements.txt          
```

---

## Models Used
- **Object Detection**: YOLOv8 fine-tuned on tire-change-specific tools and components
- **Action Recognition**: We are trying different models SlowFast , TSM , TimeDistributed EfficientNetB0
- **Voice Assistant**: Whisper-based STT with a custom NLP pipeline for contextual understanding

---

## Data
We collected and curated a custom dataset specifically for the tire change domain:

### Data Collection Methodology
- **Primary Source**: Self-collected footage changing two tires on a Renault Megane 2, recorded with Samsung A50 smartphones from chest-mounted positions
- **Secondary Source**: Curated YouTube videos showing different tire change scenarios and vehicle types
- **Annotation Process**: Manual annotation of action segments and tool detection bounding boxes

### Dataset Structure
The data directory contains the following action classes:
```
data/
├── lower_car/             # Videos lowering the car from the jack
├── lift_car_with_jack/    # Videos raising the car with jack
├── tighten_bolts/         # Videos final bolt tightening with wrench
├── initial_wrench_tighten/# Videos initial wrench positioning
├── place_spare_tire/      # Videos positioning the spare tire
├── remove_tire/           # Videos removing the flat tire
├── hand_tighten_bolts/    # Videos hand-tightening bolts
├── loosen_bolts/          # Videos loosening wheel bolts
├── remove_bolts/          # Videos removing wheel bolts
├── labels.csv             # Action timestamps and class annotations
└── README.txt             # Dataset documentation
```

Each action class directory contains video clips used for model training and validation.

https://github.com/user-attachments/assets/6d6bef7f-5d31-4b78-b57b-93b3566c5007


## Authors
- **SOHAIB DAOUDI** – [soh.daoudi@gmail.com](mailto:soh.daoudi@gmail.com)
- **MAROUANE MAJIDI** – [majidi.marouane0@gmail.com](mailto:majidi.marouane0@gmail.com)

---

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for more details.

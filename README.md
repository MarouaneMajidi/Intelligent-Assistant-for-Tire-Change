# Intelligent-Assistant-for-Tire-Change

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**🚧 Project Status: Under Development 🚧**

---

## 📖 Overview

This assistant leverages real-time computer vision and NLP to guide users through changing a flat tire using an egocentric (chest-mounted) camera. Inspired by MECCANO project, this project focuses on practical, real-world assistance using a lightweight CV-NLP pipeline.

The assistant detects tools, tracks task progress, and provides interactive, step-by-step visual and voice instructions.

---

## 🔍 System Features

- **Real-Time Tool Detection**:
  - Identifies car jack, wheel wrench, and flat/new tires.
- **Action Recognition**:
  - Tracks task progression like loosening nuts, jacking the car, replacing the wheel, etc.
- **Voice Assistant**:
  - Responds to user queries such as "What’s next?" or "Am I doing it right?"
- **Edge-Friendly Pipeline**:
  - Designed for deployment on mobile or embedded systems.

---

## 📂 Project Structure

```
├── app/                   
├── models/                 # Trained YOLO/Action Recognition/Voice Assistant models
├── notebooks/              # Data processing and model training notebooks
├── data/                   # Sample egocentric video frames, annotations
├── utils/                  # Utility scripts for preprocessing, inference
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── LICENSE.txt               
```

---

## 🧠 Models Used

- **Object Detection**
- **Action Recognition**
- **Voice Assistant**

---

## 🔮 Future Work

- [ ] Integrate multimodal reasoning for complex error recovery.
- [ ] Improve action segmentation with Transformers.
- [ ] Enhance voice assistant via on-device TTS/STT.

---

## 👨‍💻 Authors

- **SOHAIB DAOUDI** – [soh.daoudi@gmail.com](mailto:soh.daoudi@gmail.com)
- **MAROUANE MAJIDI** – [majidi.marouane0@gmail.com](mailto:majidi.marouane0@gmail.com)

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for more details.

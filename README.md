# Bottle Classification System

A computer vision system that classifies bottles in images, with a special focus on detecting PET bottles. The system includes both a command-line interface for analyzing static images and a real-time camera interface for live detection.

## Features

- Bottle detection and classification using ResNet50 architecture
- PET bottle-specific detection
- Real-time camera interface for live detection
- Support for multiple image formats (JPG, PNG, BMP)
- Confidence scores for classifications
- Automatic model weights download

## Prerequisites

- Python 3.12+
- Webcam (for real-time detection)

## Installation

### Clone the repository:

```bash
git clone https://github.com/omkar-tumati/Bottle_Classification
cd Bottle_Classification
```

# Usage

## Static Image Classification

Run the bottle classifier on a single image:

```bash
python bottle_classifier.py
```

When prompted, enter the path to your image file

## Real-time Camera Detection

Use the camera interface for real-time detection:

```bash
python camera_interface.py
```

Camera Interface Controls:

- Press 'c' to capture and classify an image
- Press 'q' to quit

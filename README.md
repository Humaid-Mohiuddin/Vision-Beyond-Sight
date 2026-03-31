# Vision Beyond Sight

**AI-powered assistive system for visually impaired individuals using computer vision and real-time audio feedback.**

The system processes live camera input, detects objects and obstacles, and provides spoken guidance to help visually impaired users navigate safely.

## Project Motivation

Millions of visually impaired individuals face difficulty navigating unfamiliar environments.

Traditional assistive tools such as canes provide limited environmental awareness.

Vision Beyond Sight aims to create a cost-effective solution which augments mobility by combining:

- Computer Vision
- AI Object Detection
- Real-time audio feedback
 
to create a portable assistive navigation system.

## Key Features

### Current:
- Real-time object detection
- Obstacle identification through depth estimation
- Audio navigation feedback

### Future:
- Lightweight architecture and optimizations for inference on edge devices
- Design for smart glasses integration
- Point A to point B navigation system influencing navigation decisions
- Rough surface alerts and warning systems
- Elevators and Staircases detection and guidance systems.

## Technology Stack
 
### Computer Vision
- OpenCV
- Numpy

### Machine Learning
- Object Detection - YOLOv8
- Depth Estimation - MiDaS, Depth Anything V2

### Programming Language
- Python

### Audio Generation
- pyttsx3

### Hardware (Planned)
- Smart glasses / camera module
- Processing on edge devices - Hailo8 / Memry / Raspberry Pi



## **INSTALLATION AND USAGE**

**Please read and follow this entire section for proper configuration and running the project without any issues.**

### Clone the repository

```
git clone https://github.com/Humaid-Mohiuddin/Vision-Beyond-Sight.git
```

### Navigate to the project folder

```
cd Vision-Beyond-Sight
```

### **Creating an environment**

#### Using python:
- Create
```
python -m venv .venv
```
- Activate
```
.venv\Scripts\activate
```



#### Using uv:
- Create
```
uv venv --python 3.11
```
- Activate
```
.venv\Scripts\activate
```


### **Installing dependencies**

#### Using pip:
```
python -m pip install -r requirements.txt
```

#### Using uv:
```
uv sync
```

### **Additional Dependencies (Pytorch)**

- CPU (general use-case):
```
python -m pip install torch torchvision torchaudio
```
- GPU:

If you have an NVIDIA gpu you can use it to boost performance, you will have to install the cuda version of pytorch which can be heavy in size ~2.4 Gb

```
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Running the system

### Navigate to the src folder of the project

```
cd src
```

### Run the main.py file

```
python main.py
```


## Research Inspiration

- Assistive technology research
- Computer vision navigation systems
- Human-computer interaction for accessibility


## Author

### [Humaid Mohiuddin](https://github.com/Humaid-Mohiuddin)
### Computer Science Student
### Interested in:
- Artificial Intelligence
- Machine Learning
- Computer Vision


## License

[MIT License](LICENSE)
 
    
            




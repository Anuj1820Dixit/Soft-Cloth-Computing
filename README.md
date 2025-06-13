# Soft Cloth Detection and Dataset Creation

This project focuses on creating a custom image dataset and building a computer vision system for detecting soft cloth in different physical states using **TensorFlow** and **OpenCV**.

## 🧩 Problem Statement

Accurate detection of soft cloth—whether **plain**, **folded**, or **crumbled**—is a challenging task due to the lack of publicly available labeled datasets and the visual similarity across states. This project addresses both the data scarcity issue and builds an efficient recognition system.

## 📂 Project Overview

### ✅ Key Contributions

- 📸 **Custom Dataset Creation**  
  Developed a labeled dataset of soft cloth images in **three distinct states**:  
  - *Plain (spread out)*  
  - *Folded*  
  - *Crumbled (irregular form)*  
  Dataset created using controlled environments and organized for training, validation, and testing.

- 🧠 **Image Recognition System**  
  Implemented a cloth recognition pipeline using:
  - **TensorFlow** – for data preparation and possible future model training.
  - **OpenCV** – for real-time cloth image processing, surface detection, and state classification.

## 🛠️ Tech Stack

| Tool/Library   | Usage                                     |
|----------------|--------------------------------------------|
| TensorFlow     | Data handling, preprocessing               |
| OpenCV         | Image processing and analysis              |
| NumPy          | Array operations and manipulation          |
| Python         | Core programming language                  |


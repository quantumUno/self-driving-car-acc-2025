# Quanser ACC 2025 Self-Driving Car Competition - Stage 1 Submission

  Welcome to our team's submission for the Quanser Self-Driving Car Student Competition at the American Control Conference (ACC) 2025. This repository contains our implementation of a self-driving algorithm for the QCar 2 digital twin, developed using Quanser Interactive Labs (QLabs). Our submission addresses the core principles of self-driving, including data collection, interpretation, control systems, and localization/path planning, as outlined in the competition handbook.

  ## Team Overview
  **Team Name**: Quantum Uno  
  **Institution**: Obafemi Awolowo University  
  **Members**: **Project Lead**: Oversees progress and ensures deadlines are met: Uthman and Bolu
               **Software Developer**: Implements ROS 2 and Python algorithms: Jonnie, Paragon and Bolu
               **Perception Engineer**: Works on sensor data processing and object detection: Feranmi and Timi
               **Control Systems Engineer**: Develops path planning and vehicle control logic: Olaoti, Iseoluwa and Uthman
               **Documentation & Presentation Lead**: Handles GitHub repo and video submission: Seun and Tijani Uthman

  **Supervisor**: Dr. Ilori Olusoji

  ## Self-Driving Car Studio Setup
  The Self-Driving Car Studio in Quanser Interactive Labs (QLabs) is configured to simulate the QCar 2 digital twin in the Detailed Scenario, including maps, peripherals, and traffic lights:
  - **Maps**: Loaded the Detailed Scenario’s road network with lanes, intersections, and obstacles.
  - **Peripherals**: Configured QCar 2 sensors (camera, LiDAR, IMU, encoders) to publish data via ROS 2 topics.
  - **Traffic Lights**: Detected using camera-based image processing (OpenCV) to enforce traffic rules (e.g., stop at red).
  See the respective sections for implementation details.

  ## Repository Structure
  - **data_collection/**: Sensor data collection scripts and examples.
  - **interpretation/**: Data processing for lane and traffic light detection.
  - **control_systems/**: Steering and throttle control algorithms.
  - **localization_path_planning/**: Localization and path planning strategies.
  - **video/**: Link to our YouTube demonstration video.

  ## Getting Started
  ### Prerequisites
  - **Software**: ROS 2 Humble on Ubuntu 20.04.
  - **Environment**: QLabs with QCar 2 digital twin.
  - **Dependencies**: NumPy, OpenCV, ROS 2 packages (see section READMEs).
  - **Hardware**: None; uses QLabs virtual environment.

  ### Installation
  1. Download this repository using GitHub’s “Code” > “Download ZIP”.
  2. Install dependencies per section READMEs.
  3. Configure QLabs to run the Detailed Scenario.

  ## Running the Code
  Run scripts in each section’s `code/` folder. See individual READMEs for instructions.

  ## Video Demonstration
  Our 3-minute video demonstrates the algorithm’s performance in the Detailed Scenario. See `video/README.md` for the YouTube link.

  ## Contact
  Contact quantumuno6@gmail.com for questions.

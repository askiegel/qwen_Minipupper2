# Software Architecture

## Layered Design

Mission Layer

↓

Behavior Layer

↓

Tracking Layer

↓

Perception Layer

↓

Motion Layer

↓

Robot Layer

## Core Components

Mission Manager

Chooses robot mission.

Behavior Manager

Chooses current robot behavior.

Target Manager

Maintains target identity.

Provides Person ReID.

Handles reacquisition.

Object Tracker

Processes YOLO detections.

Follow Manager

PID steering.

Arrival logic.

Search Behavior

Lost target recovery.

Motion

ROS cmd_vel generation.

Dashboard

Visualization and mission selection.

## ROS Topics

/image_raw

/scan

/cmd_vel

/qwen_status

/qwen_mission

## Future Architecture

Target Memory

World Memory

SLAM

Navigation

Voice Interface

Robot Communication

Distributed Planning

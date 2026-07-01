# Mini Pupper 2 Startup Guide

Project: Qwen Mini Pupper 2
Version: v0.8-alpha1

============================================================
1. START THE LINUX VISION SERVER
============================================================

On the Linux PC:

    cd ~/vision_server
    source .venv/bin/activate
    python3 -m uvicorn server:app --host 0.0.0.0 --port 8000

Expected:

    Uvicorn running on http://0.0.0.0:8000

============================================================
2. VERIFY THE VISION SERVER
============================================================

From the Mini Pupper:

    curl http://192.168.68.135:8000/docs

or

    ros2 run qwen_robot target_check

Expected detections:

- person
- backpack
- chair
- tv

============================================================
3. SSH INTO THE MINI PUPPER
============================================================

    ssh ubuntu@<robot-ip>

============================================================
4. OPEN THE ROS WORKSPACE
============================================================

    cd ~/ros2_ws

============================================================
5. SOURCE ROS
============================================================

    source /opt/ros/humble/setup.bash
    source install/setup.bash

============================================================
6. CONFIGURE THE VISION SERVER
============================================================

    export VISION_SERVER_URL=http://192.168.68.135:8000/detect

Verify:

    echo $VISION_SERVER_URL

Expected:

    http://192.168.68.135:8000/detect

============================================================
7. START MINI PUPPER BRINGUP
============================================================

    ros2 launch mini_pupper_bringup bringup.launch.py

Wait until:

- Camera starts
- LiDAR starts
- Servos initialize

============================================================
8. VERIFY CAMERA
============================================================

    ros2 topic hz /image_raw

Expected:

20-30 FPS

============================================================
9. VERIFY VISION
============================================================

    ros2 run qwen_robot target_check

Expected detections:

- person
- backpack
- chair
- tv

============================================================
10. START THE ROBOT
============================================================

    ros2 run qwen_robot qwen_robot

Expected:

Mission: FOLLOW_PERSON

Navigation:
SEARCHING or TRACKING

============================================================
11. START DASHBOARD PRO
============================================================

Open another terminal.

    cd ~/ros2_ws

    source install/setup.bash

    export VISION_SERVER_URL=http://192.168.68.135:8000/detect

    ros2 run qwen_robot qwen_dashboard

Open:

    http://<robot-ip>:5000

============================================================
12. DASHBOARD CHECKLIST
============================================================

Verify:

[ ] Live camera

[ ] Bounding boxes

[ ] Target crop

[ ] Mission buttons

[ ] Navigation panel

[ ] ReID panel

[ ] Build information

============================================================
13. PERSON TRACKING TEST
============================================================

Stand in front of the robot.

Expected:

- Green person box
- Target crop updates
- Robot follows

Walk away.

Expected:

SEARCHING

(Future release: GO_TO_LAST_SEEN)

============================================================
14. BACKPACK TEST
============================================================

Select:

Find Backpack

Expected:

- Yellow TARGET box

- Robot approaches backpack

============================================================
SHUTDOWN
============================================================

Dashboard:

STOP

Robot:

Ctrl+C

Dashboard:

Ctrl+C

Vision Server:

Ctrl+C

============================================================
TROUBLESHOOTING
============================================================

No Bounding Boxes

    echo $VISION_SERVER_URL

Restart the vision server if necessary.

------------------------------------------------------------

Vision Server Offline

    cd ~/vision_server

    source .venv/bin/activate

    python3 -m uvicorn server:app --host 0.0.0.0 --port 8000

------------------------------------------------------------

Dashboard Won't Start

    sudo lsof -i :5000

Kill the process:

    sudo kill -9 <PID>

Restart:

    ros2 run qwen_robot qwen_dashboard

------------------------------------------------------------

Person Detection Fails

    ros2 run qwen_robot target_check

If no detections:

- Verify Vision Server
- Verify network connection
- Verify VISION_SERVER_URL

============================================================
CURRENT ARCHITECTURE
============================================================

Mission Manager

↓

Behavior Manager

↓

Target Manager

↓

Memory Manager

↓

Navigation Manager

↓

TargetBus

↓

Follow Manager

↓

Motion Controller

============================================================
CURRENT RELEASE
============================================================

v0.8-alpha1

Includes:

- Dashboard Pro
- Linux Vision Server
- Person Tracking
- Backpack Tracking
- TargetSnapshot
- TargetBus
- MemoryManager
- Navigation Framework
- ReID Framework


# Development Guide

## Philosophy

This project is being developed as a professional robotics software stack.

Every subsystem should have exactly one responsibility.

The architecture is intentionally layered to allow future expansion without major rewrites.

## Design Rules

Mission logic never controls motors.

Behavior logic never performs perception.

Perception never performs motion.

Motion never decides missions.

Dashboard visualizes only.

Communication between modules occurs through ROS topics or clearly defined interfaces.

## Development Workflow

1. Make one change.

2. Build.

3. Test.

4. Commit.

5. Tag stable milestones.

## Git Workflow

main

↓

feature/person-reid

↓

feature/slam

↓

feature/multi-robot

↓

merge

↓

tag release

## Stable Milestones

v0.6

Mission Manager

Dashboard Pro

Search Behavior

LiDAR Safety

Person Follow

Backpack Follow

v0.7

Target Manager

ReID Telemetry

Dashboard Target Panel

## Coding Style

Small classes

Single responsibility

Readable code

Minimal dependencies

Avoid global state

Always test before commit.

class RobotController:
    def __init__(self, motion, lidar, camera, vision, memory, logger):
        self.motion = motion
        self.lidar = lidar
        self.camera = camera
        self.vision = vision
        self.memory = memory
        self.logger = logger

    def handle_move(self, plan):
        direction = plan.get('direction', 'forward')
        speed = max(0.0, min(float(plan.get('speed', 0.05)), 0.10))
        duration = max(0.1, min(float(plan.get('duration', 1.0)), 3.0))

        if direction == 'forward':
            if self.lidar.blocked(0.35):
                self.motion.stop()
                return 'blocked', 'Forward blocked by obstacle.'

            self.motion.move(linear_x=speed, angular_z=0.0, duration=duration)
            return 'stopped', 'Moved forward.'

        if direction == 'backward':
            self.motion.move(linear_x=-speed, angular_z=0.0, duration=duration)
            return 'stopped', 'Moved backward.'

        return 'stopped', f'Unknown move direction: {direction}'

    def handle_turn(self, plan):
        direction = plan.get('direction', 'left')
        speed = max(0.0, min(float(plan.get('speed', 0.4)), 0.8))
        duration = max(0.1, min(float(plan.get('duration', 1.0)), 3.0))

        if direction == 'left':
            self.motion.move(linear_x=0.0, angular_z=speed, duration=duration)
            return 'stopped', 'Turned left.'

        if direction == 'right':
            self.motion.move(linear_x=0.0, angular_z=-speed, duration=duration)
            return 'stopped', 'Turned right.'

        return 'stopped', f'Unknown turn direction: {direction}'

    def handle_stop(self):
        self.motion.stop()
        return 'stopped', 'Stopped.'

    def handle_picture(self):
        filename = self.camera.save()
        if filename is None:
            return 'stopped', 'No camera image received yet.'
        return 'stopped', f'Saved image to {filename}'

    def handle_status(self):
        if self.lidar.front_distance is None:
            return 'stopped', 'No LiDAR distance available yet.'
        return 'stopped', f'Closest object in front: {self.lidar.front_distance:.2f} m'

    def handle_vision(self):
        frame = self.camera.get_latest()
        self.logger.info('Sending frame to vision server...')

        result = self.vision.analyze_frame(frame)
        observation = self.memory.add_vision_observation(result)

        description = result.get('description', 'No description.')
        self.logger.info(f'Vision: {description}')
        self.logger.info(f'Memory observations: {self.memory.count()}')

        return 'stopped', description, result, observation

    def handle_memory(self):
        summary = self.memory.summary()
        return 'stopped', summary

    def handle_compare(self):
        comparison = self.memory.compare_latest_two()
        return 'stopped', comparison

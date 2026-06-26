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
        return 'stopped', description, result, observation

    def handle_memory(self):
        return 'stopped', self.memory.summary()

    def handle_compare(self):
        return 'stopped', self.memory.compare_latest_two()

    def follow_person_step(self):
        return self.find_object_step('person', follow=True)

    def find_object_step(self, target_label, follow=False):
        frame = self.camera.get_latest()
        result = self.vision.analyze_frame(frame)

        detections = result.get('detections', [])
        targets = [d for d in detections if d.get('label') == target_label]

        if not targets:
            self.motion.move(linear_x=0.0, angular_z=0.18, duration=0.25)
            return {
                'state': 'searching',
                'message': f'No {target_label} detected. Searching slowly.',
                'target': None,
                'vision': result,
                'found': False
            }

        target = max(targets, key=lambda d: d.get('width', 0) * d.get('height', 0))

        center_x = target.get('center_x', 320)
        image_width = target.get('image_width', 640)
        box_height = target.get('height', 0)
        confidence = target.get('confidence', 0.0)

        image_center = image_width / 2.0
        error = center_x - image_center
        normalized_error = error / image_center

        kp_turn = 0.45
        angular_z = -kp_turn * normalized_error
        angular_z = max(-0.35, min(angular_z, 0.35))

        if abs(normalized_error) < 0.08:
            angular_z = 0.0

        if follow:
            if box_height < 220:
                linear_x = 0.045
            elif box_height < 320:
                linear_x = 0.025
            else:
                linear_x = 0.0

            if self.lidar.blocked(0.45):
                linear_x = 0.0
        else:
            linear_x = 0.0

        self.motion.move(linear_x=linear_x, angular_z=angular_z, duration=0.25)

        if abs(angular_z) > 0:
            state = 'centering_target'
            message = f'Found {target_label}. Centering target. Error {error:.0f}px. Confidence {confidence:.2f}.'
        else:
            state = 'target_centered'
            message = f'Found {target_label}. Target centered. Confidence {confidence:.2f}.'

        return {
            'state': state,
            'message': message,
            'target': target,
            'vision': result,
            'found': True
        }

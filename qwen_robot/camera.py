import cv2


class CameraManager:
    def __init__(self, bridge):
        self.bridge = bridge
        self.latest_image = None

    def update(self, image_msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(
            image_msg,
            desired_encoding='bgr8'
        )

    def get_latest(self):
        if self.latest_image is None:
            return None

        return self.latest_image.copy()

    def save(self, filename='/tmp/qwen_robot_camera.jpg'):
        if self.latest_image is None:
            return None

        cv2.imwrite(filename, self.latest_image)
        return filename

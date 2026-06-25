import math


class LidarSafety:
    def __init__(self):
        self.front_distance = None

    def update(self, scan_msg):
        front_ranges = []
        angle = scan_msg.angle_min

        for r in scan_msg.ranges:
            if -0.35 <= angle <= 0.35:
                if not math.isinf(r) and not math.isnan(r):
                    front_ranges.append(r)

            angle += scan_msg.angle_increment

        self.front_distance = min(front_ranges) if front_ranges else None

    def blocked(self, threshold=0.35):
        return self.front_distance is not None and self.front_distance < threshold

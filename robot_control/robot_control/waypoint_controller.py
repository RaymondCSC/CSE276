import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


# ---------- CALIBRATION:  ----------
FORWARD_LEFT_COMMAND = 0.43
FORWARD_RIGHT_COMMAND = 0.40 #when speed grow command muust grow at the same time for same distance
FORWARD_SPEED = 0.54       # m/s EG not exactly 

TURN_COMMAND = 0.5        
TURN_RATE = 4.2           # rad/s EG not exactly

# --------------------------------------------


def wrap(angle):
    """Choose the shortest turn, between -pi and pi."""
    return math.atan2(math.sin(angle), math.cos(angle))


def read_waypoints():
    """Load the entire file into a list."""
    path = Path(__file__).with_name("waypoints.txt")
    points = []

    with open(path) as file:
        for line in file:
            if line.strip():
                x, y, angle = map(float, line.split(","))
                if not all(math.isfinite(v) for v in (x, y, angle)):
                    raise ValueError("Waypoint values must be finite")
                points.append((x, y, angle))

    if not points:
        raise ValueError("Waypoint file is empty")

    return points


class Robot(Node):
    def __init__(self):
        super().__init__("waypoint_controller")
        self.publisher = self.create_publisher(
            Float32MultiArray, "motor_commands", 10
        )

    def send(self, left, right):
        message = Float32MultiArray()
        message.data = [float(left), float(right)]
        self.publisher.publish(message)

    def move(self, left, right, seconds):
        """Keep sending a motor command for the specified duration."""
        end = time.monotonic() + seconds

        while time.monotonic() < end:
            self.send(left, right)
            rclpy.spin_once(self, timeout_sec=0.02)

        self.stop()

    def stop(self):
        """Send repeated zero commands and allow a short settling time."""
        end = time.monotonic() + 0.2

        while rclpy.ok() and time.monotonic() < end:
            self.send(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.02)

    def forward(self, distance):
        seconds = distance / FORWARD_SPEED
        self.move(FORWARD_LEFT_COMMAND, FORWARD_RIGHT_COMMAND, seconds)

    def turn(self, angle):
        angle = wrap(angle)
        print(f"Turning {math.degrees(angle):.1f} degrees")
        if abs(angle) < 0.001:
            return

        direction = 1 if angle > 0 else -1
        command = direction * TURN_COMMAND
        seconds = abs(angle) / TURN_RATE

        self.move(-command, command, seconds)


def main():

    values = (
        FORWARD_SPEED, TURN_RATE, FORWARD_LEFT_COMMAND, FORWARD_RIGHT_COMMAND,
        TURN_COMMAND
    )
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Calibration values must be positive and finite")

    if max(FORWARD_LEFT_COMMAND, FORWARD_RIGHT_COMMAND, TURN_COMMAND) > 0.5:
        raise ValueError("Motor commands must be <= 0.5")

    # Read the whole file before moving.
    points = read_waypoints() 

    rclpy.init()
    robot = Robot()

    try:
        # Wait until motor_control.py subscribes.
        deadline = time.monotonic() + 5.0
        while robot.publisher.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                raise RuntimeError("Start motor_control.py first")
            rclpy.spin_once(robot, timeout_sec=0.05)

        
        print("Loaded waypoints:", points)

        #just for testing turn
        #robot.turn(1.57) 
        # robot.forward(1)
        # return
        #####################
        # Physically start at the first point and its heading.
        x, y, heading = points[0]

        for next_x, next_y, next_heading in points[1:]:
            dx = next_x - x
            dy = next_y - y
            distance = math.hypot(dx, dy)

            if distance > 0.001:
                direction = math.atan2(dy, dx)

                robot.turn(direction - heading)
                robot.forward(distance)
                heading = direction

            robot.turn(next_heading - heading)

            # Assume the commanded movement was completed.
            x, y, heading = next_x, next_y, next_heading
            print("Processed waypoint:", x, y, heading)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            robot.stop()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
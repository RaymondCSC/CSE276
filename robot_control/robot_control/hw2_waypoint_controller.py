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
TURN_RATE = 3.8           # rad/s EG not exactly

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


# ---------- HW2 helpers: your Robot class above is unchanged ----------
POSITION_TOLERANCE = 0.08             # metres; accept within 8 cm
HEADING_TOLERANCE = math.radians(6)  # final heading within 6 degrees

# Copied from the previous controller: VERIFY against your physical setup.
# ID: (world x, world y, centre height, facing degrees, printed spin degrees).
# Facing: 0 = +x, 90 = +y, 180 = -x, -90 = -y.
# Spin: positive counterclockwise when looking at the printed face.
LANDMARKS = {
    0: (1.5, 0.0, 0.11, 180, 90),
    1: (1.0, 2.5, 0.11, -90, 90),
    2: (0.0, 2.0, 0.11, 0, 90),
    3: (-0.25, -0.5, 0.11, 63.434, 90),
}
CAMERA_XYZ = (0.097, 0.0, 0.12)  # camera offset: forward, left, up (metres)
CAMERA_RPY_DEG = (0.0, 0.0, 0.0)
CAMERA_FRAME = "camera_frame"     # optical axes: right, down, forward


def setup_apriltag_tracking(robot):
    """Attach a /tf subscriber without changing the HW1 Robot class."""
    import numpy as np
    from rclpy.qos import qos_profile_sensor_data
    from tf2_msgs.msg import TFMessage

    def rotation(roll, pitch, yaw):
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]])

    def transform(R, xyz):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, xyz
        return T

    def inverse(T):
        R = T[:3, :3].T
        return transform(R, -R @ T[:3, 3])

    # Build known world<-tag transforms and camera<-robot mount transform.
    world_tags = {}
    for tag_id, (x, y, z, facing, spin) in LANDMARKS.items():
        a = math.radians(facing)
        R = np.array([[-math.sin(a), 0, math.cos(a)],
                      [math.cos(a), 0, math.sin(a)], [0, 1, 0]])
        world_tags[f"tag_{tag_id}"] = transform(
            R @ rotation(0, 0, math.radians(spin)), (x, y, z))
    optical_to_forward = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    mount = rotation(*map(math.radians, CAMERA_RPY_DEG))
    camera_robot = inverse(transform(mount @ optical_to_forward, CAMERA_XYZ))
    robot.latest_pose = None
    robot.pose_stamp = -float("inf")
    robot.measure_after = -float("inf")

    def on_tf(message):
        # Use a new image captured after the most recent movement/settling.
        now = robot.get_clock().now().nanoseconds / 1e9
        candidates = []
        for item in message.transforms:
            if item.header.frame_id != CAMERA_FRAME or item.child_frame_id not in world_tags:
                continue
            stamp = item.header.stamp.sec + item.header.stamp.nanosec / 1e9
            if stamp <= max(robot.measure_after, robot.pose_stamp) or not 0 <= now-stamp <= 3.0:
                continue
            q = item.transform.rotation
            v = np.array([q.x, q.y, q.z, q.w], dtype=float)
            norm = np.linalg.norm(v)
            if not np.isfinite(norm) or norm < 1e-8:
                continue
            x, y, z, w = v / norm
            R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                          [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            p = item.transform.translation
            camera_tag = transform(R, (p.x, p.y, p.z))
            distance = np.linalg.norm(camera_tag[:3, 3])
            if not np.isfinite(camera_tag).all() or p.z <= 0 or not 0.05 < distance < 4.0:
                continue
            # world<-robot = world<-tag @ tag<-camera @ camera<-robot.
            T = world_tags[item.child_frame_id] @ inverse(camera_tag) @ camera_robot
            if abs(T[2, 3]) > 0.20 or T[2, 2] < math.cos(math.radians(25)):
                continue
            pose = (float(T[0, 3]), float(T[1, 3]), math.atan2(T[1, 0], T[0, 0]))
            candidates.append((stamp, distance, pose))
        if candidates:
            # Prefer newest observation, then nearest tag; one visible tag suffices.
            robot.pose_stamp, _, robot.latest_pose = min(candidates, key=lambda c: (-c[0], c[1]))

    robot.tag_subscription = robot.create_subscription(
        TFMessage, "/tf", on_tf, qos_profile_sensor_data)


def get_current_location(robot):
    """Wait 3 seconds; if lost, turn 20 degrees CCW and wait 1 second repeatedly."""
    # Invalidate cached observations, including images captured while moving.
    robot.latest_pose = None
    robot.measure_after = robot.get_clock().now().nanoseconds / 1e9
    deadline = time.monotonic() + 3.0
    while rclpy.ok() and time.monotonic() < deadline:
        robot.send(0.0, 0.0)
        rclpy.spin_once(robot, timeout_sec=0.02)
        if robot.latest_pose is not None:
            return robot.latest_pose

    while rclpy.ok():
        print("No tag pose: turning 20 degrees CCW, then waiting 1 second")
        robot.turn(math.radians(20))
        robot.latest_pose = None
        robot.measure_after = robot.get_clock().now().nanoseconds / 1e9
        deadline = time.monotonic() + 1.0
        # Process callbacks during the wait, instead of blocking with time.sleep.
        while rclpy.ok() and time.monotonic() < deadline:
            robot.send(0.0, 0.0)
            rclpy.spin_once(robot, timeout_sec=0.02)
        if robot.latest_pose is not None:
            return robot.latest_pose
    raise RuntimeError("ROS shut down while waiting for an AprilTag pose")


def draw_chart_of_traverled_points(traverled_points, points):
    """Save measured x/y samples and the requested route beside this script."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if traverled_points:
        x, y = zip(*traverled_points)
        ax.plot(x, y, ".-", label="Measured positions (joined samples)")
    ax.plot([p[0] for p in points], [p[1] for p in points], "k--o", label="Waypoints")
    ax.set(xlabel="World x (m)", ylabel="World y (m)", title="HW2 trajectory")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend()
    path = Path(__file__).with_name("trajectory.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory: {path}")


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

        # Start tracking; physically start at points[0], as in HW1.
        setup_apriltag_tracking(robot)
        x, y, heading = points[0]
        current_location = (x, y, heading)  # initial pose is known
        traverled_points = [current_location[0:2]]

        for next_x, next_y, next_heading in points[1:]:
            # 1. Keep approaching until measured position is within tolerance.
            while math.hypot(next_x-current_location[0], next_y-current_location[1]) > POSITION_TOLERANCE:
                x, y, heading = current_location
                dx = next_x - x
                dy = next_y - y
                distance = math.hypot(dx, dy)

                # 2. Face the target, then move straight at most 0.3 metres.
                robot.turn(wrap(math.atan2(dy, dx) - heading))
                robot.forward(min(0.3, distance))

                # 3. Measure again after the step and record the measured x/y.
                current_location = get_current_location(robot)
                print(f"x: {current_location[0]:.1f}, y: {current_location[1]:.1f}, heading: {math.degrees(current_location[2]):.1f} deg")
                traverled_points.append(current_location[0:2])

            # 4. At the position, turn toward the waypoint's final heading.
            heading = current_location[2]
            robot.turn(wrap(next_heading - heading))
            current_location = get_current_location(robot)
            traverled_points.append(current_location[0:2])

            # 5. Keep correcting final heading until it is within tolerance.
            while abs(wrap(next_heading-current_location[2])) > HEADING_TOLERANCE:
                current_heading = current_location[2]
                robot.turn(wrap(next_heading - current_heading))
                current_location = get_current_location(robot)
                traverled_points.append(current_location[0:2])

        # 6. Plot the recorded positions once all waypoints are finished.
        draw_chart_of_traverled_points(traverled_points, points)
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

import argparse
import math
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


# ---------- CALIBRATION:  ----------
FORWARD_LEFT_COMMAND = 0.418
FORWARD_RIGHT_COMMAND = 0.40 #when speed grow command muust grow at the same time for same distance
FORWARD_SPEED = 0.525       # m/s EG not exactly        # m/s EG not exactly 

TURN_COMMAND = 0.5        
TURN_RATE = 3.70        # rad/s EG not exactly

# --------------------------------------------


def wrap(angle):
    """Choose the shortest turn, between -pi and pi."""
    return math.atan2(math.sin(angle), math.cos(angle))


class Robot(Node):
    def __init__(self):
        super().__init__("waypoint_controller")
        self.publisher = self.create_publisher(
            Float32MultiArray, "motor_commands", 10
        )
        self.total_translation = 0.0
        self.total_rotation = 0.0

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
        self.total_translation += abs(distance)

    def turn(self, angle):
        angle = wrap(angle)
        if abs(angle) < 0.001:
            return

        direction = 1 if angle > 0 else -1
        command = direction * TURN_COMMAND
        seconds = abs(angle) / TURN_RATE

        self.move(-command, command, seconds)
        self.total_rotation += abs(angle)


# HW3 settings. The Robot class and motor calibration above are from HW2.
pi = math.pi
square_waypoints = [
    (0.0, 0.0, 0.0), (1.0, 0.0, pi / 2), (1.0, 1.0, pi),
    (0.0, 1.0, -pi / 2), (0.0, 0.0, 0.0),
]
octagon_waypoints = [
    (0.0, 0.0, 0.0), (0.5, 0.0, pi / 4),
    (0.853553, 0.353553, pi / 2), (0.853553, 0.853553, 3 * pi / 4),
    (0.5, 1.207107, pi), (0.0, 1.207107, -3 * pi / 4),
    (-0.353553, 0.853553, -pi / 2), (-0.353553, 0.353553, -pi / 4),
    (0.0, 0.0, 0.0),
]
POSITION_TOLERANCE = 0.05  # metres; accept within 15 cm
HEADING_TOLERANCE = math.radians(5)
MAX_FORWARD = 0.1
CAMERA_XYZ = (0.090, 0.0, 0.12)
CAMERA_RPY_DEG = (0.0, 0.0, 0.0)
CAMERA_FRAME = "camera_frame"

# Q_straight = np.array([
#     [6.36111111e-5, 2.58333333e-5, 1.16355283e-5],
#     [2.58333333e-5, 4.47222222e-5, 9.59931089e-5],
#     [1.16355283e-5, 9.59931089e-5, 4.26464388e-4]
# ])
# Q_straight = 2 * np.array([
#     [1.402778e-4, -3.333333e-5, 0.0],
#     [-3.333333e-5, 2.666667e-5, 0.0],
#     [0.0, 0.0, np.deg2rad(1.0)**2]
# ])
Q_straight = 2 * np.diag([
    0.05**2,             # Forward: 2.5 cm standard deviation
    0.03**2,             # Left/right: 1.5 cm standard deviation
    np.deg2rad(2.0)**2,    # Heading change: 2 degrees standard deviation
])
# Q_45 = 3 * np.diag([0.0, 0.01**2, 3.672332227e-3])
# Q_90 = 3 * np.diag([0.0, 0.0, 8.678211826e-3])
Q_45 = np.diag([0, 0, 0.015])
Q_90 = np.diag([0, 0, 0.02947759339])

R = np.diag([0.03**2, math.radians(2.5)**2])

u_straight = np.array([0.25, 0.0, 0.0])
u_45 = np.array([0.0, 0.0, math.radians(45)])
u_90 = np.array([0.0, 0.0, math.radians(90)])


def forward_model(distance):
    """Estimate the increment and covariance for the executed forward command."""
    fraction = distance / MAX_FORWARD
    return fraction * u_straight, fraction * Q_straight


def turn_model(angle):
    """Estimate signed motion from the left-turn calibration samples.

    Use piecewise linear interpolation of the mean and variance versus the
    commanded angle. Scale from zero below 45 degrees and extrapolate
    proportionally above 90 degrees. Assume right turns mirror left turns.
    """
    degrees = abs(math.degrees(angle))
    if degrees <= 45.0:
        fraction = degrees / 45.0
        u, q = fraction * u_45, fraction * Q_45
    elif degrees <= 90.0:
        fraction = (degrees - 45.0) / 45.0
        u = (1.0 - fraction) * u_45 + fraction * u_90
        q = (1.0 - fraction) * Q_45 + fraction * Q_90
    else:
        fraction = degrees / 90.0
        u, q = fraction * u_90, fraction * Q_90
    return (-u if angle < 0 else u), q


class Slam:
    """Joint state [robot x, y, yaw, tag x, y, ...] in the start frame."""

    def __init__(self):
        self.mu = np.zeros(3)
        self.P = np.zeros((3, 3))
        self.tag_index = {}

    def predict(self, u, Q):
        forward, left, delta_yaw = u
        c, s = math.cos(self.mu[2]), math.sin(self.mu[2])
        dx, dy = c * forward - s * left, s * forward + c * left
        F = np.eye(len(self.mu))
        F[0, 2], F[1, 2] = -dy, dx
        G = np.zeros((len(self.mu), 3))
        G[:3, :] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        self.mu[:3] += [dx, dy, delta_yaw]
        self.mu[2] = wrap(self.mu[2])
        self.P = F @ self.P @ F.T + G @ Q @ G.T

    def add_tag(self, tag_id, distance, bearing):
        x, y, yaw = self.mu[:3]
        c, s = math.cos(yaw + bearing), math.sin(yaw + bearing)
        # Jacobians of the inverse observation model.
        A = np.array([[1, 0, -distance * s], [0, 1, distance * c]])
        B = np.array([[c, -distance * s], [s, distance * c]])
        cross = A @ self.P[:3, :]
        tag_cov = A @ self.P[:3, :3] @ A.T + B @ R @ B.T
        self.tag_index[tag_id] = len(self.mu)
        self.mu = np.append(self.mu, [x + distance * c, y + distance * s])
        self.P = np.block([[self.P, cross.T], [cross, tag_cov]])

    def observe(self, tag_id, distance, bearing):
        if tag_id not in self.tag_index:
            self.add_tag(tag_id, distance, bearing)
            return

        i = self.tag_index[tag_id]
        dx, dy = self.mu[i:i + 2] - self.mu[:2]
        q = dx * dx + dy * dy
        if q < 1e-10:
            return  # Range/bearing linearization is singular at zero range.
        rho = math.sqrt(q)
        innovation = np.array([
            distance - rho,
            wrap(bearing - (math.atan2(dy, dx) - self.mu[2])),
        ])

        H = np.zeros((2, len(self.mu)))
        H[:, :3] = [[-dx / rho, -dy / rho, 0], [dy / q, -dx / q, -1]]
        H[:, i:i + 2] = [[dx / rho, dy / rho], [-dy / q, dx / q]]
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.mu += K @ innovation
        self.mu[2] = wrap(self.mu[2])
        A = np.eye(len(self.mu)) - K @ H
        updated = A @ self.P @ A.T + K @ R @ K.T
        self.P = (updated + updated.T) * 0.5


def setup_apriltag_tracking(robot):
    """Reuse the HW2 camera mount and /tf stream, using only tag centers."""
    from rclpy.qos import qos_profile_sensor_data
    from tf2_msgs.msg import TFMessage

    roll, pitch, yaw = map(math.radians, CAMERA_RPY_DEG)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    mount = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    optical_to_forward = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    camera_to_robot = mount @ optical_to_forward
    robot.measure_after = 0
    robot.frame_stamp = 0
    robot.observations = {}

    def on_tf(message):
        now = robot.get_clock().now().nanoseconds
        for item in message.transforms:
            name = item.child_frame_id
            if item.header.frame_id != CAMERA_FRAME or not name.startswith("tag_"):
                continue
            suffix = name[4:]
            if not suffix.isdecimal():
                continue
            stamp = item.header.stamp.sec * 10**9 + item.header.stamp.nanosec
            if stamp <= robot.measure_after or stamp < robot.frame_stamp:
                continue
            if not 0 <= now - stamp <= 3 * 10**9:
                continue
            p = item.transform.translation
            optical = np.array([p.x, p.y, p.z])
            if not np.isfinite(optical).all() or p.z <= 0:
                continue
            x, y, _ = camera_to_robot @ optical + CAMERA_XYZ
            distance = math.hypot(x, y)
            if not 0.05 < distance < 4.0:
                continue
            if stamp > robot.frame_stamp:
                robot.frame_stamp = stamp
                robot.observations.clear()
            robot.observations[int(suffix)] = (distance, math.atan2(y, x))

    robot.tag_subscription = robot.create_subscription(
        TFMessage, "/tf", on_tf, qos_profile_sensor_data)


def stationary_observations(robot):
    """Wait for fresh post-stop TF, collecting tags with the same image stamp."""
    robot.measure_after = robot.get_clock().now().nanoseconds
    robot.observations.clear()
    deadline = time.monotonic() + 3.0
    first_stamp = None
    observations = {}
    while rclpy.ok() and time.monotonic() < deadline:
        robot.send(0.0, 0.0)
        rclpy.spin_once(robot, timeout_sec=0.02)
        if robot.observations:
            if first_stamp is None:
                first_stamp = robot.frame_stamp
                # Some detectors publish each tag in a separate /tf message.
                deadline = min(deadline, time.monotonic() + 0.06)
            if robot.frame_stamp != first_stamp:
                break
            observations.update(robot.observations)
    return observations


def update_kalman_filter(robot, slam, trajectory, motion=None):
    if not rclpy.ok():
        raise KeyboardInterrupt
    if motion is not None:
        slam.predict(*motion)
    observations = stationary_observations(robot)
    if not rclpy.ok():
        raise KeyboardInterrupt
    for tag_id, (distance, bearing) in sorted(observations.items()):
        slam.observe(tag_id, distance, bearing)
    x, y, yaw = slam.mu[:3]
    trajectory.append((float(x), float(y)))
    print(f"EKF x={x:.3f} m  y={y:.3f} m  yaw={math.degrees(yaw):.1f} deg"
          f"  observed={sorted(observations)}", flush=True)


# ================= Real world POSITIONS (meters) =================
# We only use the Real world positions for plotting and error analysis. The EKF-SLAM algorithm does not use these values.!!!!!
# Ground-truth AprilTag positions in world coordinates.
# 
REAL_TAG_POSITIONS = {
    0: (2.43, 1.62),  # Example: (2.438, 1.62)
    1: (2.43, 0.81),  # Example: (1.50, 0.00)
    2: (1.62, 0),
    3: (0.81, 0),
    4: (0, 0.81),
    5: (0, 1.62),
    7: (0.81, 2.43),
    8: (1.62, 2.43),
}

REAL_START_POSITION = (0.81, 0.81)  # Replace with measured world (x, y)
# =================================================================


def print_graph(slam, trajectory, points, path, title):
    import math
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    # Assumes the initial robot frame and world frame have aligned axes.
    offset = np.asarray(REAL_START_POSITION, dtype=float)

    fig, (ax, info) = plt.subplots(
        1, 2,
        figsize=(14, max(7, 1.15 * len(slam.tag_index) + 3)),
        gridspec_kw={"width_ratios": [3, 2]},
        layout="constrained",
    )

    # Planned waypoints in world coordinates.
    if len(points):
        world_points = np.asarray(points, dtype=float)[:, :2] + offset
        ax.plot(
            world_points[:, 0],
            world_points[:, 1],
            "k--o",
            label="Planned waypoints",
        )

    # EKF trajectory in world coordinates.
    if len(trajectory):
        world_trajectory = (
            np.asarray(trajectory, dtype=float)[:, :2] + offset
        )
        ax.plot(
            world_trajectory[:, 0],
            world_trajectory[:, 1],
            ".-",
            label="EKF trajectory",
        )

    # Ground-truth tag positions are already in world coordinates.
    first_real_tag = True
    for tag_id, position in sorted(REAL_TAG_POSITIONS.items()):
        if position is None:
            continue

        real_x, real_y = position
        ax.scatter(
            real_x, real_y,
            marker="s",
            s=85,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            label="Ground-truth AprilTags" if first_real_tag else None,
        )
        first_real_tag = False

        ax.annotate(
            f"True tag {tag_id}",
            (real_x, real_y),
            xytext=(5, -13),
            textcoords="offset points",
        )

    info.axis("off")
    report = [
        "World coordinates and errors (m)",
        f"Robot start: ({offset[0]:.3f}, {offset[1]:.3f})",
        "Deviation = estimated - real",
        "Covariance (m²); rows/columns: x, y",
        "",
    ]

    for index, (tag_id, i) in enumerate(sorted(slam.tag_index.items())):
        # Translate estimated tag position into world coordinates.
        x, y = np.asarray(slam.mu[i:i + 2]).reshape(2) + offset

        # A fixed translation leaves covariance unchanged.
        cov = slam.P[i:i + 2, i:i + 2]
        variances, axes = np.linalg.eigh(cov)
        lengths = 2 * np.sqrt(5.991 * np.maximum(variances, 0))
        angle = math.degrees(math.atan2(axes[1, 1], axes[0, 1]))
        color = f"C{(index + 1) % 10}"

        ax.scatter(
            x, y,
            marker="x",
            color=color,
            s=65,
            zorder=5,
            label="Estimated AprilTags" if index == 0 else None,
        )
        ax.annotate(
            f"EKF tag {tag_id}",
            (x, y),
            xytext=(4, 5),
            textcoords="offset points",
            color=color,
        )
        ax.add_patch(Ellipse(
            (x, y),
            lengths[1],
            lengths[0],
            angle=angle,
            fill=False,
            edgecolor=color,
            label="95% tag covariance ellipse" if index == 0 else None,
        ))

        report.append(f"Tag {tag_id} EKF: ({x:.3f}, {y:.3f})")

        real_position = REAL_TAG_POSITIONS.get(tag_id)
        if real_position is not None:
            real_x, real_y = real_position

            dx = x - real_x
            dy = y - real_y
            position_error = math.sqrt(dx**2 + dy**2)

            ax.plot(
                [real_x, x],
                [real_y, y],
                ":",
                color=color,
                alpha=0.7,
            )

            report.extend([
                f"True: ({real_x:.3f}, {real_y:.3f})",
                f"X deviation: {dx:+.3f} m",
                f"Y deviation: {dy:+.3f} m",
                f"Position error: {position_error:.3f} m",
            ])
        else:
            report.append("True position / error: unavailable")

        report.extend([
            f"[[{cov[0, 0]: .3e}, {cov[0, 1]: .3e}],",
            f" [{cov[1, 0]: .3e}, {cov[1, 1]: .3e}]]",
            "",
        ])

    if not slam.tag_index:
        report.append("No AprilTags observed")

    info.text(
        0, 1,
        "\n".join(report),
        va="top",
        family="monospace",
        fontsize=9,
        transform=info.transAxes,
    )

    # Include both coordinate-frame origins in the plot limits.
    ax.update_datalim([(0.0, 0.0), tuple(offset)])
    ax.autoscale_view()

    # Small arrows, scaled relative to the map dimensions.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    frame_length = 0.08 * max(xmax - xmin, ymax - ymin, 1.0)

    def draw_frame(origin, color, frame_label, x_label, y_label,
                   label_offset):
        ox, oy = origin

        ax.scatter(
            ox, oy,
            s=20,
            color=color,
            zorder=8,
        )

        for endpoint in (
            (ox + frame_length, oy),
            (ox, oy + frame_length),
        ):
            ax.annotate(
                "",
                xy=endpoint,
                xytext=(ox, oy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.6,
                    mutation_scale=11,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=8,
            )

        ax.annotate(
            x_label,
            (ox + frame_length, oy),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=color,
            zorder=9,
        )
        ax.annotate(
            y_label,
            (ox, oy + frame_length),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=color,
            zorder=9,
        )
        ax.annotate(
            frame_label,
            (ox, oy),
            xytext=label_offset,
            textcoords="offset points",
            ha="right" if label_offset[0] < 0 else "left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=color,
            zorder=9,
        )

        ax.update_datalim([
            (ox, oy),
            (ox + frame_length, oy),
            (ox, oy + frame_length),
        ])

    # World frame at world (0, 0).
    draw_frame(
        origin=(0.0, 0.0),
        color="gray",
        frame_label="{W}",
        x_label=r"$X_w$",
        y_label=r"$Y_w$",
        label_offset=(-7, -7),
    )

    # Initial robot frame at the measured starting position.
    draw_frame(
        origin=offset,
        color="red",
        frame_label="{R}",
        x_label=r"$X_r$",
        y_label=r"$Y_r$",
        label_offset=(7, -20),
    )

    ax.autoscale_view()
    ax.margins(0.12)

    ax.set(
        xlabel="World x (m)",
        ylabel="World y (m)",
        title=title,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.savefig(path, dpi=160)
    plt.close(fig)

    print("\n".join(report))
    print(f"Saved {path}")


def follow_waypoint(robot, slam, trajectory, waypoint):
    tx, ty, target_heading = waypoint
    for _ in range(30):
        update_kalman_filter(robot, slam, trajectory)
        x, y, yaw = slam.mu[:3]
        distance = math.hypot(tx - x, ty - y)

        if distance <= POSITION_TOLERANCE:
            break
        correction = wrap(math.atan2(ty - y, tx - x) - yaw)
        robot.turn(correction)  
        update_kalman_filter(robot, slam, trajectory, turn_model(correction))
        update_kalman_filter(robot, slam, trajectory)  # After turn, get new observations.
        #The turn observation may have changed position or heading.
        x, y, yaw = slam.mu[:3] # could consider deleting to me more simple
        distance = math.hypot(tx - x, ty - y)
        step = min(MAX_FORWARD, distance)
        robot.forward(step) 
        update_kalman_filter(robot, slam, trajectory, forward_model(step))
        update_kalman_filter(robot, slam, trajectory)
    if math.hypot(tx - slam.mu[0], ty - slam.mu[1]) > POSITION_TOLERANCE:
        raise RuntimeError(f"Position tolerance not reached at ({tx}, {ty})")

    for _ in range(10):
        correction = wrap(target_heading - slam.mu[2])
        if abs(correction) <= HEADING_TOLERANCE:
            break
        print(f"Waypoint heading correction: {math.degrees(correction):.1f} degrees")
        robot.turn(correction) 
        update_kalman_filter(robot, slam, trajectory, turn_model(correction))
        update_kalman_filter(robot, slam, trajectory)
    if abs(wrap(target_heading - slam.mu[2])) > HEADING_TOLERANCE:
        raise RuntimeError(f"Heading tolerance not reached at ({tx}, {ty})")


def main():
    parser = argparse.ArgumentParser(description="HW3 AprilTag EKF-SLAM")
    parser.add_argument("route", choices=("square", "octagon"), nargs="?", default="square")
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("hw3_logs"))
    args = parser.parse_args()
    if args.loops < 1:
        parser.error("--loops must be at least 1")
    points = square_waypoints if args.route == "square" else octagon_waypoints

    values = (FORWARD_SPEED, TURN_RATE, FORWARD_LEFT_COMMAND,
              FORWARD_RIGHT_COMMAND, TURN_COMMAND)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Calibration values must be positive and finite")
    if max(FORWARD_LEFT_COMMAND, FORWARD_RIGHT_COMMAND, TURN_COMMAND) > 0.5:
        raise ValueError("Motor commands must be <= 0.5")

    # Check the plot dependency and output directory before enabling movement.
    import matplotlib  # noqa: F401
    output = args.output / f"{args.route}_{time.time_ns()}"
    output.mkdir(parents=True)
    rclpy.init()
    robot = Robot()
    slam = Slam()
    trajectory = []
    completed = 0
    try:
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and robot.publisher.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                raise RuntimeError("Start motor_control.py first")
            rclpy.spin_once(robot, timeout_sec=0.05)

        setup_apriltag_tracking(robot)
        update_kalman_filter(robot, slam, trajectory)  # Initial frame at (0, 0, 0).
        for lap in range(1, args.loops + 1):
            print(f"Lap {lap}/{args.loops}: {args.route}", flush=True)
            for waypoint in points[1:]:  # The first entry defines the start.
                follow_waypoint(robot, slam, trajectory, waypoint)
            completed = lap
            print_graph(slam, trajectory, points, output / f"lap_{lap:02d}.png",
                        f"{args.route.title()} SLAM after lap {lap}")
    except KeyboardInterrupt:
        print("Interrupted; stopping the robot.")
    finally:
        if rclpy.ok():
            robot.send(0.0, 0.0)
            robot.stop()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if trajectory:
            print_graph(slam, trajectory, points, output / "trajectory_final.png",
                        f"{args.route.title()} SLAM: {completed} completed laps")


if __name__ == "__main__":
    main()

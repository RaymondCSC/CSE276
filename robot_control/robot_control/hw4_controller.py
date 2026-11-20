"""HW2 timed motor commands and AprilTag localization with HW4 XY waypoints."""
import csv
import math
import time
from collections import Counter

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage

from hw4_plan import FT, TAGS, arguments, plan

FORWARD_LEFT_COMMAND = 0.40
FORWARD_RIGHT_COMMAND = 0.40 #when speed grow command muust grow at the same time for same distance
FORWARD_SPEED = 0.555       # m/s EG not exactly        # m/s EG not exactly 

TURN_COMMAND = 0.5        
TURN_RATE = 3.62   
POSITION_TOLERANCE = 0.08  # m; one grid step is 0.1524 m
CAMERA_XYZ = (0.09, 0., 0.12)
CAMERA_FRAME = "camera_frame"
TAG_HEIGHT, TAG_SPIN = 0.11, 90  # m, degrees CCW facing the print

MAX_POSE_ERROR = 0.40   # m; gate vs dead reckoning (a 0.2 m step + turn error + tag bias)
SETTLE_TIME = 0.5       # seconds
NUM_SAMPLES = 5         # stop collecting once this many fresh frames are in
MIN_SAMPLES = 3         # a fix needs at least this many frames that agree
CLUSTER_RADIUS = 0.10   # m; frames must agree with each other to this
RELAX_GATE_AFTER = 2    # failed attempts before the dead-reckoning gate is dropped
MAX_SEARCH_TURNS = 18   # 18 x 20 deg = one full circle

def wrap(angle):
    return math.atan2(math.sin(angle),math.cos(angle))

class Robot(Node):
    def __init__(self):
        super().__init__("waypoint_controller")
        self.publisher = self.create_publisher(Float32MultiArray,"motor_commands",10)
        self.total_translation = self.total_rotation = 0.
        self.predicted_pose = None
        self.use_gate = True
        self.reject = Counter()   # why tag frames were thrown away

    def pause(self, seconds):
        """Stand still while still processing callbacks (time.sleep would starve /tf)."""
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            self.send(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.02)

    def send(self,left,right):
        msg = Float32MultiArray()
        msg.data = [float(left),float(right)]
        self.publisher.publish(msg)

    def move(self,left,right,seconds):
        end = time.monotonic()+seconds
        while rclpy.ok() and time.monotonic()<end:
            self.send(left,right)
            rclpy.spin_once(self,timeout_sec=0.02)
        self.stop()

    def stop(self):
        end = time.monotonic()+0.2
        while rclpy.ok() and time.monotonic()<end:
            self.send(0,0)
            rclpy.spin_once(self,timeout_sec=0.02)

    def forward(self,distance):
        self.move(FORWARD_LEFT_COMMAND,FORWARD_RIGHT_COMMAND,distance/FORWARD_SPEED)
        self.total_translation += abs(distance)

        if self.predicted_pose is not None:
            x, y, theta = self.predicted_pose
            self.predicted_pose = (
                x + distance * math.cos(theta),
                y + distance * math.sin(theta),
                theta
            )

    def turn(self,angle):
        angle = wrap(angle)
        if abs(angle)<0.001:
            return
        command = math.copysign(TURN_COMMAND,angle)
        self.move(-command,command,abs(angle)/TURN_RATE)
        self.total_rotation += abs(angle)

        if self.predicted_pose is not None:
            x, y, theta = self.predicted_pose
            self.predicted_pose = (x, y, wrap(theta + angle))


def transform(R,xyz):
    T = np.eye(4)
    T[:3,:3], T[:3,3] = R,xyz
    return T

def inverse(T):
    R = T[:3,:3].T
    return transform(R,-R @ T[:3,3])

def setup_apriltag_tracking(robot):
    world_tags = {}
    s = math.radians(TAG_SPIN)
    spin = np.array([[math.cos(s),-math.sin(s),0],[math.sin(s),math.cos(s),0],[0,0,1]])
    for tag,(x,y,facing) in TAGS.items():
        a = math.radians(facing)
        R = np.array([[-math.sin(a),0,math.cos(a)],[math.cos(a),0,math.sin(a)],[0,1,0]])
        world_tags[f"tag_{tag}"] = transform(R @ spin,(x*FT,y*FT,TAG_HEIGHT))
    optical_to_forward = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    camera_robot = inverse(transform(optical_to_forward,CAMERA_XYZ))
    robot.latest_pose = None
    robot.pose_stamp = robot.measure_after = -math.inf
    robot.samples = []

    def on_tf(message):
        now = robot.get_clock().now().nanoseconds/1e9
        candidates = []
        print("Observed tags:", [
            item.child_frame_id for item in message.transforms
            if item.header.frame_id == CAMERA_FRAME and item.child_frame_id in world_tags
        ])
        for item in message.transforms:
            if item.header.frame_id != CAMERA_FRAME or item.child_frame_id not in world_tags:
                continue
            robot.reject["tag_seen"] += 1
            stamp = item.header.stamp.sec + item.header.stamp.nanosec/1e9
            if stamp <= robot.pose_stamp or not 0 <= now-stamp <= 3:
                robot.reject["stale_or_duplicate"] += 1
                continue
            q = item.transform.rotation
            v = np.array([q.x,q.y,q.z,q.w])
            norm = np.linalg.norm(v)
            if not np.isfinite(norm) or norm < 1e-8:
                robot.reject["bad_quaternion"] += 1
                continue
            x,y,z,w = v/norm
            R = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                          [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                          [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
            p = item.transform.translation
            camera_tag = transform(R,(p.x,p.y,p.z))
            distance = np.linalg.norm(camera_tag[:3,3])
            if p.z <= 0 or not 0.05 < distance < 4:
                robot.reject["range"] += 1
                continue
            T = world_tags[item.child_frame_id] @ inverse(camera_tag) @ camera_robot
            if abs(T[2,3]) > 0.20 or T[2,2] < math.cos(math.radians(25)):
                robot.reject["height_or_tilt"] += 1
                continue
            pose = (float(T[0,3]),float(T[1,3]),math.atan2(T[1,0],T[0,0]))

            # Reject measurements outside the arena
            if not (0 <= pose[0] <= 8 * FT and
                    0 <= pose[1] <= 8 * FT):
                robot.reject["outside_arena"] += 1
                continue

            # Reject measurements far from dead reckoning
            if robot.use_gate and robot.predicted_pose is not None:
                error = math.hypot(
                    pose[0] - robot.predicted_pose[0],
                    pose[1] - robot.predicted_pose[1]
                )
                if error > MAX_POSE_ERROR:
                    robot.reject["dead_reckoning_gate"] += 1
                    continue

            robot.reject["accepted"] += 1
            candidates.append((stamp,distance,pose))
        if candidates:
            stamp,_,pose = min(candidates,key=lambda c:(-c[0],c[1]))
            robot.pose_stamp = stamp
            robot.samples.append((stamp,*pose))
            if stamp > robot.measure_after:
                robot.latest_pose = pose

    robot.tag_subscription = robot.create_subscription(TFMessage,"/tf",on_tf,qos_profile_sensor_data)

# def get_current_location(robot):
#     """Wait 3 seconds; if lost, turn 20 degrees CCW and wait 1 second repeatedly."""
#     # Invalidate cached observations, including images captured while moving.
#     robot.latest_pose = None
#     robot.measure_after = robot.get_clock().now().nanoseconds / 1e9
#     deadline = time.monotonic() + 3.0
#     while rclpy.ok() and time.monotonic() < deadline:
#         robot.send(0.0, 0.0)
#         rclpy.spin_once(robot, timeout_sec=0.02)
#         if robot.latest_pose is not None:
#             return robot.latest_pose

#     while rclpy.ok():
#         print("No tag pose: turning 20 degrees CCW, then waiting 1 second")
#         robot.turn(math.radians(20))
#         robot.latest_pose = None
#         robot.measure_after = robot.get_clock().now().nanoseconds / 1e9
#         deadline = time.monotonic() + 1.0
#         # Process callbacks during the wait, instead of blocking with time.sleep.
#         while rclpy.ok() and time.monotonic() < deadline:
#             robot.send(0.0, 0.0)
#             rclpy.spin_once(robot, timeout_sec=0.02)
#         if robot.latest_pose is not None:
#             return robot.latest_pose
#     raise RuntimeError("ROS shut down while waiting for an AprilTag pose")

def collect_observations(robot, window=3.0):
    """Stand still and gather up to NUM_SAMPLES fresh, distinct tag poses."""
    robot.pause(SETTLE_TIME)
    robot.latest_pose = None
    robot.measure_after = robot.get_clock().now().nanoseconds / 1e9
    robot.reject.clear()          # counters now describe this attempt only
    observations = []
    last_stamp = robot.pose_stamp
    deadline = time.monotonic() + window
    while rclpy.ok() and time.monotonic() < deadline:
        robot.send(0.0, 0.0)
        rclpy.spin_once(robot, timeout_sec=0.02)
        if robot.latest_pose is not None and robot.pose_stamp > last_stamp:
            observations.append(robot.latest_pose)
            last_stamp = robot.pose_stamp
            if len(observations) >= NUM_SAMPLES:
                break
    return observations

def consistent_fix(observations):
    """Median pose of the largest group of frames that agree within CLUSTER_RADIUS."""
    best = []
    for cx, cy, _ in observations:
        group = [p for p in observations if math.hypot(p[0]-cx, p[1]-cy) <= CLUSTER_RADIUS]
        if len(group) > len(best):
            best = group
    if len(best) < MIN_SAMPLES:
        return None
    x = float(np.median([p[0] for p in best]))
    y = float(np.median([p[1] for p in best]))
    theta = math.atan2(sum(math.sin(p[2]) for p in best),
                       sum(math.cos(p[2]) for p in best))
    return x, y, theta, len(best)

def get_current_location(robot):
    """Median of agreeing stationary frames; if lost, spin 20 deg CCW and retry.

    After RELAX_GATE_AFTER failures the dead-reckoning gate is dropped, so a
    wrong earlier fix cannot lock out every correct measurement."""
    failures = 0
    try:
        while rclpy.ok():
            robot.use_gate = failures < RELAX_GATE_AFTER
            observations = collect_observations(robot)
            fix = consistent_fix(observations)
            if fix is not None:
                x, y, theta, n = fix
                print(f"Fix ({x/FT:.2f}, {y/FT:.2f}) ft, yaw {math.degrees(theta):.1f} deg "
                      f"from {n}/{len(observations)} frames; rejections {dict(robot.reject)}")
                robot.predicted_pose = (x, y, theta)
                return x, y, theta
            failures += 1
            print(f"Localization attempt {failures} failed: {len(observations)} frames, "
                  f"rejections {dict(robot.reject)}, gate {'on' if robot.use_gate else 'off'}")
            if failures > MAX_SEARCH_TURNS:
                raise RuntimeError("Localization failed after a full search rotation")
            robot.turn(math.radians(20))
    finally:
        robot.use_gate = True
    raise RuntimeError("ROS shut down while waiting for an AprilTag pose")

def main():
    args = arguments().parse_args()
    route,_ = plan(args.mode,args.start,args.goal)
    print("Planned waypoints (ft):",route)
    rclpy.init()
    robot = Robot()
    try:
        setup_apriltag_tracking(robot)
        deadline = time.monotonic()+5
        while robot.publisher.get_subscription_count()==0:
            if time.monotonic()>deadline:
                raise RuntimeError("Start motor_control.py first")
            rclpy.spin_once(robot,timeout_sec=0.05)
        start_time = time.monotonic()
        x,y,heading = get_current_location(robot)
        #if math.dist((x,y),tuple(v*FT for v in route[0])) > POSITION_TOLERANCE:
        #    raise ValueError(f"Measured start ({x/FT:.2f}, {y/FT:.2f}) ft differs from planned start {route[0]}")
        for wx,wy in route[1:]:
            dx,dy = wx*FT-x, wy*FT-y
            while math.hypot(dx,dy)>POSITION_TOLERANCE:
                print(f"Heading current: {math.degrees(heading):.1f} deg, target: {math.degrees(math.atan2(dy,dx)):.1f} deg, error: {math.degrees(wrap(math.atan2(dy,dx)-heading)):.1f} deg")
                print(f"Current position: ({x/FT:.2f}, {y/FT:.2f}) ft, target: ({wx:.2f}, {wy:.2f}) ft, error: {math.hypot(dx,dy):.3f} m")
                robot.turn(wrap(math.atan2(dy,dx)-heading))
                robot.pause(0.5)
                robot.forward(min(0.20, math.hypot(dx,dy)))
                robot.pause(0.5)
                x,y,heading = get_current_location(robot)
                dx,dy = wx*FT-x, wy*FT-y
            print(f"Reached ({wx:.2f}, {wy:.2f}) ft; position error {math.hypot(dx,dy):.3f} m")
        print(f"Elapsed {time.monotonic()-start_time:.1f} s; commanded translation "
              f"{robot.total_translation:.3f} m, rotation {robot.total_rotation:.3f} rad")
    finally:
        if rclpy.ok():
            robot.stop()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if getattr(robot,"samples",None):
            with open(f"hw4_{args.mode}.csv","w",newline="") as f:
                writer = csv.writer(f)
                writer.writerow(("stamp_s","x_m","y_m","yaw_rad"))
                writer.writerows(robot.samples)
            from hw4_plot import draw
            draw(args.mode,args.start,args.goal,[(x/FT,y/FT) for _,x,y,_ in robot.samples])

if __name__ == "__main__":
    main()

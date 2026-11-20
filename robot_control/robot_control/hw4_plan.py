"""HW4 world and planners. Geometry is in feet. The controller converts to SI."""
import argparse
import heapq
import math
from itertools import count
from itertools import combinations

FT = 0.3048
SIZE = 8.0
MARGIN = 0.20 / FT
BOX = (3., 5., 3., 5.)  # xmin, xmax, ymin, ymax
INFLATED = (3-MARGIN, 5+MARGIN, 3-MARGIN, 5+MARGIN)
START, GOAL = (7, 1), (1, 7)
# x, y, facing degrees. Height and printed spin are configured in the controller.
# (world x in ft, world y in ft, direction the printed face points in degrees)
TAGS = {
    0:  (1, 0,   90),   # bottom wall, faces +y
    5:  (7, 0,   90),

    4:  (8, 1,  180),   # right wall, faces -x
    2:  (8, 7,  180),

    7:  (1, 8,  -90),   # top wall, faces -y
    8:  (7, 8,  -90),

    1:  (0, 1,    0),   # left wall, faces +x
    3:  (0, 7,    0),

    9:  (4, 3,  -90),   # bottom face of box, faces -y
    10: (5, 4,    0),   # right face of box, faces +x
    11: (3, 4,  180),   # left face of box, faces -x
    12: (4, 5,   90),   # top face of box, faces +y
}

def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("distance", "safety"))
    p.add_argument("--start", nargs=2, type=float, default=START, metavar=("X_FT","Y_FT"))
    p.add_argument("--goal", nargs=2, type=float, default=GOAL, metavar=("X_FT","Y_FT"))
    return p

def crosses(a, b):
    """Overlap the open x and y interior intervals along a + t*(b-a)."""
    enter, leave = 0., 1.
    for start, end, low, high in zip(a, b, INFLATED[::2], INFLATED[1::2]):
        delta = end - start
        if delta == 0:
            if not low < start < high:
                return False
        else:
            first, last = sorted(((low-start)/delta, (high-start)/delta))
            enter, leave = max(enter, first), min(leave, last)
            if enter >= leave:
                return False
    return True

def allowed(p):
    return all(MARGIN <= v <= SIZE-MARGIN for v in p) and not crosses(p,p)

def path(parent, goal):
    route = []
    while goal is not None:
        route.append(goal)
        goal = parent[goal]
    return route[::-1]

def visibility(start, goal):
    left, right, bottom, top = INFLATED
    nodes = dict.fromkeys([start, goal, (left,bottom), (left,top), (right,bottom), (right,top)])
    graph = {v: [] for v in nodes}
    edges = []
    for a,b in combinations(nodes, 2):
        if not crosses(a,b):
            graph[a].append(b)
            graph[b].append(a)
            edges.append((a,b))
    dist, parent, queue = {start:0.}, {start:None}, [(0.,start)]
    while queue:
        cost, a = heapq.heappop(queue)
        if cost != dist[a]:
            continue
        if a == goal:
            return path(parent,goal), edges
        for b in graph[a]:
            candidate = cost + math.dist(a,b)
            if candidate < dist.get(b,math.inf):
                dist[b], parent[b] = candidate, a
                heapq.heappush(queue,(candidate,b))
    raise ValueError("No visibility path")



def safety(start, goal, *, seed=276):
    """Clearance-aware RRT: prefer distance from both box and walls."""
    import random

    rng = random.Random(seed)
    start, goal = tuple(start), tuple(goal)

    clearance_min = MARGIN  # 20 cm
    left, right, bottom, top = BOX

    def clearance(p):
        x, y = p

        # Distance to central obstacle
        box = math.hypot(
            max(left - x, x - right, 0),
            max(bottom - y, y - top, 0)
        )

        # Distance to arena walls
        wall = min(x, SIZE - x, y, SIZE - y)

        return min(box, wall)

    def edge_cost(a, b):
        """Integrate safety penalty along an edge."""
        if crosses(a, b):
            return math.inf

        length = math.dist(a, b)
        n = max(2, math.ceil(length / 0.10))
        total = 0.0

        for i in range(n + 1):
            t = i / n
            p = (
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1])
            )

            d = clearance(p)

            if d < clearance_min - 1e-9:
                return math.inf

            # Larger penalty when close to hazards
            weight = 1.0 + 8.0 / (d + 0.25)**2
            total += (0.5 if i in (0, n) else 1.0) * weight

        return total * length / n

    if not allowed(start) or not allowed(goal):
        raise ValueError("Start or goal violates safety margin")

    if start == goal:
        return [start], []

    parents = {start: None}
    costs = {start: 0.0}

    best_route = None
    best_cost = math.inf

    # Continue sampling instead of returning the first path
    for _ in range(2000):
        if rng.random() < 0.10:
            sample = goal
        else:
            sample = (
                rng.uniform(MARGIN, SIZE - MARGIN),
                rng.uniform(MARGIN, SIZE - MARGIN)
            )

        near = min(parents, key=lambda p: math.dist(p, sample))
        dist = math.dist(near, sample)

        if dist < 1e-9:
            continue

        # Extend at most 0.5 ft
        t = min(1.0, 0.5 / dist)
        node = (
            near[0] + t * (sample[0] - near[0]),
            near[1] + t * (sample[1] - near[1])
        )

        if node in parents or not math.isfinite(edge_cost(near, node)):
            continue

        # Choose a low-cost parent from nearby nodes
        possible = sorted(
            parents, key=lambda p: math.dist(p, node)
        )[:20]

        choices = [
            (costs[p] + edge_cost(p, node), p)
            for p in possible
        ]

        node_cost, parent = min(choices)

        if not math.isfinite(node_cost):
            continue

        parents[node] = parent
        costs[node] = node_cost

        # Keep searching if a feasible route is found
        goal_edge = edge_cost(node, goal)

        if node_cost + goal_edge < best_cost:
            best_cost = node_cost + goal_edge
            best_route = path(parents, node) + [goal]

    if best_route is None:
        raise ValueError("RRT could not find a safe route")

    # Shortcut only if it improves safety-weighted cost
    changed = True

    while changed:
        changed = False

        for i in range(len(best_route) - 2):
            for j in range(len(best_route) - 1, i + 1, -1):
                old = sum(
                    edge_cost(best_route[k], best_route[k + 1])
                    for k in range(i, j)
                )
                new = edge_cost(best_route[i], best_route[j])

                if new + 1e-5 < old:
                    best_route = (
                        best_route[:i + 1] + best_route[j:]
                    )
                    changed = True
                    break

            if changed:
                break

    return best_route, []


def plan(mode,start,goal):
    start,goal = tuple(start),tuple(goal)
    if not all(map(allowed,(start,goal))):
        raise ValueError("Endpoint violates the obstacle or wall margin")
    return {"distance":visibility,"safety":safety}[mode](start,goal)

if __name__ == "__main__":
    args = arguments().parse_args()
    route,_ = plan(args.mode,args.start,args.goal)
    print("Waypoints (ft):",route)
    print("Length (m):",round(sum(math.dist(a,b) for a,b in zip(route,route[1:]))*FT,3))

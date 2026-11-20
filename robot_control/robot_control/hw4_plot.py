"""Plot an HW4 route and optional AprilTag position samples."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import MultipleLocator

from hw4_plan import BOX, FT, INFLATED, MARGIN, SIZE, TAGS, arguments, plan


def draw(mode, start, goal, samples=(), output=None):
    route, edges = plan(mode, start, goal)
    fig, ax = plt.subplots(figsize=(8, 8))

    left, right, bottom, top = BOX
    if mode == "safety":
        clearance = 0.20 / FT  # feet, same centerline threshold as safety()
        # Four quarter circles describe the Euclidean buffer around the box.
        buffer_points = []
        for x, y, first_angle in (
            (right, top, 0),
            (left, top, 90),
            (left, bottom, 180),
            (right, bottom, 270),
        ):
            for degrees in range(first_angle, first_angle + 91, 3):
                angle = math.radians(degrees)
                buffer_points.append(
                    (x + clearance * math.cos(angle), y + clearance * math.sin(angle))
                )
        ax.add_patch(
            Polygon(buffer_points, facecolor="#ffdb99", label="20 cm obstacle buffer")
        )
        ax.add_patch(
            Rectangle(
                (MARGIN, MARGIN), SIZE - 2 * MARGIN, SIZE - 2 * MARGIN,
                fill=False, edgecolor="0.55", linestyle=":", label="Wall margin",
            )
        )
        title = "HW4 safety route (RRT)"
    else:
        l, r, b, t = INFLATED
        ax.add_patch(
            Rectangle((l, b), r - l, t - b, facecolor="#ffdb99", label="20 cm inflation")
        )
        title = "HW4 distance route (visibility graph)"

    ax.add_patch(
        Rectangle((left, bottom), right - left, top - bottom,
                  facecolor="#92999e", label="Obstacle")
    )
    ax.add_patch(Rectangle((0, 0), SIZE, SIZE, fill=False))
    for i, (a, b) in enumerate(edges):
        ax.plot(*zip(a, b), color="0.7", lw=1, zorder=2,
                label="Visibility edges" if i == 0 else None)
    ax.plot(*zip(*route), "o--", ms=4, lw=1.6, color="navy",
            label="Planned route", zorder=3)

    if len(samples):
        ax.plot(*zip(*samples), ".-", color="crimson",
                label="AprilTag samples (joined)", zorder=4)

    for tag_id, (x, y, _) in TAGS.items():
        ax.plot(x, y, "k+", ms=8)
        ax.annotate(f"LM{tag_id}", (x, y), xytext=(4, 4), textcoords="offset points")
    for point, marker, label in ((start, "s", "Start"), (goal, "*", "Goal")):
        ax.scatter(*point, marker=marker, s=100, label=label, zorder=5)

    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set(
        xlim=(-0.35, SIZE + 0.55),
        ylim=(-0.35, SIZE + 0.55),
        xlabel="World x (ft)",
        ylabel="World y (ft)",
        aspect="equal",
        title=title,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    fig.savefig(output or f"hw4_{mode}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = arguments()
    parser.add_argument("--track", type=Path, help="CSV from the controller")
    args = parser.parse_args()

    samples = []
    if args.track:
        with args.track.open(newline="") as f:
            samples = [
                (float(row["x_m"]) / FT, float(row["y_m"]) / FT)
                for row in csv.DictReader(f)
            ]
    draw(args.mode, args.start, args.goal, samples)

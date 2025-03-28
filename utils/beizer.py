import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def compute_bezier_points(control_points):
    t = np.linspace(0, 1, 100)
    n = len(control_points) - 1
    curve_points = np.zeros((len(t), 2))

    for i, point in enumerate(control_points):
        curve_points += np.outer(comb(n, i) * ((1 - t) ** (n - i)) * (t ** i), point)

    return curve_points

def convert_to_bezier_bounding_box(control_points):
    # Compute the 16-point Bézier bounding box
    bezier_bb_points = []
    for i in range(4):
        p1 = control_points[i]
        p2 = control_points[(i + 1) % 4]
        p3 = control_points[(i + 2) % 4]
        p4 = control_points[(i + 3) % 4]

        p0_tangent = (p2 - p1) * 0.1
        p1_tangent = (p3 - p2) * 0.1
        p2_tangent = (p4 - p3) * 0.1

        bezier_bb_points.append(p1)
        bezier_bb_points.append(p1 + p0_tangent)
        bezier_bb_points.append(p2 - p1_tangent)
        bezier_bb_points.append(p2)

    return np.array(bezier_bb_points)

# Define the four control points of the bounding box
control_points = np.array([[46, 453], [231, 453], [231, 521], [46, 521]])

# Convert to Bézier bounding box
bezier_bb_points = convert_to_bezier_bounding_box(control_points)
print(bezier_bb_points)

# Plot the original control points and the Bézier bounding box
plt.figure(figsize=(5, 5))
plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')
plt.plot(bezier_bb_points[:, 0], bezier_bb_points[:, 1], 'b-', label='Bézier Bounding Box')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Four-Point Bounding Box to 16-Point Bézier Bounding Box')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

import reprojection_kinematics as rpk
import numpy as np
import cv2
import matplotlib.pyplot as plt

def project_point_camera(p_cam, K):
    """
    Simple analytic projection of a 3D point in camera coords to image coords.
    Returns (u, v, Z)
    """
    X, Y, Z = p_cam
    eps = 1e-9
    if abs(Z) < eps:
        # avoid huge numbers; still treat as behind if negative
        Z = eps if Z >= 0 else -eps
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return float(u), float(v), float(Z)

def draw_arrow_at_border(img, center, edge_pt, angle, color=(0,0,255), size=30):
    """
    Draw a filled triangle arrow with its tip at edge_pt pointing toward center.
    angle is arctan2(v-center_v, u-center_u) (screen coords).
    """
    tip = np.array(edge_pt, dtype=np.float32)
    # unit vector from center -> tip (in image coords where y grows down)
    ux = np.cos(angle)
    uy = np.sin(angle)
    unit = np.array([ux, uy], dtype=np.float32)
    # base center is slightly behind the tip (so the arrow points inward)
    base_center = tip - unit * size
    # perpendicular vector for base width
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
    half_width = size * 0.5
    p1 = tip
    p2 = base_center + perp * half_width
    p3 = base_center - perp * half_width
    pts = np.array([p1, p2, p3], dtype=np.int32)
    cv2.fillConvexPoly(img, pts, color)

def main():
    # Temporary calibration - replace with your real file if you want
    calib_path = "rgb1080_camera_calibration.npz"  # Path to your calibration file
    width, height = 1920, 1080  # Example dimensions, adjust as needed

    pr = rpk.PointReprojection(calib_path, intrinsics_width=width, intrinsics_height=height)

    # Create canvas
    img = np.full((height, width, 3), 200, dtype=np.uint8)  # light gray bg
    cx, cy = width // 2, height // 2  # principal point at image center

    # Draw center cross
    cv2.line(img, (int(cx)-10, int(cy)), (int(cx)+10, int(cy)), (150,150,150), 1)
    cv2.line(img, (int(cx), int(cy)-10), (int(cx), int(cy)+10), (150,150,150), 1)

    # Example camera-frame 3D points to visualize (X right, Y down, Z forward)
    test_points = [
        (0.0, 0.0, 5.0),    # center, in front
        (1.0, 0.0, 5.0),    # right, in front
        (-1.5, 0.0, 5.0),   # left, in front
        (3.5, 0.0, 5.0),    # far right, likely offscreen
        (0.0, 0.0, -5.0),   # behind camera (center)
        (2.5, 0.0, -5.0),   # behind camera to the right
        (0.0, 2.0, 2.0),    # downwards in front
        (0.0, -3.0, 4.0),   # upward (negative Y) in front (may go off top)
    ]

    for i, p in enumerate(test_points):
        u_proj, v_proj, Z = project_point_camera(p, K)
        label = f"P{i}"

        inbounds = (Z > 0 and 0 <= u_proj < width and 0 <= v_proj < height)
        if inbounds:
            # draw green circle at projected pixel
            cv2.circle(img, (int(round(u_proj)), int(round(v_proj))), 6, (0,255,0), -1)
            cv2.putText(img, label, (int(round(u_proj))+8, int(round(v_proj))-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        else:
            # offscreen or behind -> compute marker on border
            behind = (Z <= 0)
            # If behind, mirror the projected point around principal point as UX suggests
            u_for_marker = u_proj
            v_for_marker = v_proj
            if behind:
                u_for_marker = 2*cx - u_proj
                v_for_marker = 2*cy - v_proj

            # get intersection with image border using same logic as your get_oob_marker_vector
            # We'll call into the class helper to reuse its logic:
            (edge_u, edge_v), theta = pr.get_oob_marker_vector(u_for_marker, v_for_marker, zoom_level=0, center_u=cx, center_v=cy, behind_camera=False)
            # pr.get_oob_marker_vector returns ints for u,v — ensure within bounds
            edge_u = max(0, min(width-1, int(edge_u)))
            edge_v = max(0, min(height-1, int(edge_v)))

            # draw arrow at edge pointing toward center
            draw_arrow_at_border(img, (cx, cy), (edge_u, edge_v), theta, color=(0,0,255), size=28)

            # annotate
            txt = f"{label}{' (behind)' if behind else ' (oob)'}"
            # place label slightly inward from edge
            inset_x = int(edge_u - math.cos(theta)*40)
            inset_y = int(edge_v - math.sin(theta)*40)
            inset_x = max(0, min(width-1, inset_x))
            inset_y = max(0, min(height-1, inset_y))
            cv2.putText(img, txt, (inset_x-10, inset_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # Save and show
    out_fname = "projection_visualization.png"
    cv2.imwrite(out_fname, img)
    print(f"Saved visualization to {out_fname}")

    # Display window (press any key to close)
    cv2.imshow("Projection visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
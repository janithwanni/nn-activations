import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def decision_boundary(x, displ, freq, amp):
    # print(f"{amp} * np.sin({freq} * x) + {displ}")
    return amp * np.sin(freq * x) + displ

def rotate(x, y, t):
    theta = np.radians(t)
    # Compute center of the plane
    center_x = (x.max() + x.min()) / 2
    center_y = (y.max() + y.min()) / 2

    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Shift points to center the plane
    x_shifted = x - center_x
    y_shifted = y - center_y

    # Apply the rotation
    rotated_points = np.dot(rotation_matrix, np.vstack((x_shifted, y_shifted)))

    # Shift points back to the original plane
    x_rotated = rotated_points[0, :] + center_x
    y_rotated = rotated_points[1, :] + center_y

    return x_rotated, y_rotated

def generate_data(n_samples, boundary_params, rotation_angle, margin_params, overlap, space_params, seed= 12345):
    """
        n_samples: int. Number of samples to generate (might be lower since we are sampling)
        boundary_params: dict containing displ, freq, amp
        margin_params: dict containing margin, margin_sample, outmargin_sample
        space_params: dict containing x_lower, y_lower, x_upper, y_upper
        overlap: float distance to overlap two opposing classes
        rotation_angle: angle to rotate the decision boundary in degrees
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(space_params["x_lower"] * 2, space_params["x_upper"] * 2, n_samples*5)
    y = rng.uniform(space_params["y_lower"] * 2, space_params["y_upper"] * 2, n_samples*5)
    
    # x = np.linspace(space_params["x_lower"] * 2, space_params["x_upper"] * 2, round(np.sqrt(n_samples)))
    # D = np.array(list(product(x, x)))
    # x = D[:,0]
    # y = D[:,1]
    
    dec_bound = decision_boundary(x, boundary_params["displ"], boundary_params["freq"], boundary_params["amp"])
    cl = np.array(["A" if c else "B" for c in y < dec_bound])

    # brush margin
    dec_up = decision_boundary(
        x,
        boundary_params["displ"] + margin_params["margin"],
        boundary_params["freq"],
        boundary_params["amp"]
    )
    dec_low = decision_boundary(
        x,
        boundary_params["displ"] - margin_params["margin"],
        boundary_params["freq"],
        boundary_params["amp"]
    )
    r = range(x.shape[0])
    out_margin_up_pos = [r[i] for i, v in enumerate(y > dec_up) if v]
    out_margin_up_ind = rng.choice(
        out_margin_up_pos, 
        np.floor(len(out_margin_up_pos) * margin_params["outmargin_sample"]).astype(int),
        replace = False
    )

    margin_pos = [r[i] for i, v in enumerate((y < dec_up) & (y > dec_low)) if v]
    margin_ind = rng.choice(
        margin_pos,
        np.floor(len(margin_pos) * margin_params["margin_sample"]).astype(int),
        replace = False
    )

    out_margin_down_pos = [r[i] for i, v in enumerate(y < dec_low) if v]
    out_margin_down_ind = rng.choice(
        out_margin_down_pos,
        np.floor(len(out_margin_down_pos) * margin_params["outmargin_sample"]).astype(int),
        replace = False
    )

    if len(out_margin_up_ind) == 0:
        out_margin_up_ind = [0]
    if len(margin_ind) == 0:
        margin_ind = [0]
    if len(out_margin_down_ind) == 0:
        out_margin_down_ind = [0]

    x = np.concat([x[out_margin_up_ind],x[margin_ind],x[out_margin_down_ind]])
    y = np.concat([y[out_margin_up_ind],y[margin_ind],y[out_margin_down_ind]])
    cl = np.concat([cl[out_margin_up_ind], cl[margin_ind], cl[out_margin_down_ind]])

    # overlap
    y[cl == "A"] += overlap
    y[cl == "B"] -= overlap
    
    # rotate
    x, y = rotate(x, y, rotation_angle)
    
    r = np.arange(x.shape[0])
    r = r[(x < space_params["x_upper"]) & (x > space_params["x_lower"]) & (y < space_params["y_upper"]) & (y > space_params["y_lower"])]
    x = x[r]
    y = y[r]
    cl = cl[r]

    return {
        "x": x,
        "y": y,
        "class": cl,
        "boundary_params": boundary_params
    }


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

def gen_xy(n_samples, space_params, rng, mode = "uniform"):
    """
    mode: can be either 'uniform', 'linear', 'normal'
    """
    if mode == "uniform":
        x = rng.uniform(space_params["x_lower"] * 2, space_params["x_upper"] * 2, n_samples*5)
        y = rng.uniform(space_params["y_lower"] * 2, space_params["y_upper"] * 2, n_samples*5)
    
    if mode == "linear":
        x = np.linspace(space_params["x_lower"] * 2, space_params["x_upper"] * 2, round(np.sqrt(n_samples)))
        D = np.array(list(product(x, x)))
        x = D[:,0]
        y = D[:,1]
    
    if mode == "normal":
        x_loc = (space_params["x_lower"] + space_params["x_upper"]) / 2
        y_loc = (space_params["y_lower"] + space_params["y_upper"]) / 2
        x = rng.normal(loc=x_loc, size=n_samples)
        y = rng.normal(loc=y_loc, size=n_samples)
    return x,y


def generate_data(n_samples, boundary_params, rotation_angle, margin_params, overlap, space_params, seed= 12345, mode= "uniform"):
    """
        n_samples: int. Number of samples to generate (might be lower since we are sampling)
        boundary_params: dict containing displ, freq, amp
        margin_params: dict containing margin, margin_sample, outmargin_sample
        space_params: dict containing x_lower, y_lower, x_upper, y_upper
        overlap: float distance to overlap two opposing classes
        rotation_angle: angle to rotate the decision boundary in degrees
    """
    rng = np.random.default_rng(seed)
    x,y = gen_xy(n_samples, space_params, rng, mode)
    
    dec_bound = decision_boundary(x, **boundary_params)
    cl = np.where(y < dec_bound, "A", "B")
    # cl = np.array(["A" if c else "B" for c in y < dec_bound])

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
    out_margin_up_ind = rng.choice(
        np.where(y > dec_up)[0], 
        int(len(y[y > dec_up]) * margin_params["outmargin_sample"]),
        replace = False
    ) if len(y[y > dec_up]) > 0 else np.array([])

    in_margin_cond = (y < dec_up) & (y > dec_low)
    margin_ind = rng.choice(
        np.where(in_margin_cond)[0],
        int(len(y[in_margin_cond]) * margin_params["margin_sample"]),
        replace = False
    ) if len(y[in_margin_cond]) > 0 else np.array([])

    out_margin_down_ind = rng.choice(
        np.where(y < dec_low)[0],
        int(len(y[y < dec_low]) * margin_params["outmargin_sample"]),
        replace = False
    ) if len(y[y < dec_low]) > 0 else np.array([])

    selected_indices = np.concat([
        out_margin_up_ind,
        margin_ind,
        out_margin_down_ind
    ])
    x, y, cl = x[selected_indices], y[selected_indices], cl[selected_indices]

    # overlap
    y[cl == "A"] += overlap / 2
    y[cl == "B"] -= overlap / 2
    
    # rotate
    x, y = rotate(x, y, rotation_angle)
    
    r = (x < space_params["x_upper"]) & (x > space_params["x_lower"]) & (y < space_params["y_upper"]) & (y > space_params["y_lower"])
    x = x[r]
    y = y[r]
    cl = cl[r]

    return {
        "x": x,
        "y": y,
        "class": cl,
        "boundary_params": boundary_params
    }


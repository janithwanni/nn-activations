import numpy as np
from itertools import product

def decision_boundary(x, displ, freq, amp):
    # print(f"{amp} * np.sin({freq} * x) + {displ}")
    return amp * np.sin(freq * x) + displ

def rotate(x,y, t):
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

def generate_xy(n_samples, space_params, rng, mode):
    """
    mode: can be either 'uniform', 'linear', 'normal'
    """
    if mode == "uniform":
        x = rng.uniform(space_params["x_lower"] * 3, space_params["x_upper"] * 3, n_samples*10)
        y = rng.uniform(space_params["y_lower"] * 3, space_params["y_upper"] * 3, n_samples*10)
    
    if mode == "linear":
        x = np.linspace(space_params["x_lower"], space_params["x_upper"], round(np.sqrt(n_samples)))
        D = np.array(list(product(x, x)))
        x = D[:,0]
        y = D[:,1]
    
    if mode == "normal":
        x_loc = (space_params["x_lower"] + space_params["x_upper"]) / 2
        y_loc = (space_params["y_lower"] + space_params["y_upper"]) / 2
        x = rng.normal(loc=x_loc, size=n_samples, scale = 3)
        y = rng.normal(loc=y_loc, size=n_samples, scale = 3)
    return x,y

def filter_margin(
    x,
    y,
    dec_bound,
    rotation_angle,
    rng,
    margin,
    margin_sample,
    outmargin_sample
):
    theta = np.radians(rotation_angle)
    x_push = -np.sin(theta) * margin
    y_push = np.cos(theta) * margin
    
    x_up = x + x_push
    x_down = x - x_push
    dec_up = dec_bound + y_push
    dec_down = dec_bound - y_push

    above_margin_cond = (y > dec_up) & (x >= x_up)
    below_margin_cond = (y < dec_down) & (x <= x_down)
    # in_margin_cond = (y <= dec_up) & (x >= x_up) & (y >= dec_down) & (x <= x_down)
    in_margin_cond = (~above_margin_cond) & (~below_margin_cond)
    
    above_margin_ind = rng.choice(
        np.where(above_margin_cond)[0],
        int(len(y[above_margin_cond]) * outmargin_sample),
        replace = False
    ) if len(y[above_margin_cond]) > 0 else np.array([])

    margin_ind = rng.choice(
        np.where(in_margin_cond)[0],
        int(len(y[in_margin_cond]) * margin_sample),
        replace = False
    ) if len(y[in_margin_cond]) > 0 else np.array([])

    below_margin_ind = rng.choice(
        np.where(below_margin_cond)[0],
        int(len(y[below_margin_cond]) * outmargin_sample),
        replace = False
    ) if len(y[below_margin_cond]) > 0 else np.array([])

    selected_indices = np.concat([
        above_margin_ind,
        margin_ind,
        below_margin_ind
    ]).astype(int)
    return selected_indices

def overlap_sides(x,y,cl,rotation_angle,overlap):
    theta = np.radians(rotation_angle)
    x_push = -np.sin(theta) * (overlap / 2)
    y_push = np.cos(theta) * (overlap / 2)
    A_mask = cl == "A"
    B_mask = cl == "B"

    x[A_mask] += x_push
    x[B_mask] -= x_push
    y[A_mask] += y_push
    y[B_mask] -= y_push
    return x,y

def generate_data(
    n_samples,
    boundary_params,
    margin_params,
    space_params,
    rotation_angle,
    overlap,
    seed=12345,
    mode="uniform"
):
    rng = np.random.default_rng(seed)
    x,y = generate_xy(n_samples, space_params, rng, mode)
    dec_bound = decision_boundary(x, **boundary_params)
    x_r, dec_bound_r = rotate(x, dec_bound, rotation_angle)

    cl = np.where(y < dec_bound, "A", "B")
    x_r, y_r = rotate(x, y, rotation_angle)
    
    # we aren't cropping, scratch that we are to make sense of it

    selected_indices = filter_margin(x_r, y_r, dec_bound_r, rotation_angle, rng, **margin_params)
    
    x_f, y_f, cl_f = x_r[selected_indices], y_r[selected_indices], cl[selected_indices]

    x_o, y_o = overlap_sides(x_f,y_f, cl_f, rotation_angle, overlap)

    crop_inds = (x_o >= space_params["x_lower"]) & (x_o <= space_params["x_upper"]) & (y_o >= space_params["y_lower"]) & (y_o <= space_params["y_upper"])
    x_r, y_r, cl_r = x_o[crop_inds], y_o[crop_inds], cl_f[crop_inds]

    return {
        "x": x_r, "y": y_r, "class": cl_r
    }
# This is effectively a direct translation of the OpenCV CUDA implementation
# of Brox optical flow, found here: 
# https://github.com/opencv/opencv_contrib/blob/4.x/modules/cudalegacy/src/cuda/NCVBroxOpticalFlow.cu

# Disclaimer most of this code was AI-generated, but it seems to produce approximately
# the same results as the OpenCV implementation (but OpenCV is really annoying
# to setup so that's why we have this instead)

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.ndimage import map_coordinates
from jax.scipy.signal import correlate2d
from functools import partial

# -----------------------------------------------------------------------------
# Constants and Utility Functions
# -----------------------------------------------------------------------------

EPS2 = 1e-6

def get_derivatives_kernel():
    """
    Corresponds to the 5-point derivative filter used in the CUDA code.
    Normalized by 1/12.0f.
    """
    return jnp.array([1.0, -8.0, 0.0, 8.0, -1.0], dtype=jnp.float32) / 12.0

@partial(jit, static_argnames=['axis'])
def compute_derivative(img, kernel, axis):
    """
    Computes derivative using correlation.
    Uses 'edge' padding (replication) which is generally safer for flow boundaries.
    """
    if axis == 1: # Row filter (derivative in x)
        k = kernel.reshape(1, -1)
        # Pad width: 2 on left, 2 on right for 5-tap filter
        img_padded = jnp.pad(img, ((0,0), (2,2)), mode='edge')
    else: # Column filter (derivative in y)
        k = kernel.reshape(-1, 1)
        img_padded = jnp.pad(img, ((2,2), (0,0)), mode='edge')
    
    return correlate2d(img_padded, k, mode='valid')

def warp_image(img, u, v):
    """
    Warps image using bilinear interpolation.
    """
    h, w = img.shape
    y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
    
    coords_x = x + u
    coords_y = y + v
    
    coords = jnp.stack([coords_y, coords_x])
    return map_coordinates(img, coords, order=1, mode='nearest') 

# -----------------------------------------------------------------------------
# Stage 1: Coefficient Preparation
# -----------------------------------------------------------------------------

@jit
def compute_diffusivity(u, v, du, dv):
    """
    Calculates the smoothness diffusivity term.
    """
    u_total = u + du
    v_total = v + dv
    
    # Pad with reflection to handle boundaries
    u_pad = jnp.pad(u_total, 1, mode='reflect')
    v_pad = jnp.pad(v_total, 1, mode='reflect')
    
    # --- Diffusivity X ---
    val_ux = u_pad[1:-1, 2:] - u_pad[1:-1, 1:-1]
    val_vx = v_pad[1:-1, 2:] - v_pad[1:-1, 1:-1]
    
    cdy_i   = u_pad[2:, 1:-1] - u_pad[:-2, 1:-1]
    cdy_ip1 = u_pad[2:, 2:]   - u_pad[:-2, 2:]
    cdy_v_i   = v_pad[2:, 1:-1] - v_pad[:-2, 1:-1]
    cdy_v_ip1 = v_pad[2:, 2:]   - v_pad[:-2, 2:]
    
    term_uy = 0.25 * (cdy_i + cdy_ip1)
    term_vy = 0.25 * (cdy_v_i + cdy_v_ip1)
    
    diff_x = 0.5 * jax.lax.rsqrt(val_ux**2 + val_vx**2 + term_uy**2 + term_vy**2 + EPS2)
    diff_x = diff_x.at[:, -1].set(0.0)
    
    # --- Diffusivity Y ---
    val_uy = u_pad[2:, 1:-1] - u_pad[1:-1, 1:-1]
    val_vy = v_pad[2:, 1:-1] - v_pad[1:-1, 1:-1]
    
    cdx_j   = u_pad[1:-1, 2:] - u_pad[1:-1, :-2]
    cdx_jp1 = u_pad[2:, 2:]   - u_pad[2:, :-2]
    cdx_v_j   = v_pad[1:-1, 2:] - v_pad[1:-1, :-2]
    cdx_v_jp1 = v_pad[2:, 2:]   - v_pad[2:, :-2]
    
    term_ux = 0.25 * (cdx_j + cdx_jp1)
    term_vx = 0.25 * (cdx_v_j + cdx_v_jp1)
    
    diff_y = 0.5 * jax.lax.rsqrt(val_uy**2 + val_vy**2 + term_ux**2 + term_vx**2 + EPS2)
    diff_y = diff_y.at[-1, :].set(0.0)
    
    return diff_x, diff_y

@jit
def prepare_sor_stage_1(u, v, du, dv, alpha, gamma, 
                        I0, Ix0, Iy0, 
                        I1, Ix1, Iy1, Ixx1, Ixy1, Iyy1):
    """
    New version: Accepts PRE-COMPUTED derivatives for I1 (Ix1, Iy1, etc.)
    and warps them, rather than warping I1 and deriving afterwards.
    """
    
    # 1. Warp everything (Image AND Derivatives)
    I1_w   = warp_image(I1, u, v)
    Ix_w   = warp_image(Ix1, u, v)
    Iy_w   = warp_image(Iy1, u, v)
    Ixx_w  = warp_image(Ixx1, u, v)
    Ixy_w  = warp_image(Ixy1, u, v)
    Iyy_w  = warp_image(Iyy1, u, v)
    
    # 2. Compute differences
    Iz = I1_w - I0
    Ixz = Ix_w - Ix0
    Iyz = Iy_w - Iy0
    
    # 3. Compute Data Term
    # Using warped spatial derivatives for the linearization
    q0 = Iz  + Ix_w * du + Iy_w * dv
    q1 = Ixz + Ixx_w * du + Ixy_w * dv
    q2 = Iyz + Ixy_w * du + Iyy_w * dv
    
    psi_data = 0.5 * jax.lax.rsqrt(q0**2 + gamma * (q1**2 + q2**2) + EPS2)
    psi_data = psi_data / alpha
    
    # 4. Compute Coefficients
    num_dudv = psi_data * (Ix_w * Iy_w + gamma * Ixy_w * (Ixx_w + Iyy_w))
    num_u    = psi_data * (Ix_w * Iz   + gamma * (Ixx_w * Ixz + Ixy_w * Iyz))
    num_v    = psi_data * (Iy_w * Iz   + gamma * (Iyy_w * Iyz + Ixy_w * Ixz))
    denom_u  = psi_data * (Ix_w * Ix_w + gamma * (Ixy_w * Ixy_w + Ixx_w * Ixx_w))
    denom_v  = psi_data * (Iy_w * Iy_w + gamma * (Ixy_w * Ixy_w + Iyy_w * Iyy_w))
    
    # 5. Compute Smoothness
    diff_x, diff_y = compute_diffusivity(u, v, du, dv)

    return diff_x, diff_y, denom_u, denom_v, num_dudv, num_u, num_v

# -----------------------------------------------------------------------------
# Stage 2 & Solver
# -----------------------------------------------------------------------------

@jit
def prepare_sor_stage_2(diff_x, diff_y, denom_u, denom_v):
    diff_x_left = jnp.pad(diff_x[:, :-1], ((0,0), (1,0)))
    diff_y_up   = jnp.pad(diff_y[:-1, :], ((1,0), (0,0)))
    
    diff_sum = diff_x + diff_x_left + diff_y + diff_y_up
    denom_u_final = denom_u + diff_sum
    denom_v_final = denom_v + diff_sum
    
    return 1.0 / denom_u_final, 1.0 / denom_v_final

@jit
def sor_pass(du, dv, u, v, diff_x, diff_y, inv_denom_u, inv_denom_v, num_u, num_v, num_dudv, omega, is_black):
    h, w = du.shape
    y_idx, x_idx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
    mask = (x_idx + y_idx) % 2 == is_black
    
    def get_neighbors_reflect(arr):
        pad = jnp.pad(arr, 1, mode='reflect')
        return pad[1:-1, :-2], pad[1:-1, 2:], pad[:-2, 1:-1], pad[2:, 1:-1]

    s_right = diff_x
    s_left  = jnp.pad(diff_x[:, :-1], ((0,0), (1,0)))
    s_down  = diff_y
    s_up    = jnp.pad(diff_y[:-1, :], ((1,0), (0,0)))
    
    u_l, u_r, u_u, u_d = get_neighbors_reflect(u)
    du_l, du_r, du_u, du_d = get_neighbors_reflect(du)
    
    sigma_u = (s_left * (u_l + du_l) + s_up * (u_u + du_u) + s_right * (u_r + du_r) + s_down * (u_d + du_d))
    center_smooth_coeff = s_left + s_right + s_up + s_down
    target_num_u = sigma_u - u * center_smooth_coeff - num_u - num_dudv * dv
    du_new = (1.0 - omega) * du + omega * inv_denom_u * target_num_u
    du_final = jnp.where(mask, du_new, du)

    # Note: Using updated du_final here
    v_l, v_r, v_u, v_d = get_neighbors_reflect(v)
    dv_l, dv_r, dv_u, dv_d = get_neighbors_reflect(dv)
    sigma_v = (s_left * (v_l + dv_l) + s_up * (v_u + dv_u) + s_right * (v_r + dv_r) + s_down * (v_d + dv_d))
    target_num_v = sigma_v - v * center_smooth_coeff - num_v - num_dudv * du_final
    dv_new = (1.0 - omega) * dv + omega * inv_denom_v * target_num_v
    dv_final = jnp.where(mask, dv_new, dv)
    
    return du_final, dv_final

# -----------------------------------------------------------------------------
# Main Algorithm
# -----------------------------------------------------------------------------

@partial(jit, static_argnames=['inner_iterations', 'outer_iterations', 'solver_iterations', 'scale_factor'])
def brox_optical_flow(img0, img1, 
                      alpha, 
                      gamma, 
                      scale_factor, 
                      inner_iterations, 
                      outer_iterations, 
                      solver_iterations):
    
    # Pyramid construction
    pyramid_i0 = [img0]
    pyramid_i1 = [img1]
    curr_i0, curr_i1 = img0, img1
    
    for _ in range(1, outer_iterations):
        h, w = curr_i0.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        if new_w < 16 or new_h < 16: break
        
        # Use linear resizing
        curr_i0 = jax.image.resize(curr_i0, (new_h, new_w), method='linear')
        curr_i1 = jax.image.resize(curr_i1, (new_h, new_w), method='linear')
        pyramid_i0.append(curr_i0)
        pyramid_i1.append(curr_i1)
    
    h_coarse, w_coarse = pyramid_i0[-1].shape
    u = jnp.zeros((h_coarse, w_coarse), dtype=jnp.float32)
    v = jnp.zeros((h_coarse, w_coarse), dtype=jnp.float32)
    kernel = get_derivatives_kernel()
    
    # Coarse-to-fine loop
    for level in range(len(pyramid_i0) - 1, -1, -1):
        I0_level = pyramid_i0[level]
        I1_level = pyramid_i1[level]
        h, w = I0_level.shape
        
        if level < len(pyramid_i0) - 1:
            u = jax.image.resize(u, (h, w), method='linear') * (1.0 / scale_factor)
            v = jax.image.resize(v, (h, w), method='linear') * (1.0 / scale_factor)
        
        # --- PRE-COMPUTE DERIVATIVES (Derive-then-Warp) ---
        # 1. Derivatives of I0
        Ix0 = compute_derivative(I0_level, kernel, axis=1)
        Iy0 = compute_derivative(I0_level, kernel, axis=0)
        
        # 2. Derivatives of I1 (Unwarped)
        Ix1  = compute_derivative(I1_level, kernel, axis=1)
        Iy1  = compute_derivative(I1_level, kernel, axis=0)
        
        # 3. Second derivatives of I1 (Unwarped)
        Ixx1 = compute_derivative(Ix1, kernel, axis=1)
        Iyy1 = compute_derivative(Iy1, kernel, axis=0)
        Ixy1 = compute_derivative(Iy1, kernel, axis=1)
        
        du = jnp.zeros_like(u)
        dv = jnp.zeros_like(v)
        
        for _ in range(inner_iterations):
            # Pass all pre-computed maps to the preparation stage
            diff_x, diff_y, denom_u, denom_v, num_dudv, num_u, num_v = \
                prepare_sor_stage_1(u, v, du, dv, alpha, gamma,
                                    I0_level, Ix0, Iy0,
                                    I1_level, Ix1, Iy1, Ixx1, Ixy1, Iyy1)
            
            inv_denom_u, inv_denom_v = prepare_sor_stage_2(diff_x, diff_y, denom_u, denom_v)
            
            def solver_body(i, val):
                c_du, c_dv = val
                omega = 1.99
                c_du, c_dv = sor_pass(c_du, c_dv, u, v, diff_x, diff_y, inv_denom_u, inv_denom_v, num_u, num_v, num_dudv, omega, 0)
                c_du, c_dv = sor_pass(c_du, c_dv, u, v, diff_x, diff_y, inv_denom_u, inv_denom_v, num_u, num_v, num_dudv, omega, 1)
                return c_du, c_dv

            du, dv = lax.fori_loop(0, solver_iterations, solver_body, (du, dv))
        
        u = u + du
        v = v + dv
        
    return u, v
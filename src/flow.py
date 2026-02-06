from typing import Literal, Union

import jax.numpy as jnp
import numpy as np
from interpax import interp2d
from jaxtyping import Array

from util import hsv_to_rgb, indices
from brox import brox_optical_flow

Direction = Union[Literal["forwards"], Literal["backwards"]]


# inspired by https://github.com/CSRavasio/oflibnumpy
# For now, the flow direction is always "backwards" for performance reasons
# offsets should be of shape (x, y, 2)
class Flow:
    offsets: Array
    direction: Direction

    def __init__(self, offsets, direction: Direction = "backwards"):
        self.offsets = offsets
        self.direction = direction

    def __neg__(self):
        return Flow(-self.offsets)
        
    def __sub__(self, other):
        if isinstance(other, jnp.ndarray):
            return Flow(self.offsets - other)
        return Flow(self.offsets - other.offsets)
        
    def __add__(self, other):
        if isinstance(other, jnp.ndarray):
            return Flow(self.offsets + other)
        return Flow(self.offsets + other.offsets)

    # warps an input single-channel image using bilinear interpolation
    # this function should work on any vector-valued input (e.g. 3D matrices
    # where the first two axes are x/y coordinates.
    def apply(self, image: Array):
        if self.direction == "backwards":
            if len(image.shape) == 2:
                image = jnp.expand_dims(image, axis=-1)

            H, W, C = image.shape

            # H*W query points, one for each pixel in warped image.
            # Offsets are given as as [dx, dy], while images are indexed [y, x], so we need to reverse the last axis of offsets.
            q = jnp.reshape(indices(H, W) + self.offsets[:, :, ::-1], shape=(H * W, 2))

            results = interp2d(
                q[:, 0],
                q[:, 1],
                jnp.arange(H),
                jnp.arange(W),
                image,
                method="linear",
                extrap=0,
            )

            return jnp.reshape(results, shape=image.shape).squeeze()
        else:
            raise NotImplementedError()

    # the flow that would result from applying the current flow and then other_flow
    def compose(self, other_flow):
        return Flow(other_flow.offsets + other_flow.apply(self.offsets))

    @staticmethod
    def zero(shape):
        return Flow(jnp.zeros((shape[0], shape[1], 2)))

    def to_rgb(self, mask=None, scale=None):
        """
        Can pass in a boolean mask of shape (H, W) in to ignore invalid areas in the image.
        """
        if mask is not None:
            flow = self.offsets * jnp.expand_dims(mask, axis=-1)
        else:
            flow = self.offsets

        angle = jnp.arctan2(flow[:, :, 1], flow[:, :, 0])
        angle = (angle + jnp.pi) / (2 * jnp.pi)
        
        magnitude = jnp.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        if scale is None:
            magnitude = magnitude / magnitude.max()
        else:
            magnitude = magnitude / scale

        hsv = jnp.stack([
            angle,
            magnitude,
            jnp.full(angle.shape, 1),
        ], axis=-1)

        return hsv_to_rgb(hsv)

    @staticmethod
    def brox(img1: Array,
             img2: Array,
             alpha=1.0,
             gamma = 150.0,
             scale_factor = 0.7,
             inner_iterations = 5,
             outer_iterations = 150,
             solver_iterations = 10,
             ):
        u_flow, v_flow = brox_optical_flow(
            img2, img1,
            alpha,
            gamma,
            scale_factor,
            inner_iterations,
            outer_iterations,
            solver_iterations,
        )
        uv_flow = jnp.stack([u_flow, v_flow], axis=-1)
        return Flow(uv_flow)

    @staticmethod
    def brox_opencv(img1: Array,
             img2: Array,
             alpha=0.197,
             gamma = 50.0,
             scale_factor = 0.8,
             inner_iterations = 5,
             outer_iterations = 150,
             solver_iterations = 10,
             ):

        import cv2

        denseflow = cv2.cuda.BroxOpticalFlow_create(
            alpha, gamma, scale_factor, inner_iterations, outer_iterations, solver_iterations,
        )

        # opencv is currently not directly compatible with JAX arrays, so we convert them to numpy arrays first
        a = cv2.cuda_GpuMat(np.expand_dims(img1, axis=-1).astype(np.float32))
        b = cv2.cuda_GpuMat(np.expand_dims(img2, axis=-1).astype(np.float32))

        flow = denseflow.calc(b, a, None).download()
        
        return Flow(jnp.array(flow))

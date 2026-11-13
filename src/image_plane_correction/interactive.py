import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .util import indices
from .flow import Flow


def plot_image(image, name="image", title="", vmin=0, vmax=1, **kwargs):
    plt.close(name)
    plt.figure(name)
    plt.title(title)
    plt.imshow(
        image, interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax, **kwargs
    )
    plt.colorbar()
    plt.show()


def toggle_plots(
    plot1,
    plot2,
    plot_bg=None,
    title_1="Plot 1",
    title_2="Plot 2",
    colorbar=True,
    name="toggle",
):
    plt.close(name)
    plt.figure(name)
    plt.title(title_1)

    if plot_bg is not None:
        plot_bg()

    p1 = plot1()
    p2 = plot2()

    if colorbar:
        plt.colorbar()

    p2.set_visible(False)

    def toggle(event):
        plt.figure(name)
        "toggle the visible state of the two images"
        if event.key != "t":
            return
        b1 = p1.get_visible()
        b2 = p2.get_visible()
        p1.set_visible(not b1)
        p2.set_visible(not b2)

        if b1:
            plt.title(title_2)
        else:
            plt.title(title_1)

        plt.draw()

    plt.connect("key_press_event", toggle)

    plt.show()


def toggle_images(
    x1,
    x2,
    name="qwerty",
    vmin=0,
    vmax=1,
    title_1="Image 1",
    title_2="Image 2",
    N=4096,
    **kwargs,
):
    extent = (-0.5, N - 0.5, 0.5, N - 0.5)
    toggle_plots(
        lambda: plt.imshow(
            x1,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            origin="lower",
            **kwargs,
        ),
        lambda: plt.imshow(
            x2,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            origin="lower",
            **kwargs,
        ),
        title_1=title_1,
        title_2=title_2,
        name=name,
    )


def plot_offsets(
    offsets, downsample=16, scale=None, title="", bg=None, N=4096, name="offsets"
):
    idxs = indices(N, N)[::downsample, ::downsample].reshape(
        (N * N // downsample // downsample, 2)
    )

    filter_idxs = jnp.linalg.norm(idxs - N // 2, axis=1) < 0.85 * N / 2

    offsets_xy = offsets[::downsample, ::downsample].reshape(
        (N * N // downsample // downsample, 2)
    )
    magnitudes = jnp.sqrt(offsets_xy[:, 0] ** 2 + offsets_xy[:, 1] ** 2)

    plt.close(name)
    plt.figure(name)
    plt.title(title)
    if bg is not None:
        plt.imshow(bg, interpolation="nearest", origin="lower", vmin=0, vmax=1)
    plt.quiver(
        idxs[filter_idxs, 0],
        idxs[filter_idxs, 1],
        offsets_xy[filter_idxs, 0],
        offsets_xy[filter_idxs, 1],
        magnitudes[filter_idxs],
        cmap="plasma",
        scale=scale,
    )
    plt.colorbar()
    plt.show()


def plot_offsets_bounded(
    offsets,
    bounds,
    downsample=16,
    scale=None,
    title="",
    bg=None,
    N=4096,
    name="offsets",
):
    (x1, x2, y1, y2) = bounds
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    size = len(range(x1, x2, downsample)) * len(range(y1, y2, downsample))

    idxs = indices(N, N)[x1:x2:downsample, y1:y2:downsample].reshape((size, 2))
    offsets_xy = offsets[x1:x2:downsample, y1:y2:downsample].reshape((size, 2))
    magnitudes = jnp.sqrt(offsets_xy[:, 0] ** 2 + offsets_xy[:, 1] ** 2)

    plt.close(name)
    plt.figure(name)
    plt.title(title)
    if bg is not None:
        plt.imshow(bg, interpolation="nearest", origin="lower", vmin=0, vmax=1)
    plt.quiver(
        idxs[:, 0],
        idxs[:, 1],
        offsets_xy[:, 0],
        offsets_xy[:, 1],
        magnitudes[:],
        cmap="plasma",
        scale=scale,
    )
    plt.colorbar()
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    plt.show()


def toggle_offsets(
    offsets_1,
    offsets_2,
    downsample=16,
    scale=None,
    title=None,
    bg=None,
    N=4096,
    title_1="Offsets 1",
    title_2="Offsets 2",
    name="toggle_offsets",
):
    idxs = indices(N, N)[::downsample, ::downsample].reshape(
        (N * N // downsample // downsample, 2)
    )

    filter_idxs = jnp.linalg.norm(idxs - N // 2, axis=1) < 0.85 * N / 2

    offsets_1_xy = offsets_1[::downsample, ::downsample].reshape(
        (N * N // downsample // downsample, 2)
    )
    magnitudes_1 = jnp.sqrt(offsets_1_xy[:, 0] ** 2 + offsets_1_xy[:, 1] ** 2)

    offsets_2_xy = offsets_2[::downsample, ::downsample].reshape(
        (N * N // downsample // downsample, 2)
    )
    magnitudes_2 = jnp.sqrt(offsets_2_xy[:, 0] ** 2 + offsets_2_xy[:, 1] ** 2)

    toggle_plots(
        lambda: plt.quiver(
            idxs[filter_idxs, 0],
            idxs[filter_idxs, 1],
            offsets_1_xy[filter_idxs, 0],
            offsets_1_xy[filter_idxs, 1],
            magnitudes_1[filter_idxs],
            cmap="plasma",
            scale=scale,
        ),
        lambda: plt.quiver(
            idxs[filter_idxs, 0],
            idxs[filter_idxs, 1],
            offsets_2_xy[filter_idxs, 0],
            offsets_2_xy[filter_idxs, 1],
            magnitudes_2[filter_idxs],
            cmap="plasma",
            scale=scale,
        ),
        plot_bg=(
            None
            if bg is None
            else (
                lambda: plt.imshow(
                    bg, interpolation="nearest", origin="lower", vmin=0, vmax=1
                )
            )
        ),
        title_1=title_1,
        title_2=title_2,
        name=name,
    )


def sequence_images(images, name="seq", titles=None, vmin=0, vmax=1, N=4096, **kwargs):
    plt.close(name)
    plt.figure(name)
    extent = (-0.5, N - 0.5, -0.5, N - 0.5)
    N = len(images)

    imgs = []

    if titles is None:
        titles = [f"Image {idx + 1}" for idx in range(N)]
    else:
        assert len(titles) == N, "Invalid number of titles provided"

    for i in range(N):
        img = plt.imshow(
            images[i],
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            origin="lower",
            **kwargs,
        )
        img.set_visible(False)
        imgs.append(img)

    idx = 0

    def show():
        plt.figure(name)
        nonlocal idx  # grab idx from outer scope
        for i in range(N):
            if i == idx:
                imgs[i].set_visible(True)
            else:
                imgs[i].set_visible(False)

        plt.title(titles[idx])
        plt.draw()

    def on_press(event):
        nonlocal idx  # grab idx from outer scope
        if event.key == "p":
            idx = (idx + 1) % N
            show()
        elif event.key == "n":
            idx = (idx - 1 + N) % N
            show()

    show()
    plt.connect("key_press_event", on_press)
    plt.show()


def plot_hist_loglog(data, start=0, stop=5, step=0.1, name="loglog"):
    plt.close(name)
    plt.figure(name)

    bins = 10 ** (np.arange(start, stop, step))
    plt.xscale("log")
    plt.hist(data, bins=bins, log=True)

    plt.show()

def plot_flow(flow, scale=None, mask=None, name="flow"):
    if isinstance(flow, jnp.ndarray):
        flow = Flow(flow)
    plt.close(name)
    plt.figure(name)
    plt.imshow(flow.to_rgb(scale=scale, mask=mask), origin="lower")
    plt.show()

def toggle_flow(flow1, flow2, scale=None, mask=None, name="flow_toggle"):
    if isinstance(flow1, jnp.ndarray):
        flow1 = Flow(flow1)
    if isinstance(flow2, jnp.ndarray):
        flow2 = Flow(flow2)
        
    toggle_images(
        flow1.to_rgb(scale=scale, mask=mask),
        flow2.to_rgb(scale=scale, mask=mask),
        name=name,
    )
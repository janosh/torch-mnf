from functools import wraps
from os.path import abspath, dirname
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.ndimage import rotate


ROOT = dirname(dirname(abspath(__file__)))


def plot_model_preds_for_rotating_img(pred_fn, img, plot_type="violin", axes=(1, 2)):
    """Rotate an image 180° in steps of 20°. For the example of an MNIST 9
    digit, this starts out on the training manifold, leaves it when the 9
    lies on its side and reenters it once we're at 180° and the 9 looks like
    a 6. In the middle, it's an invalid digit so a good Bayesian model should
    assign it increased uncertainty.
    """
    for idx in range(9):
        ax1 = plt.subplot(3, 3, idx + 1)

        img_rot = rotate(img, idx * 20, reshape=False, axes=axes)

        # insert batch dim
        preds = pred_fn(img_rot[None, ...])

        df = pd.DataFrame(preds).melt(var_name="digit", value_name="softmax")
        # scale="count": set violin width according to number of predictions in that bin
        # cut=0: limit the violin range to the range of observed data
        if plot_type == "violin":
            sns.violinplot(
                data=df, x="digit", y="softmax", scale="count", cut=0, ax=ax1
            )
        elif plot_type == "bar":
            sns.barplot(data=df, x="digit", y="softmax", ax=ax1)

        ax1.set(ylim=[None, 1.1], title=f"mean max: {preds.mean(0).argmax()}")
        ax2 = ax1.inset_axes([0, 0.5, 0.4, 0.4])
        ax2.axis("off")
        ax2.imshow(img_rot.squeeze(), cmap="gray")

    plt.tight_layout()  # keeps titles clear of above subplots


def interruptible(orig_func: Callable = None, handler: Callable = None):
    """Gracefully abort calls to the decorated function with ctrl + c."""

    def wrapper(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                handler() if handler else print(
                    f"\nDetected KeyboardInterrupt: Aborting {func.__name__}"
                )

        return wrapped_function

    if orig_func:
        return wrapper(orig_func)

    return wrapper

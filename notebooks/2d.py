# %%
# import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
from matplotlib import collections as mc

import data.sets
import nf_lib.flows as nf

# from nf_lib.spline_flows import NSF_AR, NSF_CL

# %%
target_dist = data.sets.Moons()
# target_dist = data.sets.Mixture()
# target_dist = data.sets.SIGGRAPH()
samples = target_dist.sample(256)
plt.title("target distribution")
plt.scatter(*samples.T, s=10)


# %%
# Construct the base distribution for a normalizing flow model.
base = td.MultivariateNormal(torch.zeros(2), torch.eye(2))
# base = td.TransformedDistribution(
#     td.Uniform(torch.zeros(2), torch.ones(2)), td.SigmoidTransform().inv
# )  # Logistic distribution

# Construct the flow.

# --- RealNVP
# flows = [nf.AffineHalfFlow(dim=2, parity=i % 2) for i in range(9)]

# --- NICE
# flows = [nf.AffineHalfFlow(dim=2, parity=i % 2, scale=False) for i in range(4)]
# flows.append(nf.AffineConstantFlow(dim=2, shift=False))

# --- SlowMAF (MAF, but without any parameter sharing for each dimension's scale/shift)
# flows = [nf.SlowMAF(dim=2, parity=i % 2) for i in range(4)]

# --- MAF (with MADE net, so we get very fast density estimation)
flows = [nf.MAF(dim=2, parity=i % 2) for i in range(9)]

# --- IAF (with MADE net, so we get very fast sampling)
# flows = [nf.IAF(dim=2, parity=i % 2) for i in range(3)]

# --- insert ActNorms to any of the flows above
# norms = [nf.ActNorm(dim=2) for _ in flows]
# flows = list(itertools.chain(*zip(norms, flows)))

# --- Glow paper
# flows = [nf.Invertible1x1Conv(dim=2) for i in range(3)]
# norms = [nf.ActNorm(dim=2) for _ in flows]
# couplings = [nf.AffineHalfFlow(dim=2, parity=i % 2, nh=32) for i in range(len(flows))]
# flows = list(
#     itertools.chain(*zip(norms, flows, couplings))
# )  # append a coupling layer after each 1x1

# --- Neural splines, coupling
# nfs_flow = NSF_CL if True else NSF_AR
# flows = [nfs_flow(dim=2, K=8, B=3, hidden_dim=16) for _ in range(3)]
# convs = [nf.Invertible1x1Conv(dim=2) for _ in flows]
# norms = [nf.ActNorm(dim=2) for _ in flows]
# flows = list(itertools.chain(*zip(norms, convs, flows)))

# Construct the model.
model = nf.NormalizingFlowModel(base, flows)


# %%
# todo tune WD
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
print("number of params: ", sum(p.numel() for p in model.parameters()))


def train(steps=1000, report_every=100, cb=None, samples=128):
    model.train()
    for step in range(steps):
        x = target_dist.sample(samples)

        zs, base_logprob, log_det = model(x)
        logprob = base_logprob + log_det
        loss = -torch.sum(logprob)  # NLL

        model.zero_grad()  # Reset gradients.
        loss.backward()  # Compute new gradients.
        optimizer.step()  # Update weights.

        if step % report_every == 0:
            print(f"loss at step {step}: {loss:.4g}")
            if callable(cb):
                cb()


# %%
train()


# %%
model.eval()

x = target_dist.sample(128)
zs, base_logprob, log_det = model(x)
z = zs[-1]

x = x.detach().numpy()
z = z.detach().numpy()
p = model.base.sample([128, 2])
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(*p.T, c="g", s=5)
plt.scatter(*z.T, c="r", s=5)
plt.scatter(*x.T, c="b", s=5)
plt.legend(["base", "x->z", "data"])
plt.axis("scaled")
plt.title("x -> z")

zs = model.sample(128 * 8)
z = zs[-1]
z = z.detach().numpy()
plt.subplot(122)
plt.scatter(*x.T, c="b", s=5, alpha=0.5)
plt.scatter(*z.T, c="r", s=5, alpha=0.5)
plt.legend(["data", "z->x"])
plt.title("z -> x")
plt.axis("scaled")


# %%
# plot the coordinate warp
n_grid = 20  # number of grid points
x_ticks, y_ticks = np.linspace(-3, 3, n_grid), np.linspace(-3, 3, n_grid)
xv, yv = np.meshgrid(x_ticks, y_ticks)
xy = np.stack([xv, yv], axis=-1)
# seems appropriate since we use radial distributions as base distributions
in_circle = np.sqrt((xy ** 2).sum(axis=2)) <= 3
xy = xy.reshape((n_grid * n_grid, 2))
xy = torch.from_numpy(xy.astype("float32"))

zs, log_det = model.backward(xy)

backward_flow_names = [type(f).__name__ for f in reversed(model.flow.flows)]
nz = len(zs)
for i in range(nz - 1):
    z0 = zs[i].detach().numpy()
    z1 = zs[i + 1].detach().numpy()

    # plot how the samples travel at this stage
    figs, [ax1, ax2] = plt.subplots(1, 2, figsize=(6, 3))
    # plt.figure(figsize=(20,10))
    ax1.scatter(*z0.T, c="r", s=3)
    ax1.scatter(*z1.T, c="b", s=3)
    ax1.axis([-3, 3, -3, 3])
    ax1.set_title(f"layer {i} ->{i+1} ({backward_flow_names[i]})")

    q = z1.reshape((n_grid, n_grid, 2))
    # y coords
    p1 = np.reshape(q[1:, :, :], (n_grid ** 2 - n_grid, 2))
    p2 = np.reshape(q[:-1, :, :], (n_grid ** 2 - n_grid, 2))
    lcy = mc.LineCollection(zip(p1, p2), linewidths=1, alpha=0.5, color="k")
    # x coords
    p1 = np.reshape(q[:, 1:, :], (n_grid ** 2 - n_grid, 2))
    p2 = np.reshape(q[:, :-1, :], (n_grid ** 2 - n_grid, 2))
    lcx = mc.LineCollection(zip(p1, p2), linewidths=1, alpha=0.5, color="k")
    # draw the lines
    ax2.add_collection(lcy)
    ax2.add_collection(lcx)
    ax2.axis([-3, 3, -3, 3])
    ax2.set_title(f"grid warp after layer {i+1}")

    # draw the data too
    plt.scatter(*x.T, c="r", s=5, alpha=0.5)


# %%
# Train and render. Do this with an untrained model to see changes.

n_grid = 20
x_ticks, y_ticks = np.linspace(-3, 3, n_grid), np.linspace(-3, 3, n_grid)
xv, yv = np.meshgrid(x_ticks, y_ticks)
xy = np.stack([xv, yv], axis=-1)
in_circle = np.sqrt((xy ** 2).sum(axis=2)) <= 3
xy = xy.reshape((n_grid * n_grid, 2))
xy = torch.from_numpy(xy.astype("float32"))

x_val = target_dist.sample(128 * 5)


# %%
def plot_learning():
    zs, _ = model.backward(xy)

    # plot how the samples travel at this stage
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    for idx, ax in enumerate(axes[:, :3].flat):
        zi, zip1 = [zs[i].detach().numpy() for i in [idx, idx + 1]]
        ax.scatter(*zi.T, c="r", s=1)
        ax.scatter(*zip1.T, c="b", s=1)
        ax.axis([-4, 4, -4, 4])
        ax.set(xticks=[], yticks=[])

    ax = fig.add_subplot(122)
    grid = zs[-1].detach().numpy().reshape((n_grid, n_grid, 2))
    # y coords
    p1 = np.reshape(grid[1:, :, :], (n_grid ** 2 - n_grid, 2))
    p2 = np.reshape(grid[:-1, :, :], (n_grid ** 2 - n_grid, 2))
    lcy = mc.LineCollection(zip(p1, p2), linewidths=1, alpha=0.5, color="k")
    # x coords
    p1 = np.reshape(grid[:, 1:, :], (n_grid ** 2 - n_grid, 2))
    p2 = np.reshape(grid[:, :-1, :], (n_grid ** 2 - n_grid, 2))
    lcx = mc.LineCollection(zip(p1, p2), linewidths=1, alpha=0.5, color="k")
    # draw the lines
    ax.add_collection(lcy)
    ax.add_collection(lcx)
    # draw the data too
    ax.scatter(*x_val.T, c="r", s=20, alpha=0.5)
    ax.set(xlim=[-2, 3], ylim=[-1.5, 2], xticks=[], yticks=[])

    for ax in axes[:, 3:].flat:
        ax.axis("off")


# %%
train(steps=400, cb=plot_learning)

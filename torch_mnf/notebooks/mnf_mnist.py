"""
Pit an MNF LeNet model against the regular non-Bayesian LeNet on the MNIST dataset.
"""


# %%
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torchvision as tv
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from torch_mnf import models
from torch_mnf.data import ROOT
from torch_mnf.evaluate import rot_img

# %%
batch_size = 32
plt.rcParams["figure.figsize"] = [12, 8]

torch.manual_seed(0)  # ensure reproducible results


# %%
# torchvision.transforms.Normalize() seems to be unnecessary
train_set, test_set = [
    MNIST(ROOT + "/data", transform=ToTensor(), download=True, train=x)
    for x in [True, False]
]

train_loader, test_loader = [
    DataLoader(x, batch_size=batch_size, shuffle=True, drop_last=True)
    for x in [train_set, test_set]
]


# %%
def train_step(model, optim, loss_fn, images, labels):
    # We could draw multiple posterior samples here to get unbiased Monte Carlo
    # estimate for the NLL which would decrease training variance but slow us down.
    optim.zero_grad()
    preds = model(images)
    loss = loss_fn(preds, labels)
    loss.backward()
    optim.step()
    return loss, preds


def train_fn(net, optim, loss_fn, loader, epochs=1, log_every=None):

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for samples, labels in pbar:

            loss, preds = train_step(net, optim, loss_fn, samples, labels)

            if log_every and net.step % log_every == 0:

                # Accuracy estimated by single call for speed. Would be more accurate to
                # approximately integrate over parameter posteriors by averaging across
                # multiple calls.
                val_preds = net(X_val)
                val_acc = (y_val == val_preds.argmax(1)).float().mean()
                train_acc = (labels == preds.argmax(1)).float().mean()
                pbar.set_postfix(loss=f"{loss:.4g}", val_acc=f"{val_acc:.4g}")

                writer.add_scalar("accuracy/training", train_acc, net.step)
                writer.add_scalar("accuracy/validation", val_acc, net.step)

            net.step += 1


# %%
# create validation set
X_val = test_set.data[:100].unsqueeze(1).float() / 255
y_val = test_set.targets[:100]
img9 = test_set[12][0]


# %%
mnf_lenet = models.MNFLeNet()
mnf_lenet.step = 0

mnf_adam = torch.optim.Adam(mnf_lenet.parameters())
print(f"MNFLeNet param count: {sum(p.numel() for p in mnf_lenet.parameters()):,}")

writer = SummaryWriter(ROOT + f"/runs/mnf_lenet/{datetime.now():%m.%d-%H:%M:%S}")


# %%
def mnf_loss_fn(preds, labels):
    nll = F.nll_loss(preds, labels).mean()

    # The KL divergence acts as a regularizer to prevent overfitting.
    kl_div = mnf_lenet.kl_div() / len(train_loader)
    loss = nll + kl_div

    writer.add_scalar("loss/NLL", nll, mnf_lenet.step)
    writer.add_scalar("loss/KL div", kl_div, mnf_lenet.step)
    writer.add_scalar("loss/NLL + KL", loss, mnf_lenet.step)

    return loss


# %%
train_fn(mnf_lenet, mnf_adam, mnf_loss_fn, train_loader, log_every=20)


# %%
# repeat the same image along axis 0 to run multiple forward passes in parallel
mnf_pred = lambda x: mnf_lenet(torch.tensor(x.repeat(500, 0))).exp().detach().numpy()
rot_img(mnf_pred, img9)


# %%
grid = tv.utils.make_grid(X_val)
writer.add_image("images", grid, 0)
writer.add_graph(mnf_lenet, X_val)  # add model graph to TensorBoard summary
writer.close()


# %%
lenet = models.LeNet()
lenet_adam = torch.optim.Adam(lenet.parameters(), lr=1e-3)
lenet.step = 0
print(f"LeNet param count: {sum(p.numel() for p in lenet.parameters()):,}")


# %%
lenet_loss_fn = lambda preds, labels: F.nll_loss(preds, labels).mean()

train_fn(lenet, lenet_adam, lenet_loss_fn, train_loader)


# %%
rot_img(lambda x: lenet(torch.tensor(x)).exp(), img9, plot_type="bar")

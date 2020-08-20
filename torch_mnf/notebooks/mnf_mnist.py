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
    MNIST(root=ROOT + "/data", transform=ToTensor(), download=True, train=x)
    for x in [True, False]
]

train_loader, test_loader = [
    DataLoader(dataset=x, batch_size=batch_size, shuffle=True, drop_last=True)
    for x in [train_set, test_set]
]


# %%
mnf_lenet = models.MNFLeNet(
    use_z=True,  # Whether to use auxiliary random variable z ~ q(z) to increase
    # expressivity of weight posteriors q(W|z).
    n_flows_q=2,
    n_flows_r=2,
    learn_p=False,
    max_std=1,  # Maximum stddev for layer weights. Larger values clipped at call time.
    flow_h_sizes=[50],  # Size and count of layers to use in the auxiliary rv flow.
    std_init=1,  # Scaling factor for stddev of unit Gaussian initialized params.
)
mnf_lenet.step = 0

mnf_adam = torch.optim.Adam(mnf_lenet.parameters(), lr=1e-3)
print(f"MNFLeNet param count: {sum(p.numel() for p in mnf_lenet.parameters()):,}")

writer = SummaryWriter(ROOT + f"/runs/mnf_lenet/{datetime.now():%m.%d-%H:%M:%S}")


# %%
def loss_fn(labels, preds):
    nll = F.nll_loss(preds, labels).mean()

    # The KL divergence acts as a regularizer to prevent overfitting.
    batch_size = labels.size(0)
    kl_div = mnf_lenet.kl_div() / batch_size
    loss = nll + kl_div

    writer.add_scalar("NLL", nll, mnf_lenet.step)
    writer.add_scalar("KL regularization", kl_div, mnf_lenet.step)
    writer.add_scalar("VI lower bound loss (NLL + KL)", loss, mnf_lenet.step)

    return nll


# %%
def train_step(images, labels):
    # We could draw multiple posterior samples here to get unbiased Monte Carlo
    # estimate for the NLL which would decrease training variance but slow us down.
    preds = mnf_lenet(images)
    loss = loss_fn(labels, preds)
    mnf_adam.zero_grad()
    loss.backward()
    mnf_adam.step()
    return loss


# %%
# create validation set
X_val = test_set.data[:100].unsqueeze(1).float() / 255
y_val = test_set.targets[:100]


# %%
grid = tv.utils.make_grid(X_val)
writer.add_image("images", grid, 0)
writer.add_graph(mnf_lenet, X_val)  # add model graph to TensorBoard summary
writer.close()


# %%
epochs = 2
log_every = 50

for epoch in range(epochs):
    pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
    for samples, labels in pbar:

        loss = train_step(samples, labels)

        if mnf_lenet.step % log_every == 0:

            # Accuracy estimated by single call for speed. Would be more accurate to
            # approximately integrate over the parameter posteriors by averaging across
            # multiple calls.
            val_preds = mnf_lenet(X_val)
            val_acc = (y_val == val_preds.argmax(1)).float().mean()
            pbar.set_postfix(loss=f"{loss:.4g}", val_acc=f"{val_acc:.4g}")

            writer.add_scalar("validation accuracy", val_acc, mnf_lenet.step)

        mnf_lenet.step += 1


# %%
img9 = test_set[12][0]
rot_img(  # via repeat run multiple forward passes for the same tensor in parallel
    lambda x: mnf_lenet(torch.tensor(x).repeat(500, 1, 1, 1)).exp().detach().numpy(),
    img9,
)

# %%
lenet = models.LeNet()
lenet_adam = torch.optim.Adam(lenet.parameters(), lr=1e-3)


# %%
rot_img(lambda x: lenet(torch.tensor(x)).exp(), img9, plot_type="bar")

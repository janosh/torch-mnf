import torch
from torch.nn.functional import nll_loss
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torch_mnf.data import ROOT
from torch_mnf.models import MNFLeNet

torch.manual_seed(0)  # ensure reproducible results


train_set, test_set = [
    MNIST(ROOT + "/data", transform=ToTensor(), download=True, train=x)
    for x in [True, False]
]

train_loader, test_loader = [
    DataLoader(x, batch_size=32, shuffle=True, drop_last=True)
    for x in [train_set, test_set]
]

X_val = test_set.data[:500].unsqueeze(1).float() / 255
y_val = test_set.targets[:500]


def mnf_loss_fn(preds, labels):
    nll = nll_loss(preds, labels).mean()
    kl_div = mnf_lenet.kl_div() * 1e-3
    loss = nll + kl_div
    return loss


def trainer(model, optim, loss_fn, data_loader):
    for samples, labels in data_loader:
        optim.zero_grad()
        preds = model(samples)
        loss = loss_fn(preds, labels)
        loss.backward()
        optim.step()
        acc = (labels == preds.argmax(1)).float().mean()
        if acc > 0.95:
            break  # save CI quota


mnf_lenet = MNFLeNet()
adam = torch.optim.Adam(mnf_lenet.parameters())


def test_acc():
    trainer(mnf_lenet, adam, mnf_loss_fn, train_loader)

    val_preds = mnf_lenet(X_val)
    val_acc = (y_val == val_preds.argmax(1)).float().mean()
    print(f"val_acc: {val_acc:.4g}")
    assert val_acc > 0.8  # not trying to win a contest, just make sure it trains

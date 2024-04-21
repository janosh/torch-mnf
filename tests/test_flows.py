from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal

import torch_mnf.flows as nf
from torch_mnf import data


torch.manual_seed(0)  # ensure reproducible results


def train(
    flow_model: nf.NormalizingFlowModel,
    optim: torch.optim.Optimizer,
    samples: Tensor,
    steps: int = 70,
) -> float:
    for _ in range(steps):
        _, log_det = flow_model.inverse(samples)
        base_log_prob = flow_model.base_log_prob(samples)
        log_prob = log_det + base_log_prob
        loss = -log_prob.sum()  # NLL

        flow_model.zero_grad()  # reset gradients
        loss.backward()  # compute new gradients
        optim.step()  # update weights

    return loss  # return final loss for e2e tests to discover regressions


samples = data.sample_moons(128)


# We use the same base distribution for all flow models.
base = MultivariateNormal(torch.zeros(2), torch.eye(2))


def e2e_test_flow_model(flow: Sequence[nn.Module], loss_bound: float) -> None:
    """End-to-end test ensuring some flow works with NormalizingFlowModel
    and training performance did not regress.
    """
    model = nf.NormalizingFlowModel(base, flow)
    adam = torch.optim.Adam(model.parameters())
    loss1 = train(model, adam, samples, steps=1)
    loss2 = train(model, adam, samples)
    assert loss1 > loss2
    assert loss2 < loss_bound, f"{loss2=:.4} > {loss_bound=}"


def test_rnvp() -> None:
    flow = [nf.AffineHalfFlow(dim=2, parity=i % 2 == 0) for i in range(2)]
    e2e_test_flow_model(flow, 236)


def test_maf() -> None:
    flow = [nf.MAF(dim=2, parity=i % 2 == 0) for i in range(2)]
    e2e_test_flow_model(flow, 250)


def test_maf_with_actnorm() -> None:
    flow = [nf.MAF(dim=2, parity=i % 2 == 0) for i in range(2)]
    # prepend each MAF with ActNormFlow
    for idx in reversed(range(len(flow))):
        flow.insert(idx, nf.ActNormFlow(dim=2))
    e2e_test_flow_model(flow, 226)


def test_iaf() -> None:
    flow = [nf.IAF(dim=2, parity=i % 2 == 0) for i in range(2)]
    e2e_test_flow_model(flow, 300)


def test_glow() -> None:
    flow = [nf.Glow(dim=2) for _ in range(2)]
    e2e_test_flow_model(flow, 308)


def test_glow_with_actnorm() -> None:
    flow = [nf.Glow(dim=2) for _ in range(2)]
    # prepend each Glow (1x1 convolution) with ActNormFlow
    for idx in reversed(range(len(flow))):
        flow.insert(idx, nf.ActNormFlow(dim=2))
    e2e_test_flow_model(flow, 246)


def test_nsfcl() -> None:
    flow = [nf.NSF_CL(dim=2, K=8, B=3, n_h=16) for _ in range(2)]
    e2e_test_flow_model(flow, 207)


def test_nsfcl_with_actnorm() -> None:
    flow = [nf.NSF_CL(dim=2, K=8, B=3, n_h=16) for _ in range(2)]
    # prepend each NSF flow with ActNormFlow and Glow
    for idx in reversed(range(len(flow))):
        flow.insert(idx, nf.ActNormFlow(dim=2))
    e2e_test_flow_model(flow, 184)


def test_nsfar() -> None:
    flow = [nf.NSF_AR(dim=2, K=8, B=3, n_h=16) for _ in range(2)]
    e2e_test_flow_model(flow, 318)


def test_nsfar_with_actnorm() -> None:
    flow = [nf.NSF_AR(dim=2, K=8, B=3, n_h=16) for _ in range(2)]
    # prepend each NSF flow with ActNormFlow and Glow
    for idx in reversed(range(len(flow))):
        flow.insert(idx, nf.ActNormFlow(dim=2))
    e2e_test_flow_model(flow, 213)

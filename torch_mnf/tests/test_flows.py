import torch
from torch.distributions import MultivariateNormal

import torch_mnf.flows as nf
from torch_mnf import data

torch.manual_seed(0)  # ensure reproducible results


def train(model, optim, samples, steps=50):
    for _ in range(steps):
        _, log_det = model.inverse(samples)
        base_log_prob = model.base_log_prob(samples)
        log_prob = log_det + base_log_prob
        loss = -torch.sum(log_prob)  # NLL

        model.zero_grad()  # reset gradients
        loss.backward()  # compute new gradients
        optim.step()  # update weights

    return loss  # return final loss for e2e tests to discover regresions


samples = data.sample_moons(128)


# Construct the base distribution for a normalizing flow model.
base = MultivariateNormal(torch.zeros(2), torch.eye(2))


def e2e_test_flow_model(flow, loss_bound):
    """End-to-end test ensuring some flow works with NormalizingFlowModel
    and trains correctly.
    """
    model = nf.NormalizingFlowModel(base, flow)
    adam = torch.optim.Adam(model.parameters())
    loss1 = train(model, adam, samples, steps=1)
    loss2 = train(model, adam, samples)
    assert loss1 > loss2
    assert loss2 < loss_bound


def test_rnvp():
    flow = [nf.AffineHalfFlow(dim=2, parity=i % 2) for i in range(2)]
    e2e_test_flow_model(flow, 236)


def test_maf():
    flow = [nf.MAF(dim=2, parity=i % 2) for i in range(2)]
    e2e_test_flow_model(flow, 247)


def test_maf_with_actnorm():
    flow = [nf.MAF(dim=2, parity=i % 2) for i in range(2)]
    # prepend each MAF with ActNormFlow
    for idx in reversed(range(len(flow))):
        flow.insert(idx, nf.ActNormFlow(dim=2))
    e2e_test_flow_model(flow, 226)


def test_iaf():
    flow = [nf.IAF(dim=2, parity=i % 2) for i in range(2)]
    e2e_test_flow_model(flow, 300)


def test_glow():
    flow = [nf.Glow(dim=2) for _ in range(2)]
    e2e_test_flow_model(flow, 308)


def test_glow_with_actnorm():
    flow = [nf.Glow(dim=2) for _ in range(2)]
    # prepend each Glow (1x1 convolution) with ActNormFlow
    for idx in reversed(range(len(flow))):
        flow.insert(idx, nf.ActNormFlow(dim=2))
    e2e_test_flow_model(flow, 243)


def test_nsfcl():
    flow = [nf.NSF_CL(dim=2, K=8, B=3, n_h=16) for _ in range(2)]
    e2e_test_flow_model(flow, 207)


def test_nsfcl_with_actnorm():
    flow = [nf.NSF_CL(dim=2, K=8, B=3, n_h=16) for _ in range(2)]
    # prepend each NSF flow with ActNormFlow and Glow
    for idx in reversed(range(len(flow))):
        flow.insert(idx, nf.ActNormFlow(dim=2))
    e2e_test_flow_model(flow, 184)


def test_nsfar():
    flow = [nf.NSF_AR(dim=2, K=8, B=3, n_h=16) for _ in range(2)]
    e2e_test_flow_model(flow, 318)


def test_nsfar_with_actnorm():
    flow = [nf.NSF_AR(dim=2, K=8, B=3, n_h=16) for _ in range(2)]
    # prepend each NSF flow with ActNormFlow and Glow
    for idx in reversed(range(len(flow))):
        flow.insert(idx, nf.ActNormFlow(dim=2))
    e2e_test_flow_model(flow, 213)

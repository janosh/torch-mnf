# Torch MNF &nbsp; [![Test Status](https://github.com/janosh/torch-mnf/workflows/tests/badge.svg)](https://github.com/janosh/torch-mnf/actions)

PyTorch implementation of Multiplicative Normalizing Flows [[1]](#mnf-bnn).

With flow implementations courtesy of [Andrej Karpathy](https://github.com/karpathy/pytorch-normalizing-flows).

## Files of Interest

- [`notebooks/mnf_mnist.py`](torch_mnf/notebooks/mnf_mnist.py)
- [`models/mnf_lenet.py`](torch_mnf/models/mnf_lenet.py)
- [`flows/*.py`](torch_mnf/flows)
- [`layers/*.py`](torch_mnf/layers)

## References

1. <a id="mnf-bnn"></a> **MNF**: _Multiplicative Normalizing Flows for Variational Bayesian Neural Networks_ | Christos Louizos, Max Welling (Mar 2017) | [1703.01961](https://arxiv.org/abs/1703.01961)

2. <a id="vi-nf"></a> **VI-NF**: _Variational Inference with Normalizing Flows_ | Danilo Rezende, Shakir Mohamed (May 2015) | [1505.05770](https://arxiv.org/abs/1505.05770)

3. <a id="made"></a> **MADE**: _Masked Autoencoder for Distribution Estimation_ | Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle (Jun 2015) | [1502.03509](https://arxiv.org/abs/1502.03509)

4. <a id="nice"></a> **NICE**: _Non-linear Independent Components Estimation_ | Laurent Dinh, David Krueger, Yoshua Bengio (Oct 2014) | [1410.8516](https://arxiv.org/abs/1410.8516)

5. <a id="rnvp"></a> **RNVP**: _Density estimation using Real NVP_ | Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio (May 2016) | [1605.08803](https://arxiv.org/abs/1605.08803)

6. <a id="maf"></a> **MAF**: _Masked Autoregressive Flow for Density Estimation_ | George Papamakarios, Theo Pavlakou, Iain Murray (Jun 2018) | [1705.07057](https://arxiv.org/abs/1705.07057)

7. <a id="iaf"></a> **IAF**: _Improving Variational Inference with Inverse Autoregressive Flow_ | Diederik Kingma et al. (Jun 2016) | [1606.04934](https://arxiv.org/abs/1606.04934)

8. <a id="nsf"></a> **NSF**: _Neural Spline Flows_ | Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios (Jun 2019) | [1906.04032](https://arxiv.org/abs/1906.04032)

## Debugging Tips

A great method of checking for infinite or `NaN` gradients is

```py
for name, param in model.named_parameters():
    print(name, torch.isfinite(param.grad).all())
    print(name, torch.isnan(param.grad).any())
```

There's also [`torch.autograd.detect_anomaly()`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.detect_anomaly) used as context manager:

```py
with torch.autograd.detect_anomaly():
    x = torch.rand(10, 10, requires_grad=True)
    out = model(x)
    out.backward()
```

and [`torch.autograd.set_detect_anomaly(True)`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.set_detect_anomaly). See [here](https://discuss.pytorch.org/t/87594) for an issue that used these tools.

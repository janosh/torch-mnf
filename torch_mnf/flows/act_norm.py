from torch_mnf.flows import AffineConstantFlow


class ActNorm(AffineConstantFlow):
    """Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we cleverly initialize the scale and
    translate function (s, t) so that the output is unit Gaussian. As
    described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    def forward(self, x):
        # first batch is used for init
        if not self.data_dep_init_done:
            if self.scale:
                self.s.data = -x.std(dim=0, keepdim=True).log().detach()
            if self.shift:
                self.t.data = -(x * self.s.exp()).mean(dim=0, keepdim=True).detach()
            self.data_dep_init_done = True
        return super().forward(x)

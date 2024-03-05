# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This module implements the Causal Effect Variational Autoencoder [1], which
demonstrates a number of innovations including:

- a generative model for causal effect inference with hidden confounders;
- a model and guide with twin neural nets to allow imbalanced treatment; and
- a custom training loss that includes both ELBO terms and extra terms needed
  to train the guide to be able to answer counterfactual queries.

The main interface is the :class:`CEVAE` class, but users may customize by
using components :class:`Model`, :class:`Guide`,
:class:`TraceCausalEffect_ELBO` and utilities.

**References**

[1] C. Louizos, U. Shalit, J. Mooij, D. Sontag, R. Zemel, M. Welling (2017).
    | Causal Effect Inference with Deep Latent-Variable Models.
    | http://papers.nips.cc/paper/7223-causal-effect-inference-with-deep-latent-variable-models.pdf
    | https://github.com/AMLab-Amsterdam/CEVAE
"""
import logging
import utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI
from causal.trace_elbo_local import Trace_ELBO
from pyro.infer.util import torch_item
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam
from pyro.util import torch_isnan
from tqdm import tqdm
from models import CEVAEEmbedding, CEVAETransformer
import pandas as pd
from IPython.display import display
import wandb
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """

    def __init__(self, sizes, final_activation=None):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)


class DistributionNet(nn.Module):
    """
    Base class for distribution nets.
    """

    @staticmethod
    def get_class(dtype):
        """
        Get a subclass by a prefix of its name, e.g.::

            assert DistributionNet.get_class("bernoulli") is BernoulliNet
        """
        for cls in DistributionNet.__subclasses__():
            if cls.__name__.lower() == dtype + "net":
                return cls
        raise ValueError("dtype not supported: {}".format(dtype))


# class MultinormNet(DistributionNet):

#     def __init__(self, sizes, classes=7):
#         assert len(sizes) >= 1
#         super().__init__()
#         self.fc = FullyConnected(sizes + [classes])

#     def forward(self, x):
#         logits = self.fc(x).clamp(min=-10, max=10)
#         return (logits,)

#     @staticmethod
#     def make_dist(logits):
#         return dist.Categorical(logits=logits)
    

class MultinormNet(DistributionNet):

    def __init__(self, sizes, classes=7, final_activation=None):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [classes], final_activation=final_activation)

    def forward(self, x):
        logits = self.fc(x).clamp(min=-10, max=10)
        return (logits,)

    @staticmethod
    def make_dist(logits):
        return dist.Categorical(logits=logits)



class BernoulliNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = BernoulliNet([3, 4])
        z = torch.randn(3)
        logits, = net(z)
        t = net.make_dist(logits).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=-10, max=10)
        return (logits,)

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)


class ExponentialNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``rate``.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = ExponentialNet([3, 4])
        x = torch.randn(3)
        rate, = net(x)
        y = net.make_dist(rate).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        scale = nn.functional.softplus(self.fc(x).squeeze(-1)).clamp(min=1e-3, max=1e6)
        rate = scale.reciprocal()
        return (rate,)

    @staticmethod
    def make_dist(rate):
        return dist.Exponential(rate)


class LaplaceNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Laplace random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = LaplaceNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Laplace(loc, scale)


class NormalNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = NormalNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    # TODO : 이거 코드 이해
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])
        # self.fc = FullyConnected(sizes + [14])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Normal(loc, scale)


class StudentTNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``df,loc,scale``
    triple, with shared ``df > 1``.

    This is used to represent a conditional probability distribution of a
    single Student's t random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = StudentTNet([3, 4])
        x = torch.randn(3)
        df, loc, scale = net(x)
        y = net.make_dist(df, loc, scale).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])
        self.df_unconstrained = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        df = nn.functional.softplus(self.df_unconstrained).add(1).expand_as(loc)
        return df, loc, scale

    @staticmethod
    def make_dist(df, loc, scale):
        return dist.StudentT(df, loc, scale)


class DiagNormalNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    ``sizes[-1]``-sized diagonal Normal random variable conditioned on a
    ``sizes[0]``-size real value, for example::

        net = DiagNormalNet([3, 4, 5])
        z = torch.randn(3)
        loc, scale = net(z)
        x = dist.Normal(loc, scale).sample()

    This is intended for the latent ``z`` distribution and the prewhitened
    ``x`` features, and conservatively clips ``loc`` and ``scale`` values.
    """

    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., : self.dim].clamp(min=-1e2, max=1e2)
        scale = (
            nn.functional.softplus(loc_scale[..., self.dim :]).add(1e-3).clamp(max=1e2)
        )
        return loc, scale


class PreWhitener(nn.Module):
    """
    Data pre-whitener.
    """

    def __init__(self, data):
        super().__init__()
        with torch.no_grad():
            loc = data.mean(0)
            scale = data.std(0)
            scale[~(scale > 0)] = 1.0
            self.register_buffer("loc", loc)
            self.register_buffer("inv_scale", scale.reciprocal())

    def forward(self, data):
        return (data - self.loc) * self.inv_scale


class Model(PyroModule):
    """
    Generative model for a causal model with latent confounder ``z`` and binary
    treatment ``t``::

        z ~ p(z)      # latent confounder
        x ~ p(x|z)    # partial noisy observation of z
        t ~ p(t|z)    # treatment, whose application is biased by z
        y ~ p(y|t,z)  # outcome

    Each of these distributions is defined by a neural network.  The ``y``
    distribution is defined by a disjoint pair of neural networks defining
    ``p(y|t=0,z)`` and ``p(y|t=1,z)``; this allows highly imbalanced treatment.

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """

    def __init__(self, config):
        self.latent_dim = config["latent_dim"]
        super().__init__()
        self.x_nn = DiagNormalNet(
            [config["latent_dim"]]
            + [config["hidden_dim"]] * config["num_layers"]
            + [config["feature_dim"]]
        )
        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        # The y network is split between the two t values.
        self.y0_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.y1_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.y2_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.y3_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.y4_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.y5_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.y6_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )

        # The d network is split between the two t values.
        self.d0_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.d1_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.d2_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.d3_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.d4_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.d5_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.d6_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.t_nn = MultinormNet([config["latent_dim"]] + [config["hidden_dim"]] * (config["num_layers"] - 1),
            final_activation=nn.ELU())
        # self.t_nn = BernoulliNet([config["latent_dim"]])

    def forward(self, x=None, t=None, y=None, d=None, size=None, mode=None, z=None):
        if mode != "mae":
            if size is None:
                size = x.size(0)
            with pyro.plate("data", size, subsample=x):
                z = pyro.sample("z", self.z_dist())
                x = pyro.sample("x", self.x_dist(z), obs=x)
                t = pyro.sample("t", self.t_dist(z), obs=t)
                y = pyro.sample("y", self.y_dist(t, z), obs=y)
                d = pyro.sample("d", self.d_dist(t, z), obs=d)
        else:
            y = self.y_dist(t, z).mean
            d = self.d_dist(t, z).mean
            return y,d # svi.step에서는 안쓴다..!

    def y_mean(self, x, t=None):
        with pyro.plate("data", x.size(0)):
            z = pyro.sample("z", self.z_dist())
            x = pyro.sample("x", self.x_dist(z), obs=x)
            t = pyro.sample("t", self.t_dist(z), obs=t)
        return self.y_dist(t, z).mean
    
    def d_mean(self, x, t=None):
        with pyro.plate("data", x.size(0)):
            z = pyro.sample("z", self.z_dist())
            x = pyro.sample("x", self.x_dist(z), obs=x)
            t = pyro.sample("t", self.t_dist(z), obs=t)
        return self.d_dist(t, z).mean

    def z_dist(self):
        # return dist.Normal(0, 1).expand([self.latent_dim]).to_event(1)
        return dist.Normal(0, 1).expand([self.latent_dim]).to_event(1)

    def x_dist(self, z):
        loc, scale = self.x_nn(z)
        return dist.Normal(loc, scale).to_event(1)

    # def y_dist(self, t, z):
    #     # Parameters are not shared among t values.
    #     params0 = self.y0_nn(z)
    #     params1 = self.y1_nn(z)
    #     t = t.bool()
    #     params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
    #     return self.y0_nn.make_dist(*params)
    
    def y_dist(self, t, z):
        # In the final layer params are not shared among t values.
        all_params = [
            self.y0_nn(z),
            self.y1_nn(z),
            self.y2_nn(z),
            self.y3_nn(z),
            self.y4_nn(z),
            self.y5_nn(z),
            self.y6_nn(z)
        ]
        t = t.int()
        selected_locs = []
        selected_scales = []
        for batch_idx, ti in enumerate(t):
            param_for_t = all_params[ti.item()]
            selected_loc = param_for_t[0][batch_idx]
            selected_scale = param_for_t[1][batch_idx]
            selected_locs.append(selected_loc)
            selected_scales.append(selected_scale)
        
        locs_tensor = torch.stack(selected_locs)
        scales_tensor = torch.stack(selected_scales)
        return self.y0_nn.make_dist(locs_tensor, scales_tensor)

    # def d_dist(self, t, z):
    #     # Parameters are not shared among t values.
    #     params0 = self.d0_nn(z)
    #     params1 = self.d1_nn(z)
    #     t = t.bool()
    #     params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
    #     return self.d0_nn.make_dist(*params)
    def d_dist(self, t, z):
        all_params = [
            self.d0_nn(z),
            self.d1_nn(z),
            self.d2_nn(z),
            self.d3_nn(z),
            self.d4_nn(z),
            self.d5_nn(z),
            self.d6_nn(z)
        ]
        t = t.int()
        selected_locs = []
        selected_scales = []
        for batch_idx, ti in enumerate(t):
            param_for_t = all_params[ti.item()]
            selected_loc = param_for_t[0][batch_idx]
            selected_scale = param_for_t[1][batch_idx]
            selected_locs.append(selected_loc)
            selected_scales.append(selected_scale)
        
        locs_tensor = torch.stack(selected_locs)
        scales_tensor = torch.stack(selected_scales)
        return self.d0_nn.make_dist(locs_tensor, scales_tensor)
    
    def t_dist(self, z):
        (logits,) = self.t_nn(z)
        
        # return dist.Bernoulli(logits=logits)
        return dist.Categorical(logits=logits)


class Guide(PyroModule):
    """
    Inference model for causal effect estimation with latent confounder ``z``
    and binary treatment ``t``::

        t ~ q(t|x)      # treatment
        y ~ q(y|t,x)    # outcome
        z ~ q(z|y,t,x)  # latent confounder, an embedding

    Each of these distributions is defined by a neural network.  The ``y`` and
    ``z`` distributions are defined by disjoint pairs of neural networks
    defining ``p(-|t=0,...)`` and ``p(-|t=1,...)``; this allows highly
    imbalanced treatment.

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """

    def __init__(self, config):
        self.latent_dim = config["latent_dim"]
        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])

        super().__init__()
        # self.t_nn = BernoulliNet([config["feature_dim"]])
        self.t_nn = MultinormNet([config["feature_dim"]] + [config["latent_dim"]] * (config["num_layers"] - 1),
            final_activation=nn.ELU())

        # The y and z networks both follow an architecture where the first few
        # layers are shared for t in {0,1}, but the final layer is split
        # between the two t values.
        self.y_nn = FullyConnected(
            [config["feature_dim"]]
            + [config["hidden_dim"]] * (config["num_layers"] - 1),
            final_activation=nn.ELU(),
        )
        self.y0_nn = OutcomeNet([config["hidden_dim"]])
        self.y1_nn = OutcomeNet([config["hidden_dim"]])
        self.y2_nn = OutcomeNet([config["hidden_dim"]])
        self.y3_nn = OutcomeNet([config["hidden_dim"]])
        self.y4_nn = OutcomeNet([config["hidden_dim"]])
        self.y5_nn = OutcomeNet([config["hidden_dim"]])
        self.y6_nn = OutcomeNet([config["hidden_dim"]])

        self.d_nn = FullyConnected(
            [config["feature_dim"]]
            + [config["hidden_dim"]] * (config["num_layers"] - 1),
            final_activation=nn.ELU(),
        )
        self.d0_nn = OutcomeNet([config["hidden_dim"]])
        self.d1_nn = OutcomeNet([config["hidden_dim"]])
        self.d2_nn = OutcomeNet([config["hidden_dim"]])
        self.d3_nn = OutcomeNet([config["hidden_dim"]])
        self.d4_nn = OutcomeNet([config["hidden_dim"]])
        self.d5_nn = OutcomeNet([config["hidden_dim"]])
        self.d6_nn = OutcomeNet([config["hidden_dim"]])

        self.z_nn = FullyConnected(
            [2 + config["feature_dim"]]
            + [config["hidden_dim"]] * (config["num_layers"] - 1),
            final_activation=nn.ELU(),
        )

        self.z0_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim"]])
        self.z1_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim"]])
        self.z2_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim"]])
        self.z3_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim"]])
        self.z4_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim"]])
        self.z5_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim"]])
        self.z6_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim"]])

    def forward(self, x, t=None, y=None, d=None, size=None, mode=None):
        if mode == "enc_out":
            y = self.y_dist(t, x).mean
            d = self.d_dist(t, x).mean
            return y, d
        else:
            if size is None:
                size = x.size(0)
            with pyro.plate("data", size, subsample=x):
                # The t and y sites are needed for prediction, and participate in
                # the auxiliary CEVAE loss. We mark them auxiliary to indicate they
                # do not correspond to latent variables during training.
                t = pyro.sample("t", self.t_dist(x), obs=t, infer={"is_auxiliary": True})
                y = pyro.sample("y", self.y_dist(t, x), obs=y, infer={"is_auxiliary": True})
                d = pyro.sample("d", self.d_dist(t, x), obs=d, infer={"is_auxiliary": True})
                # The z site participates only in the usual ELBO loss.
                z = pyro.sample("z", self.z_dist(y, d, t, x))
                return z

    def t_dist(self, x):
        (logits,) = self.t_nn(x)
        return dist.Categorical(logits=logits)
        # return dist.Bernoulli(logits=logits)

    def y_dist(self, t, x):
        # The first n-1 layers are identical for all t values.
        hidden = self.y_nn(x)
        # In the final layer params are not shared among t values.

        all_params = [
            self.y0_nn(hidden),
            self.y1_nn(hidden),
            self.y2_nn(hidden),
            self.y3_nn(hidden),
            self.y4_nn(hidden),
            self.y5_nn(hidden),
            self.y6_nn(hidden)
        ]
        t = t.int()
        selected_locs = []
        selected_scales = []
        for batch_idx, ti in enumerate(t):
            param_for_t = all_params[ti.item()]
            selected_loc = param_for_t[0][batch_idx]
            selected_scale = param_for_t[1][batch_idx]
            selected_locs.append(selected_loc)
            selected_scales.append(selected_scale)
        
        locs_tensor = torch.stack(selected_locs)
        scales_tensor = torch.stack(selected_scales)
        return self.y0_nn.make_dist(locs_tensor, scales_tensor)

    def d_dist(self, t, x):
        # The first n-1 layers are identical for all t values.
        hidden = self.d_nn(x)
        # In the final layer params are not shared among t values.
        all_params = [
            self.d0_nn(hidden),
            self.d1_nn(hidden),
            self.d2_nn(hidden),
            self.d3_nn(hidden),
            self.d4_nn(hidden),
            self.d5_nn(hidden),
            self.d6_nn(hidden)
        ]
        t = t.int()
        selected_locs = []
        selected_scales = []
        for batch_idx, ti in enumerate(t):
            param_for_t = all_params[ti.item()]
            selected_loc = param_for_t[0][batch_idx]
            selected_scale = param_for_t[1][batch_idx]
            selected_locs.append(selected_loc)
            selected_scales.append(selected_scale)
        
        locs_tensor = torch.stack(selected_locs)
        scales_tensor = torch.stack(selected_scales)
        return self.d0_nn.make_dist(locs_tensor, scales_tensor)

    def z_dist(self, y, d, t, x):
        # The first n-1 layers are identical for all t values.
        try:
            y_d_x = torch.cat([y.unsqueeze(-1), d.unsqueeze(-1), x], dim=-1)
        except:
            import pdb;pdb.set_trace()
        hidden = self.z_nn(y_d_x)
        # In the final layer params are not shared among t values.
        all_params = [
            self.z0_nn(hidden),
            self.z1_nn(hidden),
            self.z2_nn(hidden),
            self.z3_nn(hidden),
            self.z4_nn(hidden),
            self.z5_nn(hidden),
            self.z6_nn(hidden)
        ]
        t = t.int()
        selected_locs = []
        selected_scales = []
        for batch_idx, ti in enumerate(t):
            param_for_t = all_params[ti.item()]
            selected_loc = param_for_t[0][batch_idx]
            selected_scale = param_for_t[1][batch_idx]
            selected_locs.append(selected_loc)
            selected_scales.append(selected_scale)
        
        locs_tensor = torch.stack(selected_locs)
        scales_tensor = torch.stack(selected_scales)
        return dist.Normal(locs_tensor, scales_tensor).to_event(1) # Independent(Normal(loc: torch.Size([32, 20]), scale: torch.Size([32, 20])), 1)

class TraceCausalEffect_ELBO(Trace_ELBO):
    """
    Loss function for training a :class:`CEVAE`.
    From [1], the CEVAE objective (to maximize) is::

        -loss = ELBO + log q(t|x) + log q(y|t,x)
    """
    def __init__(self, lambdas, elbo_lambdas, additional_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambdas = lambdas
        self.elbo_lambdas = elbo_lambdas
        self.additional_loss = additional_loss

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        # Construct -ELBO part.
        blocked_names = [
            name
            for name, site in guide_trace.nodes.items()
            if site["type"] == "sample" and site["is_observed"]
        ]
        blocked_guide_trace = guide_trace.copy()
        for name in blocked_names:
            del blocked_guide_trace.nodes[name]
        loss, surrogate_loss = super()._differentiable_loss_particle(
            model_trace, blocked_guide_trace, self.elbo_lambdas
        )
        # Add log q terms. 
        for i, name in enumerate(blocked_names): #for 문에서 lambdas 추가
            if self.additional_loss == "log_likelihood":
                # print(name+"q loss") t y d
                log_q = guide_trace.nodes[name]["log_prob_sum"]
                loss = loss - torch_item(log_q) * self.lambdas[i]
                surrogate_loss = surrogate_loss - log_q * self.lambdas[i]
            elif self.additional_loss == "mse": # print(name+"q loss") t y d
                if name == 't':
                    pred = guide_trace.nodes[name]["fn"].logits
                    value = guide_trace.nodes[name]["value"].long()
                    predictive_loss = F.cross_entropy(pred, value) # ce
                else:
                    pred = guide_trace.nodes[name]["fn"].mean
                    value = guide_trace.nodes[name]["value"]
                    predictive_loss = (pred - value) ** 2 # mse
                    predictive_loss = predictive_loss.mean()
                loss = loss + predictive_loss * self.lambdas[i]
                surrogate_loss = surrogate_loss + predictive_loss * self.lambdas[i]
        return loss, surrogate_loss

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, guide, *args, **kwargs))


class CEVAE(nn.Module):
    """
    Main class implementing a Causal Effect VAE [1]. This assumes a graphical model

    .. graphviz:: :graphviz_dot: neato

        digraph {
            Z [pos="1,2!",style=filled];
            X [pos="2,1!"];
            y [pos="1,0!"];
            t [pos="0,1!"];
            Z -> X;
            Z -> t;
            Z -> y;
            t -> y;
        }

    where `t` is a binary treatment variable, `y` is an outcome, `Z` is
    an unobserved confounder, and `X` is a noisy function of the hidden
    confounder `Z`.

    Example::

        cevae = CEVAE(feature_dim=5)
        cevae.fit(x_train, t_train, y_train)
        ite = cevae.ite(x_test)  # individual treatment effect
        ate = ite.mean()         # average treatment effect

    :ivar Model ~CEVAE.model: Generative model.
    :ivar Guide ~CEVAE.guide: Inference model.
    :param int feature_dim: Dimension of the feature space `x`.
    :param str outcome_dist: One of: "bernoulli" (default), "exponential", "laplace",
        "normal", "studentt".
    :param int latent_dim: Dimension of the latent variable `z`.
        Defaults to 20.
    :param int hidden_dim: Dimension of hidden layers of fully connected
        networks. Defaults to 200.
    :param int num_layers: Number of hidden layers in fully connected networks.
    :param int num_samples: Default number of samples for the :meth:`ite`
        method. Defaults to 100.
    """

    def __init__(
        self,
        feature_dim,
        outcome_dist="bernoulli",
        latent_dim=20,
        hidden_dim=200,
        num_layers=3,
        num_samples=100,
        ignore_wandb=False,
        lambdas=None,
        elbo_lambdas=None,
        args=None
    ):
        self.lambdas = lambdas
        self.elbo_lambdas = elbo_lambdas
        self.ignore_wandb=ignore_wandb
        config = dict(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_samples=num_samples,
        )
        if not ignore_wandb:
            wandb.init(entity="mlai_medical_ai" ,project="causal-effect-vae", config=args, group=args.sweep_group)
            wandb.run.name=f"cevae_lambda1_{lambdas[0]}_lambda2_{lambdas[1]}_lambda3_{lambdas[2]}_lr_{args.learning_rate}_lrd_{args.learning_rate_decay}_wd_{args.weight_decay}_beta_{args.beta}"
        for name, size in config.items():
            if not (isinstance(size, int) and size > 0):
                raise ValueError("Expected {} > 0 but got {}".format(name, size))
        config["outcome_dist"] = outcome_dist
        self.feature_dim = feature_dim
        self.num_samples = num_samples

        super().__init__()
        self.model = Model(config)
        self.guide = Guide(config)

        self.x_emb = CEVAEEmbedding(output_size=args.embedding_dim)
        self.transformer_encoder = CEVAETransformer(input_size=args.embedding_dim, hidden_size=args.embedding_dim//2, num_layers=3, num_heads=2, drop_out=0)

    def fit(
        self,
        train_dataset,
        valid_dataset,
        test_dataset,
        num_epochs=100,
        batch_size=100,
        learning_rate=1e-3,
        learning_rate_decay=0.1,
        weight_decay=1e-4,
        log_every=1,
        args=None
    ):
        """
        Train using :class:`~pyro.infer.svi.SVI` with the
        :class:`TraceCausalEffect_ELBO` loss.

        :param ~torch.Tensor x:
        :param ~torch.Tensor t:
        :param ~torch.Tensor y:
        :param int num_epochs: Number of training epochs. Defaults to 100.
        :param int batch_size: Batch size. Defaults to 100.
        :param float learning_rate: Learning rate. Defaults to 1e-3.
        :param float learning_rate_decay: Learning rate decay over all epochs;
            the per-step decay rate will depend on batch size and number of epochs
            such that the initial learning rate will be ``learning_rate`` and the final
            learning rate will be ``learning_rate * learning_rate_decay``.
            Defaults to 0.1.
        :param float weight_decay: Weight decay. Defaults to 1e-4.
        :param int log_every: Log loss each this-many steps. If zero,
            do not log loss. Defaults to 100.
        :return: list of epoch losses
        """
        # assert x.dim() == 2 and x.size(-1) == self.feature_dim
        # assert t.shape == x.shape[:1]
        # assert y.shape == y.shape[:1]
        # assert d.shape == d.shape[:1]
        self.whiten = None
        # dataset = TensorDataset(x, t, y, d)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device='cuda'))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator(device='cuda'))
        logger.info("Training with {} minibatches per epoch".format(len(train_dataloader)))
        num_steps = num_epochs * len(train_dataloader)
        optim = ClippedAdam(
            {
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "lrd": learning_rate_decay ** (1 / num_steps),
            }
        )
        svi = SVI(self.model, self.guide, optim, TraceCausalEffect_ELBO(lambdas=self.lambdas, elbo_lambdas=self.elbo_lambdas, additional_loss=args.additional_loss))
        #--------------------------------------------------------------------
        # losses = []; dataloader=train_dataloader; dataset=train_dataset
        # for epoch in tqdm(range(num_epochs), desc="Epoch"):
        #     for cont, cat, _len, yd, diff, t in tqdm(dataloader, desc="Batch", leave=False):
        #         if False :        
        #             self.whiten = PreWhitener(x) #TODO : 이거 x 데이터 처리 후 들어가야함 수정 필요
        #         x = self.x_emb(cont, cat, _len, diff)
        #         x = self.transformer_encoder(x, _len)
        #         # x = self.whiten(x) #TODO : 이거 x 데이터 처리 후 들어가야함
        #         t = (t*6)
        #         loss = svi.step(x, t, yd[:,0], yd[:,1], size=len(dataset)) / len(dataset)
        #         assert not torch_isnan(loss)
        #         losses.append(loss)
        #         print(
        #             "step {: >5d} loss = {:0.6g}".format(len(losses), loss)
        #         )
        # return losses
        #--------------------------------------------------------------------
        best_mae = float('inf')
        train_losses = []; val_losses = []; test_losses =[];train_epoch_loss=[];val_epoch_loss =[]; test_epoch_loss=[]
        with tqdm(initial = 0, total = num_epochs) as pbar:
            for i in range(num_epochs):
                train_epoch_loss = self.cal_svi_loss(train_dataloader, train_dataset, svi)
                val_epoch_loss = self.cal_svi_loss(valid_dataloader, valid_dataset, svi, eval=True)
                test_epoch_loss = self.cal_svi_loss(test_dataloader, test_dataset, svi, eval=True)
                train_losses= train_losses + train_epoch_loss; val_losses = val_losses + val_epoch_loss; test_losses = test_losses + test_epoch_loss
                train_metrics = self.cal_yd_loss(train_dataloader, train_dataset, data_type="train", args=args)
                val_metrics = self.cal_yd_loss(valid_dataloader, valid_dataset, data_type="val", args=args)
                test_metrics = self.cal_yd_loss(test_dataloader, test_dataset, data_type="test", args=args)

                val_mae_sum = val_metrics['val_y_mae'] + val_metrics['val_d_mae']

                # WandB로 메트릭 로깅
                if not self.ignore_wandb:  
                    ce_metrics = {
                    'train_CE_ELBO': sum(train_epoch_loss)/len(train_epoch_loss),
                    'valid_CE_ELBO': sum(val_losses)/len(val_losses),
                    'test_CE_ELBO': sum(test_losses)/len(test_losses),
                    }
                    epoch_metrics = {**train_metrics, **val_metrics, **test_metrics, **ce_metrics}              
                    
                    wandb.log(epoch_metrics)
                    if val_mae_sum < best_mae:
                        best_mae = val_mae_sum
                        wandb.run.summary["best_epoch"] = i
                        wandb.run.summary["best_train_ceelbo_loss"] = sum(train_epoch_loss)/len(train_epoch_loss)
                        wandb.run.summary["best_val_ceelbo_loss"] = sum(val_losses)/len(val_losses)
                        wandb.run.summary["best_test_ceelbo_loss"] = sum(test_losses)/len(test_losses)
                        
                        wandb.run.summary["best_train_y_mae_loss"] = train_metrics["train_y_mae"]
                        wandb.run.summary["best_train_y_rmse_loss"] = train_metrics["train_y_rmse"]
                        wandb.run.summary["best_train_d_mae_loss"] = train_metrics["train_d_mae"]
                        wandb.run.summary["best_train_d_rmse_loss"] = train_metrics["train_d_rmse"]

                        wandb.run.summary["best_val_y_mae_loss"] = val_metrics["val_y_mae"]
                        wandb.run.summary["best_val_y_rmse_loss"] = val_metrics["val_y_rmse"]
                        wandb.run.summary["best_val_d_mae_loss"] = val_metrics["val_d_mae"]
                        wandb.run.summary["best_val_d_rmse_loss"] = val_metrics["val_d_rmse"]
                        wandb.run.summary["best_val_tot_mae_loss"] = best_mae

                        wandb.run.summary["best_test_y_mae_loss"] = test_metrics["test_y_mae"]
                        wandb.run.summary["best_test_y_rmse_loss"] = test_metrics["test_y_rmse"]
                        wandb.run.summary["best_test_d_mae_loss"] = test_metrics["test_d_mae"]
                        wandb.run.summary["best_test_d_rmse_loss"] = test_metrics["test_d_rmse"]

                pbar.set_description(f'tr_loss: {sum(train_epoch_loss)/len(train_epoch_loss):.4f} / val_loss: {sum(val_losses)/len(val_losses):.4f} / test_loss: {sum(test_losses)/len(test_losses):.4f}')
                pbar.update(1)
                
        return train_losses, val_losses, test_losses
    
    @torch.no_grad()
    def ite(self, x, num_samples=None, batch_size=None):
        r"""
        Computes Individual Treatment Effect for a batch of data ``x``.

        .. math::

            ITE(x) = \mathbb E\bigl[ \mathbf y \mid \mathbf X=x, do(\mathbf t=1) \bigr]
                   - \mathbb E\bigl[ \mathbf y \mid \mathbf X=x, do(\mathbf t=0) \bigr]

        This has complexity ``O(len(x) * num_samples ** 2)``.

        :param ~torch.Tensor x: A batch of data.
        :param int num_samples: The number of monte carlo samples.
            Defaults to ``self.num_samples`` which defaults to ``100``.
        :param int batch_size: Batch size. Defaults to ``len(x)``.
        :return: A ``len(x)``-sized tensor of estimated effects.
        :rtype: ~torch.Tensor
        """
        if num_samples is None:
            num_samples = self.num_samples
        if not torch._C._get_tracing_state():
            assert x.dim() == 2 and x.size(-1) == self.feature_dim

        dataloader = [x] if batch_size is None else DataLoader(x, batch_size=batch_size)
        logger.info("Evaluating {} minibatches".format(len(dataloader)))
        result = []
        for x in dataloader:
            x = self.whiten(x)
            ## TODO : ite evaluation 시 y, d 둘다 봐야함.
            with pyro.plate("num_particles", num_samples, dim=-2):
                with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
                    self.guide(x)
                with poutine.do(data=dict(t=torch.zeros(()))):
                    y0 = poutine.replay(self.model.y_mean, tr.trace)(x)
                with poutine.do(data=dict(t=torch.ones(()))):
                    y1 = poutine.replay(self.model.y_mean, tr.trace)(x)
            ite = (y1 - y0).mean(0)
            if not torch._C._get_tracing_state():
                logger.debug("batch ate = {:0.6g}".format(ite.mean()))
            result.append(ite)
        return torch.cat(result)

    def to_script_module(self):
        """
        Compile this module using :func:`torch.jit.trace_module` ,
        assuming self has already been fit to data.

        :return: A traced version of self with an :meth:`ite` method.
        :rtype: torch.jit.ScriptModule
        """
        self.train(False)
        fake_x = torch.randn(2, self.feature_dim)
        with pyro.validation_enabled(False):
            # Disable check_trace due to nondeterministic nodes.
            result = torch.jit.trace_module(self, {"ite": (fake_x,)}, check_trace=False)
        return result
    
    def cal_svi_loss(self, dataloader, dataset, svi, eval = False):
        losses = []
        if eval:
            self.set_mode("eval")
            with torch.no_grad():  # Disable gradient computation
                for cont_p, cont_c, cat_p, cat_c, _len, yd, diff, t in tqdm(dataloader, desc="Batch", leave=False):
                # for cont, cat, _len, yd, diff, t in tqdm(dataloader, desc="Batch", leave=False):
                    if False :        
                        self.whiten = PreWhitener(x) #TODO : 이거 x 데이터 처리 후 들어가야함 수정 필요
                    x = self.x_emb(cont_p,cont_c, cat_p,cat_c, _len, diff)
                    # x = self.x_emb(cont, cat, _len, diff)
                    x = self.transformer_encoder(x, _len)
                    # x = self.whiten(x) #TODO : 이거 x 데이터 처리 후 들어가야함
                    t = (t*6)
                    loss = svi.step(x, t, yd[:,0], yd[:,1], size=len(dataset)) / len(dataset)
                    assert not torch_isnan(loss)
                    losses.append(loss)
        else:
            self.set_mode("train")
            for cont_p, cont_c, cat_p, cat_c, _len, yd, diff, t in tqdm(dataloader, desc="Batch", leave=False):
            # for cont, cat, _len, yd, diff, t in tqdm(dataloader, desc="Batch", leave=False):
                if False :        
                    self.whiten = PreWhitener(x) #TODO : 이거 x 데이터 처리 후 들어가야함 수정 필요
                x = self.x_emb(cont_p,cont_c, cat_p,cat_c, _len, diff)
                # x = self.x_emb(cont, cat, _len, diff)
                x = self.transformer_encoder(x, _len)
                # x = self.whiten(x) #TODO : 이거 x 데이터 처리 후 들어가야함
                t = (t*6)
                loss = svi.step(x, t, yd[:,0], yd[:,1], size=len(dataset)) / len(dataset)
                assert not torch_isnan(loss)
                losses.append(loss)
        return losses
    
    @torch.no_grad()
    def cal_yd_loss(self, dl, dataset, data_type, args):
        correct_predictions = 0
        total_predictions = 0

        y_diffs = []
        d_diffs = []
        t_losses = []
        for cont_p, cont_c, cat_p, cat_c, _len, yd, diff, t in dl:
        # for cont, cat, _len, yd, diff, t in dl:
            x = self.x_emb(cont_p,cont_c, cat_p,cat_c, _len, diff)
            # x = self.x_emb(cont, cat, _len, diff)
            x = self.transformer_encoder(x, _len)
            # x = self.whiten(x) #TODO : 이거 x 데이터 처리 후 들어가야함
            t = (t*6)
            if args.eval_model=="encoder":
                y_hat, d_hat = self.guide(x=x, t=t, mode="enc_out")
                t_logits = self.guide.t_dist(x).logits
            elif args.eval_model == "decoder":
                z = self.guide(x=x, t=t)
                y_hat, d_hat = self.model(z=z, t=t, mode="mae")
                t_logits = self.model.t_dist(z)
                
            y_hat = utils.inverse_tukey_transformation(y_hat, args=args)
            d_hat = utils.inverse_tukey_transformation(d_hat, args=args)
            y_ori = utils.inverse_tukey_transformation(yd[:, 0], args=args)
            d_ori = utils.inverse_tukey_transformation(yd[:, 1], args=args)

            y_hat = utils.restore_minmax(y_hat, dataset.dataset.a_y, dataset.dataset.b_y)
            d_hat = utils.restore_minmax(d_hat, dataset.dataset.a_d, dataset.dataset.b_d)
            y_ori = utils.restore_minmax(y_ori, dataset.dataset.a_y, dataset.dataset.b_y)
            d_ori = utils.restore_minmax(d_ori, dataset.dataset.a_d, dataset.dataset.b_d)
            
            predicted_class = torch.argmax(t_logits, dim=1)
            correct_predictions += (predicted_class == t.long()).sum().item()
            total_predictions += t.size(0)
            t_loss = F.cross_entropy(t_logits, t.long())
            t_losses.append(t_loss)

            y_diffs.append(y_ori - y_hat)
            d_diffs.append(d_ori - d_hat)

        t_accuracy = correct_predictions / total_predictions
        y_diffs_tensor = torch.cat(y_diffs)
        d_diffs_tensor = torch.cat(d_diffs)
        t_ce_loss = sum(t_losses)/len(t_losses)
        y_mae = torch.mean(torch.abs(y_diffs_tensor))
        y_rmse = torch.sqrt(torch.mean(y_diffs_tensor ** 2))

        d_mae = torch.mean(torch.abs(d_diffs_tensor))
        d_rmse = torch.sqrt(torch.mean(d_diffs_tensor ** 2))
        metrics = {
            f'{data_type}_t_acc': t_accuracy,
            f'{data_type}_t_ce': t_ce_loss,
            f'{data_type}_y_mae': y_mae,
            f'{data_type}_y_rmse': y_rmse,
            f'{data_type}_d_mae': d_mae,
            f'{data_type}_d_rmse': d_rmse,
            f'{data_type}_tot_mae': y_mae+d_mae,
            f'{data_type}_tot_rmse': y_rmse+d_rmse
        }
        return metrics
    
    def set_mode(self, mode):
        for obj in [self.model, self.guide, self.x_emb, self.transformer_encoder]:
            if isinstance(obj, torch.nn.Module):
                if mode == 'train':
                    obj.train()
                elif mode == 'eval':
                    obj.eval()
                else:
                    raise ValueError("Mode must be 'train' or 'eval'")
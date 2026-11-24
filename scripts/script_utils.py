import torch
import torch.nn as nn

from scripts.resnet18.dataloaders import get_LC_samples
from scripts.resnet18.samplers import get_ortho_hull_around_samples, get_ortho_hull_around_samples_w_orig


class SplitLayerNorm(nn.Module):
    def __init__(self, dim, n_free=8, *args, **kwargs):
        super().__init__()
        self.n_free = n_free
        self.norm = nn.LayerNorm(dim - n_free, *args, **kwargs)

    def forward(self, x):
        free = x[..., :self.n_free]
        rest = x[..., self.n_free:]
        return torch.cat([free, self.norm(rest)], dim=-1)


class GrokAlign:
    """
    Reference: https://arxiv.org/abs/1908.02729
    """
    def __init__(self, model: nn.Module, num_projections: int = 1, device=None):
        self.model = model
        self.num_projections = num_projections
        self.device = device if device is not None else next(model.parameters()).device

    def _get_random_projections(self, batch_size, output_dim):
        v = torch.randn(self.num_projections, batch_size, output_dim, device=self.device)
        return v / torch.clamp(v.norm(dim=-1, keepdim=True), min=1e-8)

    def compute_jacobian_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)

        output = self.model(x)
        batch_size, output_dim = output.shape

        v_vectors = self._get_random_projections(batch_size, output_dim)

        norm_sum = torch.zeros(batch_size, device=self.device)

        for i in range(self.num_projections):
            v = v_vectors[i]
            Jv = torch.autograd.grad(
                outputs=output,
                inputs=x,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,
            )[0]

            Jv_flat = Jv.flatten(start_dim=1)
            norm_sum += (Jv_flat ** 2).sum(dim=1)

        scaling = output_dim / self.num_projections
        return torch.sqrt(torch.clamp(norm_sum * scaling, min=1e-8))

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.compute_jacobian_norm(inputs).mean()

def get_hulls(config, train_loader, test_loader, normalize):
    sampler_params = {'n' : config.n_frame if not config.inc_centroid \
                      else config.n_frame+1,
                    'r' : config.r_frame, 'seed':config.seed}

    sampler = get_ortho_hull_around_samples_w_orig if config.inc_centroid \
              else get_ortho_hull_around_samples

    ## select samples for neighborhood computation
    train_LC_batch, _ = get_LC_samples(train_loader,config)
    test_LC_batch, _ = get_LC_samples(test_loader,config)
    if normalize:
      rand_LC_batch = torch.rand_like(test_LC_batch
                                  )*(config.dmax-config.dmin) + config.dmin
    else:
      rand_LC_batch = torch.rand_like(test_LC_batch)   ## Data domain [0,1]


    ## sample hulls/neighborhoods
    train_hulls = sampler(
      train_LC_batch.cuda(),
      **sampler_params
    ).cpu()
    test_hulls = sampler(
        test_LC_batch.cuda(),
        **sampler_params
    ).cpu()
    rand_hulls = sampler(
        rand_LC_batch.cuda(),
        **sampler_params
    ).cpu()

    ## make hull dict. the keys of this dict will be used for logging
    ## you can add hulls separately for different classes as well to
    ## keep track of classwise statistics
    hulls = {
      'train' : train_hulls,
      'test' : test_hulls,
      'rand' : rand_hulls
    }
    return hulls

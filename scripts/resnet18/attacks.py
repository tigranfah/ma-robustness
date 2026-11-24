## Attack models. TODO: Add layerwise attacks

import torch
import torch.nn.functional as F
import time
import torch.nn as nn
import numpy as np

import math

from torchattacks.attack import Attack


class EmbeddingPGD:
    def __init__(self, model, embed, eps, alpha, steps, device):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self.embed = embed

    def attack(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            clean_embeds = self.embed(input_ids)  # [B, T, D]

        adv = clean_embeds.clone().detach()
        adv = adv + torch.zeros_like(adv).uniform_(-self.eps, self.eps)
        adv = adv.detach()

        for _ in range(self.steps):
            adv.requires_grad_(True)
            outputs = self.model(
                inputs_embeds=adv,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + self.alpha * grad.sign()
            adv = torch.clamp(adv, clean_embeds - self.eps, clean_embeds + self.eps)
            adv = adv.detach()

        return adv


class EmbeddingPGDMC:
    def __init__(self, model, eps, alpha, steps, device):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self.embed = model.backbone.gpt_neox.embed_in

    def attack(self, input_ids, attention_mask, labels):
        # Flatten [B, C, T] to [B*C, T] before creating embeddings
        bsz, nchoice, seqlen = input_ids.shape
        flat_ids = input_ids.view(bsz * nchoice, seqlen).to(self.device)
        flat_mask = attention_mask.view(bsz * nchoice, seqlen).to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            clean_embeds = self.embed(flat_ids)

        adv = clean_embeds.clone().detach()
        delta = torch.zeros_like(adv).uniform_(-self.eps, self.eps).to(self.device)

        adv = adv + delta
        adv = adv.detach()

        for _ in range(self.steps):
            adv.requires_grad_(True)

            outputs = self.model(
                inputs_embeds=adv,
                attention_mask=flat_mask,
                labels=labels
            )

            # Untargeted attack: MAXIMIZE the probe's native CrossEntropy loss
            loss = outputs["loss"] 
            grad = torch.autograd.grad(loss, adv)[0]

            adv = adv + self.alpha * grad.sign()
            adv = torch.clamp(adv, clean_embeds - self.eps, clean_embeds + self.eps)
            adv = adv.detach()

            del outputs, loss, grad

        return adv


class PGD(object):
    """
    Untargeted PGD attack. Modified from torchattack.attack to allow non-[0,1] domain data
    https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgd.html#PGD
    """
    def __init__(self, model,
                 eps=8/255,
                 alpha=2/255,
                 steps=10,
                 random_start=True,
                 dmax=None, dmin=None,
                 device='cuda'
                ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = device
        self.dmax = dmax
        self.dmin = dmin
            
    def __call__(self, images, labels):
        ## TODO: add targeted attacks
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device) 

        loss = torch.nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        dmin = images.min() if self.dmin is None else self.dmin
        dmax = images.max() if self.dmax is None else self.dmax

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.rand_like(adv_images)*2*self.eps - self.eps
            adv_images = torch.clamp(adv_images, min=dmin, max=dmax).detach()

        for _ in range(self.steps):
            
            adv_images.requires_grad = True
            with torch.enable_grad():
                outputs = self.model(adv_images)
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=dmin, max=dmax).detach()

        return adv_images
    
    
class TPGD(object):
    r"""
    PGD based on KL-Divergence loss in the paper 'Theoretically Principled Trade-off between Robustness and Accuracy'
    [https://arxiv.org/abs/1901.08573]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images)

    """

    def __init__(self, model,
                 eps=8 / 255,
                 alpha=2 / 255,
                 dmin = 0,
                 dmax = 1,
                 steps=10,
                 device='cuda'
                ):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ["default"]
        self.model = model
        self.dmin = dmin
        self.dmax = dmax
        self.device = device

    def __call__(self, images, labels=None):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        logit_ori = self.model(images).detach()

        adv_images = images + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=self.dmin, max=self.dmax).detach()

        loss = torch.nn.KLDivLoss(reduction="sum")

        for _ in range(self.steps):
            adv_images.requires_grad = True
            logit_adv = self.model(adv_images)

            # Calculate loss
            cost = loss(F.log_softmax(logit_adv, dim=1), F.softmax(logit_ori, dim=1))

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=self.dmin, max=self.dmax).detach()

        return adv_images

class APGD(object):
    r"""
    APGD in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of steps. (Default: 10)
        n_restarts (int): number of random restarts. (Default: 1)
        seed (int): random seed for the starting point. (Default: 0)
        loss (str): loss function optimized. ['ce', 'dlr'] (Default: 'ce')
        eot_iter (int): number of iteration for EOT. (Default: 1)
        rho (float): parameter for step-size update (Default: 0.75)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        norm="Linf",
        eps=8 / 255,
        steps=10,
        n_restarts=1,
        seed=0,
        loss="ce",
        eot_iter=1,
        rho=0.75,
        verbose=False,
        dmin = 0.0,
        dmax = 1.0,
        device = 'cuda',
    ):
#         super().__init__("APGD", model)
        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.supported_mode = ["default"]
        self.model = model
        self.dmin = dmin
        self.dmax = dmax
        self.device = device

    def __call__(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        _, adv_images = self.perturb(images, labels, cheap=True)

        return adv_images

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k * k3 * np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(
            x[np.arange(x.shape[0]), y]
            - x_sorted[:, -2] * ind
            - x_sorted[:, -1] * (1.0 - ind)
        ) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        self.steps_2, self.steps_min, self.size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )
        if self.verbose:
            print(
                "parameters: ", self.steps, self.steps_2, self.steps_min, self.size_decr
            )

        if self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                t.reshape([t.shape[0], -1])
                .abs()
                .max(dim=1, keepdim=True)[0]
                .reshape([-1, 1, 1, 1])
            )  # nopep8
        elif self.norm == "L2":
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
            )  # nopep8
        x_adv = x_adv.clamp(self.dmin, self.dmax)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        if self.loss == "ce":
            criterion_indiv = nn.CrossEntropyLoss(reduction="none")
        elif self.loss == "dlr":
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError("unknown loss")

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                # 1 forward pass (eot_iter = 1)
                logits = self.model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()

            # 1 backward pass (eot_iter = 1)
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = (
            self.eps
            * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
            * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        )  # nopep8
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        # n_reduced = 0
        for i in range(self.steps):
            # gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
                        self.dmin,
                        self.dmax,
                    )
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(
                                x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                                x - self.eps,
                            ),
                            x + self.eps,
                        ),
                        self.dmin,
                        self.dmax,
                    )

                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size * grad / (
                        (grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt(),
                        ),
                        self.dmin,
                        self.dmax,
                    )  # nopep8
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12,
                        ),
                        self.dmin,
                        self.dmax,
                    )  # nopep8

                x_adv = x_adv_1 + 0.0

            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    # 1 forward pass (eot_iter = 1)
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                x_adv[(pred == 0).nonzero().squeeze()] + 0.0
            )  # nopep8
            if self.verbose:
                print("iteration: {} - Best loss: {:.6f}".format(i, loss_best.sum()))

            # check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(),
                        i,
                        k,
                        loss_best.detach().cpu().numpy(),
                        k3=self.thr_decr,
                    )
                    fl_reduce_no_impr = (~reduced_last_check) * (
                        loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
                    )  # nopep8
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.steps_min)

        return x_best, acc, loss_best, x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ["Linf", "L2"]
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        acc = self.model(x).max(1)[1] == y
        # loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print(
                "-------------------------- running {}-attack with epsilon {:.4f} --------------------------".format(
                    self.norm, self.eps
                )
            )
            print("initial accuracy: {:.2%}".format(acc.float().mean()))
        startt = time.time()

        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            if not cheap:
                raise ValueError("not implemented yet")

            else:
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = (
                            x[ind_to_fool].clone(),
                            y[ind_to_fool].clone(),
                        )  # nopep8
                        (
                            best_curr,
                            acc_curr,
                            loss_curr,
                            adv_curr,
                        ) = self.attack_single_run(
                            x_to_fool, y_to_fool
                        )  # nopep8
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        #
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print(
                                "restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s".format(
                                    counter, acc.float().mean(), time.time() - startt
                                )
                            )

            return acc, adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (
                -float("inf")
            )  # nopep8
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.0
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.0

                if self.verbose:
                    print("restart {} - loss: {:.5f}".format(counter, loss_best.sum()))

            return loss_best, adv_best


class Square(Attack):
    r"""
    Square Attack in the paper 'Square Attack: a query-efficient black-box adversarial attack via random search'
    [https://arxiv.org/abs/1912.00049]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        n_queries (int): max number of queries (each restart). (Default: 5000)
        n_restarts (int): number of random restarts. (Default: 1)
        p_init (float): parameter to control size of squares. (Default: 0.8)
        loss (str): loss function optimized ['margin', 'ce'] (Default: 'margin')
        resc_schedule (bool): adapt schedule of p to n_queries (Default: True)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)
        targeted (bool): targeted. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.Square(model, model, norm='Linf', eps=8/255, n_queries=5000, n_restarts=1, eps=None, p_init=.8, seed=0, verbose=False, targeted=False, loss='margin', resc_schedule=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        norm="Linf",
        eps=8 / 255,
        n_queries=5000,
        n_restarts=1,
        p_init=0.8,
        loss="margin",
        resc_schedule=True,
        seed=0,
        verbose=False,
        dmin=0.0,
        dmax=1.0,
    ):
        super().__init__("Square", model)
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.supported_mode = ["default", "targeted"]

        self.dmin = dmin
        self.dmax = dmax

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.perturb(images, labels)

        return adv_images

    def margin_and_loss(self, x, y):
        """
        :param y:        correct labels if untargeted else target labels
        """

        logits = self.get_logits(x)
        xent = F.cross_entropy(logits, y, reduction="none")
        u = torch.arange(x.shape[0])
        y_corr = logits[u, y].clone()
        logits[u, y] = -float("inf")
        y_others = logits.max(dim=-1)[0]

        if not self.targeted:
            if self.loss == "ce":
                return y_corr - y_others, -1.0 * xent
            elif self.loss == "margin":
                return y_corr - y_others, y_corr - y_others
        else:
            if self.loss == "ce":
                return y_others - y_corr, xent
            elif self.loss == "margin":
                return y_others - y_corr, y_others - y_corr

    def init_hyperparam(self, x):
        assert self.norm in ["Linf", "L2"]
        assert not self.eps is None
        assert self.loss in ["ce", "margin"]

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def normalize_delta(self, x):
        if self.norm == "Linf":
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == "L2":
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def lp_norm(self, x):
        if self.norm == "L2":
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def eta_rectangles(self, x, y):
        delta = torch.zeros([x, y]).to(self.device)
        x_c, y_c = x // 2 + 1, y // 2 + 1

        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[
                max(counter2[0], 0) : min(counter2[0] + (2 * counter + 1), x),
                max(0, counter2[1]) : min(counter2[1] + (2 * counter + 1), y),
            ] += (
                1.0 / (torch.Tensor([counter + 1]).view(1, 1).to(self.device) ** 2)
            )  # nopep8
            counter2[0] -= 1
            counter2[1] -= 1

        delta /= (delta ** 2).sum(dim=(0, 1), keepdim=True).sqrt()

        return delta

    def eta(self, s):
        delta = torch.zeros([s, s]).to(self.device)
        delta[: s // 2] = self.eta_rectangles(s // 2, s)
        delta[s // 2 :] = -1.0 * self.eta_rectangles(s - s // 2, s)
        delta /= (delta ** 2).sum(dim=(0, 1), keepdim=True).sqrt()
        if torch.rand([1]) > 0.5:
            delta = delta.permute([1, 0])

        return delta

    def p_selection(self, it):
        """ schedule to decrease the parameter p """

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it:
            p = self.p_init / 512
        else:
            p = self.p_init

        return p

    def attack_single_run(self, x, y):
        with torch.no_grad():
            adv = x.clone()
            c, h, w = x.shape[1:]
            n_features = c * h * w
            n_ex_total = x.shape[0]

            if self.norm == "Linf":
                x_best = torch.clamp(
                    x + self.eps * self.random_choice([x.shape[0], c, 1, w]),
                    self.dmin,
                    self.dmax
                )
                margin_min, loss_min = self.margin_and_loss(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)
                s_init = int(math.sqrt(self.p_init * n_features / c))

                for i_iter in range(self.n_queries):
                    idx_to_fool = (margin_min > 0.0).nonzero().flatten()

                    if len(idx_to_fool) == 0:
                        break

                    x_curr = self.check_shape(x[idx_to_fool])
                    x_best_curr = self.check_shape(x_best[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr = y_curr.unsqueeze(0)
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]

                    p = self.p_selection(i_iter)
                    s = max(int(round(math.sqrt(p * n_features / c))), 1)
                    vh = self.random_int(0, h - s)
                    vw = self.random_int(0, w - s)
                    new_deltas = torch.zeros([c, h, w]).to(self.device)
                    new_deltas[:, vh : vh + s, vw : vw + s] = (
                        2.0 * self.eps * self.random_choice([c, 1, 1])
                    )

                    x_new = x_best_curr + new_deltas
                    x_new = torch.min(
                        torch.max(x_new, x_curr - self.eps), x_curr + self.eps
                    )
                    x_new = torch.clamp(x_new, self.dmin, self.dmax)
                    x_new = self.check_shape(x_new)

                    margin, loss = self.margin_and_loss(x_new, y_curr)

                    # update loss if new loss is better
                    idx_improved = (loss < loss_min_curr).float()

                    loss_min[idx_to_fool] = (
                        idx_improved * loss + (1.0 - idx_improved) * loss_min_curr
                    )

                    # update margin and x_best if new loss is better
                    # or misclassification
                    idx_miscl = (margin <= 0.0).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)

                    margin_min[idx_to_fool] = (
                        idx_improved * margin + (1.0 - idx_improved) * margin_min_curr
                    )
                    idx_improved = idx_improved.reshape([-1, *[1] * len(x.shape[:-1])])
                    x_best[idx_to_fool] = (
                        idx_improved * x_new + (1.0 - idx_improved) * x_best_curr
                    )
                    n_queries[idx_to_fool] += 1.0

                    ind_succ = (margin_min <= 0.0).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        print(
                            "{}".format(i_iter + 1),
                            "- success rate={}/{} ({:.2%})".format(
                                ind_succ.numel(),
                                n_ex_total,
                                float(ind_succ.numel()) / n_ex_total,
                            ),
                            "- avg # queries={:.1f}".format(
                                n_queries[ind_succ].mean().item()
                            ),
                            "- med # queries={:.1f}".format(
                                n_queries[ind_succ].median().item()
                            ),
                            "- loss={:.3f}".format(loss_min.mean()),
                        )

                    if ind_succ.numel() == n_ex_total:
                        break

            elif self.norm == "L2":
                delta_init = torch.zeros_like(x)
                s = h // 5
                sp_init = (h - s * 5) // 2
                vh = sp_init + 0
                for _ in range(h // s):
                    vw = sp_init + 0
                    for _ in range(w // s):
                        delta_init[:, :, vh : vh + s, vw : vw + s] += self.eta(s).view(
                            1, 1, s, s
                        ) * self.random_choice([x.shape[0], c, 1, 1])
                        vw += s
                    vh += s

                x_best = torch.clamp(
                    x + self.normalize_delta(delta_init) * self.eps,
                    self.dmin,
                    self.dmax
                )
                margin_min, loss_min = self.margin_and_loss(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)
                s_init = int(math.sqrt(self.p_init * n_features / c))

                for i_iter in range(self.n_queries):
                    idx_to_fool = (margin_min > 0.0).nonzero().flatten()

                    if len(idx_to_fool) == 0:
                        break

                    x_curr = self.check_shape(x[idx_to_fool])
                    x_best_curr = self.check_shape(x_best[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr = y_curr.unsqueeze(0)
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]

                    delta_curr = x_best_curr - x_curr
                    p = self.p_selection(i_iter)
                    s = max(int(round(math.sqrt(p * n_features / c))), 3)
                    if s % 2 == 0:
                        s += 1

                    vh = self.random_int(0, h - s)
                    vw = self.random_int(0, w - s)
                    new_deltas_mask = torch.zeros_like(x_curr)
                    new_deltas_mask[:, :, vh : vh + s, vw : vw + s] = 1.0
                    norms_window_1 = (
                        (delta_curr[:, :, vh : vh + s, vw : vw + s] ** 2)
                        .sum(dim=(-2, -1), keepdim=True)
                        .sqrt()
                    )

                    vh2 = self.random_int(0, h - s)
                    vw2 = self.random_int(0, w - s)
                    new_deltas_mask_2 = torch.zeros_like(x_curr)
                    new_deltas_mask_2[:, :, vh2 : vh2 + s, vw2 : vw2 + s] = 1.0

                    norms_image = self.lp_norm(x_best_curr - x_curr)
                    mask_image = torch.max(new_deltas_mask, new_deltas_mask_2)
                    norms_windows = self.lp_norm(delta_curr * mask_image)

                    new_deltas = torch.ones([x_curr.shape[0], c, s, s]).to(self.device)
                    new_deltas *= self.eta(s).view(1, 1, s, s) * self.random_choice(
                        [x_curr.shape[0], c, 1, 1]
                    )
                    old_deltas = delta_curr[:, :, vh : vh + s, vw : vw + s] / (
                        1e-12 + norms_window_1
                    )
                    new_deltas += old_deltas
                    new_deltas = (
                        new_deltas
                        / (
                            1e-12
                            + (new_deltas ** 2).sum(dim=(-2, -1), keepdim=True).sqrt()
                        )
                        * (
                            torch.max(
                                (self.eps * torch.ones_like(new_deltas)) ** 2
                                - norms_image ** 2,
                                torch.zeros_like(new_deltas),
                            )
                            / c
                            + norms_windows ** 2
                        ).sqrt()
                    )
                    delta_curr[:, :, vh2 : vh2 + s, vw2 : vw2 + s] = 0.0
                    delta_curr[:, :, vh : vh + s, vw : vw + s] = new_deltas + 0

                    x_new = torch.clamp(
                        x_curr + self.normalize_delta(delta_curr) * self.eps, 0.0, 1.0
                    )
                    x_new = self.check_shape(x_new)
                    norms_image = self.lp_norm(x_new - x_curr)

                    margin, loss = self.margin_and_loss(x_new, y_curr)

                    # update loss if new loss is better
                    idx_improved = (loss < loss_min_curr).float()

                    loss_min[idx_to_fool] = (
                        idx_improved * loss + (1.0 - idx_improved) * loss_min_curr
                    )

                    # update margin and x_best if new loss is better
                    # or misclassification
                    idx_miscl = (margin <= 0.0).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)

                    margin_min[idx_to_fool] = (
                        idx_improved * margin + (1.0 - idx_improved) * margin_min_curr
                    )
                    idx_improved = idx_improved.reshape([-1, *[1] * len(x.shape[:-1])])
                    x_best[idx_to_fool] = (
                        idx_improved * x_new + (1.0 - idx_improved) * x_best_curr
                    )
                    n_queries[idx_to_fool] += 1.0

                    ind_succ = (margin_min <= 0.0).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        print(
                            "{}".format(i_iter + 1),
                            "- success rate={}/{} ({:.2%})".format(
                                ind_succ.numel(),
                                n_ex_total,
                                float(ind_succ.numel()) / n_ex_total,
                            ),
                            "- avg # queries={:.1f}".format(
                                n_queries[ind_succ].mean().item()
                            ),
                            "- med # queries={:.1f}".format(
                                n_queries[ind_succ].median().item()
                            ),
                            "- loss={:.3f}".format(loss_min.mean()),
                        )

                    assert (x_new != x_new).sum() == 0
                    assert (x_best != x_best).sum() == 0

                    if ind_succ.numel() == n_ex_total:
                        break

        return n_queries, x_best

    def perturb(self, x, y=None):
        """
        :param x:           clean images
        :param y:           untargeted attack -> clean labels,
                            if None we use the predicted labels
                            targeted attack -> target labels, if None random classes,
                            different from the predicted ones, are sampled
        """

        self.init_hyperparam(x)

        adv = x.clone()
        if y is None:
            if not self.targeted:
                with torch.no_grad():
                    output = self.get_logits(x)
                    y_pred = output.max(1)[1]
                    y = y_pred.detach().clone().long().to(self.device)
            else:
                with torch.no_grad():
                    y = self.get_target_label(x, None)
        else:
            if not self.targeted:
                y = y.detach().clone().long().to(self.device)
            else:
                y = self.get_target_label(x, y)

        if not self.targeted:
            acc = self.get_logits(x).max(1)[1] == y
        else:
            acc = self.get_logits(x).max(1)[1] != y

        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                _, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)

                output_curr = self.get_logits(adv_curr)
                if not self.targeted:
                    acc_curr = output_curr.max(1)[1] == y_to_fool
                else:
                    acc_curr = output_curr.max(1)[1] != y_to_fool
                ind_curr = (acc_curr == 0).nonzero().squeeze()

                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                if self.verbose:
                    print(
                        "restart {} - robust accuracy: {:.2%}".format(
                            counter, acc.float().mean()
                        ),
                        "- cum. time: {:.1f} s".format(time.time() - startt),
                    )

        return adv
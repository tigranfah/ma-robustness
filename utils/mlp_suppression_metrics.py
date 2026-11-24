import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.fcn import FCN
from models.model_utils import add_hooks_fcn
from scripts.script_utils import get_hulls
from utils.lc_utils import get_intersections_for_hulls
from utils.mlp_compute_metrics import get_loaders


def filter_loader_by_class(loader, target_class):
    indices = [
        index
        for index in range(len(loader.dataset))
        if loader.dataset[index][1] == target_class
    ]
    dataset = Subset(loader.dataset, indices)
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )


class CoordinateSuppressor:
    def __init__(self, module, coordinate):
        self.coordinate = coordinate
        self.factor = 1.0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        suppressed = output.clone()
        suppressed[..., self.coordinate] *= self.factor
        return suppressed

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def identify_massive_coordinate(
    model, loader, layer, device, metric="observed_max", max_batches=None
):
    activation_sum = None
    activation_max = None
    top_counts = None
    sample_count = 0

    def capture(module, inputs, output):
        nonlocal activation_sum, activation_max, top_counts, sample_count
        activation = output.detach().abs()
        batch_sum = activation.sum(dim=0)
        batch_max = activation.amax(dim=0)
        batch_top_counts = torch.bincount(
            activation.argmax(dim=-1), minlength=activation.size(-1)
        )
        activation_sum = batch_sum if activation_sum is None else activation_sum + batch_sum
        activation_max = batch_max if activation_max is None else torch.maximum(activation_max, batch_max)
        top_counts = batch_top_counts if top_counts is None else top_counts + batch_top_counts
        sample_count += output.size(0)

    handle = layer.register_forward_hook(capture)
    model.eval()
    for batch_index, (inputs, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        model(inputs.to(device))
    handle.remove()

    coordinate_scores = {
        "mean_abs": activation_sum / sample_count,
        "observed_max": activation_max,
        "top_frequency": top_counts / sample_count,
    }[metric]
    coordinate = coordinate_scores.argmax().item()
    return coordinate, coordinate_scores.cpu()


@torch.no_grad()
def compute_layer_activation_stats(model, loader, layer, coordinate, device, max_batches=None):
    selected_sum = 0.0
    maximum_sum = 0.0
    sample_count = 0

    def capture(module, inputs, output):
        nonlocal selected_sum, maximum_sum, sample_count
        activation = output.detach().abs()
        selected_sum += activation[..., coordinate].sum().item()
        maximum_sum += activation.amax(dim=-1).sum().item()
        sample_count += activation.size(0)

    handle = layer.register_forward_hook(capture)
    model.eval()
    for batch_index, (inputs, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        model(inputs.to(device))
    handle.remove()

    return selected_sum / sample_count, maximum_sum / sample_count


def build_model(config, input_dim, output_dim, device):
    model = FCN(
        depth=config.depth,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config.hidden_dim,
        bias=config.bias,
        act_fn=config.act_fn,
        norm_type=config.norm_type,
        residual_stream=config.residual_stream,
        init_scale=config.init_scale,
        block_type=config.block_type,
    ).to(device)
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def resolve_intervention_layer(model, layer_index):
    if layer_index < 0:
        layer_index += len(model.layers)
    if not 0 <= layer_index < len(model.layers) - 1:
        raise ValueError(
            "intervention_layer must select a hidden layer, not the output classifier"
        )
    return layer_index


def run_experiment(config):
    device = torch.device(config.device)
    train_loader, test_loader, input_dim, output_dim, evaluate_adv, compute_accuracy = get_loaders(config)
    model = build_model(config, input_dim, output_dim, device)
    layer_index = resolve_intervention_layer(model, config.intervention_layer)
    intervention_layer = model.layers[layer_index]

    coordinate_loader = train_loader if config.coordinate_split == "train" else test_loader
    if config.coordinate_class is not None:
        coordinate_loader = filter_loader_by_class(coordinate_loader, config.coordinate_class)

    evaluation_train_loader = train_loader
    evaluation_test_loader = test_loader
    if config.evaluation_class is not None:
        evaluation_train_loader = filter_loader_by_class(train_loader, config.evaluation_class)
        evaluation_test_loader = filter_loader_by_class(test_loader, config.evaluation_class)

    coordinate, coordinate_scores = identify_massive_coordinate(
        model,
        coordinate_loader,
        intervention_layer,
        device,
        config.coordinate_metric,
        config.max_batches,
    )
    top_values, top_indices = coordinate_scores.topk(min(5, coordinate_scores.numel()))
    print(f"Intervention layer: {layer_index}")
    print(f"Coordinate metric: {config.coordinate_metric}")
    print(f"Massive coordinate: {coordinate}")
    print(f"Top coordinates: {list(zip(top_indices.tolist(), top_values.tolist()))}")

    hulls = get_hulls(
        config, evaluation_train_loader, evaluation_test_loader, normalize=True
    )
    suppressor = CoordinateSuppressor(intervention_layer, coordinate)
    model, layer_names, activation_buffer = add_hooks_fcn(model, verbose=False)

    results = {
        "suppression_factors": [],
        "train_accs": [],
        "test_accs": [],
        "adv_accs": [],
        "train_lc": [],
        "train_lc_std": [],
        "train_lc_per_layer": [],
        "test_lc": [],
        "test_lc_std": [],
        "test_lc_per_layer": [],
        "selected_coordinate_mean_abs": [],
        "layer_max_mean_abs": [],
    }

    for factor in tqdm(config.suppression_factors, desc="Suppression factors"):
        suppressor.factor = factor
        model.eval()
        train_acc = compute_accuracy(model, evaluation_train_loader, device)
        model.eval()
        test_acc = compute_accuracy(model, evaluation_test_loader, device)
        model.eval()
        adv_acc = evaluate_adv(
            model,
            evaluation_test_loader,
            config.adv_eps,
            config.adv_alpha,
            config.adv_itrs,
            config.dmin,
            config.dmax,
            device,
        )
        model.eval()

        selected_mean, layer_max_mean = compute_layer_activation_stats(
            model,
            evaluation_test_loader,
            intervention_layer,
            coordinate,
            device,
            config.max_batches,
        )

        n_inters_train, _ = get_intersections_for_hulls(
            hulls=hulls["train"],
            model=model,
            layer_names=layer_names,
            activation_buffer=activation_buffer,
            batch_size=config.lc_batch_size,
            verbose=False,
        )
        n_inters_test, _ = get_intersections_for_hulls(
            hulls=hulls["test"],
            model=model,
            layer_names=layer_names,
            activation_buffer=activation_buffer,
            batch_size=config.lc_batch_size,
            verbose=False,
        )
        train_lc = n_inters_train.float().sum(dim=1)
        test_lc = n_inters_test.float().sum(dim=1)

        results["suppression_factors"].append(factor)
        results["train_accs"].append(train_acc * 100.0)
        results["test_accs"].append(test_acc * 100.0)
        results["adv_accs"].append(adv_acc.item() * 100.0)
        results["train_lc"].append(train_lc.mean().item())
        results["train_lc_std"].append(train_lc.std().item())
        results["train_lc_per_layer"].append(n_inters_train.float().mean(dim=0).numpy())
        results["test_lc"].append(test_lc.mean().item())
        results["test_lc_std"].append(test_lc.std().item())
        results["test_lc_per_layer"].append(n_inters_test.float().mean(dim=0).numpy())
        results["selected_coordinate_mean_abs"].append(selected_mean)
        results["layer_max_mean_abs"].append(layer_max_mean)

        print(
            f"factor={factor:.3f} test_acc={test_acc * 100.0:.2f} "
            f"adv_acc={adv_acc.item() * 100.0:.2f} test_lc={test_lc.mean().item():.3f}"
        )

    suppressor.remove()
    metrics = {name: np.asarray(values) for name, values in results.items()}
    metrics.update(
        {
            "massive_coordinate": np.asarray(coordinate),
            "coordinate_scores": coordinate_scores.numpy(),
            "intervention_layer": np.asarray(layer_index),
            "coordinate_metric": np.asarray(config.coordinate_metric),
            "coordinate_class": np.asarray(config.coordinate_class),
            "evaluation_class": np.asarray(config.evaluation_class),
            "lc_layer_names": np.asarray(layer_names),
            "adv_eps": np.asarray(config.adv_eps),
            "r_frame": np.asarray(config.r_frame),
            "checkpoint_path": np.asarray(config.checkpoint_path),
        }
    )

    checkpoint_name = Path(config.checkpoint_path).stem
    class_suffix = (
        f"-class:{config.evaluation_class}"
        if config.evaluation_class is not None
        else ""
    )
    save_path = Path(config.save_path) / (
        f"{checkpoint_name}-layer:{layer_index}-coord:{coordinate}"
        f"{class_suffix}-adv_eps:{config.adv_eps}-r_frame:{config.r_frame}.npz"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **metrics)
    print(f"Metrics saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="utils/metrics/mlp_suppression")
    parser.add_argument("--dataset", choices=["mnist"], default="mnist")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--coordinate_split", choices=["train", "test"], default="train")
    parser.add_argument("--coordinate_class", type=int, default=None)
    parser.add_argument("--evaluation_class", type=int, default=None)
    parser.add_argument(
        "--coordinate_metric",
        choices=["observed_max", "mean_abs", "top_frequency"],
        default="observed_max",
    )
    parser.add_argument("--intervention_layer", type=int, default=-2)
    parser.add_argument(
        "--suppression_factors",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
    )
    parser.add_argument("--max_batches", type=int, default=8)
    parser.add_argument("--train_points", type=int, default=1024)
    parser.add_argument("--test_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--residual_stream", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--act_fn", type=str, default="relu")
    parser.add_argument("--norm_type", type=str, default=None)
    parser.add_argument("--block_type", type=str, default="pre_act")
    parser.add_argument("--init_scale", type=float, default=8.0)
    parser.add_argument("--adv_eps", type=float, default=0.2)
    parser.add_argument("--adv_alpha", type=float, default=4 / 255)
    parser.add_argument("--adv_itrs", type=int, default=10)
    parser.add_argument("--dmin", type=float, default=None)
    parser.add_argument("--dmax", type=float, default=None)
    parser.add_argument("--approx_n", type=int, default=1024)
    parser.add_argument("--n_frame", type=int, default=25)
    parser.add_argument("--r_frame", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lc_batch_size", type=int, default=256)
    parser.add_argument("--inc_centroid", action="store_true")
    args = parser.parse_args()

    if args.dmin is None or args.dmax is None:
        args.dmin = -0.4242
        args.dmax = 2.8215

    run_experiment(args)

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import tqdm


def plot_repr_3d_vit_top_activations(model, image_tensor, layers=None, top_k=10):
    device = next(model.parameters()).device
    input_tensor = image_tensor.to(device)

    total_layers = len(model.blocks)
    layers_to_plot = layers if layers is not None else list(range(total_layers))

    # 1. Hook-Free Extraction using timm's native API
    model.eval()
    with torch.no_grad():
        try:
            intermediates = model.forward_intermediates(
                input_tensor,
                indices=layers_to_plot,
                return_prefix_tokens=True,
                intermediates_only=True
            )
        except TypeError:
            _, intermediates = model.forward_intermediates(
                input_tensor,
                indices=layers_to_plot,
                return_prefix_tokens=True
            )

    # 2. Setup Plot Grid
    num_plots = len(layers_to_plot)
    cols = 3
    rows = (num_plots + cols - 1) // cols

    fig = plt.figure(figsize=(6 * cols, 5 * rows))

    # 3. Process and Plot per Layer
    for plot_idx, layer_idx in enumerate(tqdm(layers_to_plot)):

        interm = intermediates[plot_idx]

        if isinstance(interm, tuple) and len(interm) == 2:
            spatial_tokens, prefix_tokens = interm

            # 4D tensors come from convolutional-stem hybrids, NOT standard ViTs.
            # Flatten so the rest of the pipeline treats everything as [B, SeqLen, D].
            if spatial_tokens.dim() == 4:
                B, C, H, W = spatial_tokens.shape
                spatial_tokens = spatial_tokens.view(B, C, H * W).transpose(1, 2)

            layer_out = torch.cat([prefix_tokens, spatial_tokens], dim=1)[0].cpu()
        else:
            layer_out = interm[0].cpu()

        # ── Massive Activation Detection ─────────────────────────────────────────
        # Rank by peak absolute value per token, not sum — a single outlier
        # dimension is the canonical massive-activation signature.
        spatial_out = layer_out[1:]
        token_peak  = spatial_out.abs().max(dim=1).values

        actual_k = min(top_k, token_peak.shape[0])
        _, top_spatial_indices = torch.topk(token_peak, k=actual_k)
        top_spatial_indices = top_spatial_indices + 1  # re-add [CLS] offset

        cls_index      = torch.tensor([0])
        combined       = torch.cat([cls_index, top_spatial_indices])
        top_indices, _ = torch.sort(combined)
        # ─────────────────────────────────────────────────────────────────────────

        token_labels = ['[CLS]' if idx == 0 else f'P{idx.item()}' for idx in top_indices]
        repr_data = layer_out[top_indices].abs().numpy()  # [S, D]
        S, D = repr_data.shape
        
        # ── Vectorised 3D Stem Plot via Line3DCollection ──────────────────────────
        token_coords = np.repeat(np.arange(S), D)
        dim_coords   = np.tile(np.arange(D), S)
        values       = repr_data.flatten()

        mask         = values > 0.05
        token_coords = token_coords[mask]
        dim_coords   = dim_coords[mask]
        values       = values[mask]

        # segments shape: [N_lines, 2, 3]  —  each row is (base, tip) in 3D
        segments = np.stack([
            np.stack([token_coords, dim_coords, np.zeros_like(values)], axis=1),
            np.stack([token_coords, dim_coords, values],                axis=1),
        ], axis=1)

        ax = fig.add_subplot(rows, cols, plot_idx + 1, projection='3d')

        lc = Line3DCollection(segments, colors='blue', alpha=1.0, linewidth=3.0)
        ax.add_collection3d(lc)

        # add_collection3d doesn't autoscale — set limits explicitly
        ax.set_xlim(0, S - 1)
        ax.set_ylim(0, D - 1)
        ax.set_zlim(0, values.max() if values.size > 0 else 1)
        # ax.set_zlim(0, 60)
        # ─────────────────────────────────────────────────────────────────────────

        ax.set_xticks(range(S))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_title(f"Layer {layer_idx}", fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_pre_mlp_vit_top_activations(model, image_tensor, layers=None, top_k=10):
    device = next(model.parameters()).device
    input_tensor = image_tensor.to(device)

    total_layers = len(model.blocks)
    layers_to_plot = layers if layers is not None else list(range(total_layers))

    model.eval()
    with torch.no_grad():
        # manually run the patch embedding and pos embedding
        x = model.patch_embed(input_tensor)
        x = model._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)

        pre_mlp_per_layer = {}

        for layer_idx, block in enumerate(model.blocks):
            # attention + residual
            attn_out = block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            x = x + attn_out

            if layer_idx in layers_to_plot:
                # this is exactly what goes into the MLP: norm2 applied to post-attn residual
                pre_mlp_per_layer[layer_idx] = block.norm2(x)[0].cpu()

            # mlp + residual
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

    num_plots = len(layers_to_plot)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    fig = plt.figure(figsize=(6 * cols, 5 * rows))

    for plot_idx, layer_idx in enumerate(tqdm(layers_to_plot)):
        layer_out = pre_mlp_per_layer[layer_idx]   # [SeqLen, D]

        spatial_out = layer_out[1:]
        token_peak  = spatial_out.abs().max(dim=1).values

        actual_k           = min(top_k, token_peak.shape[0])
        _, top_spatial_idx = torch.topk(token_peak, k=actual_k)
        top_spatial_idx    = top_spatial_idx + 1

        cls_index      = torch.tensor([0])
        combined       = torch.cat([cls_index, top_spatial_idx])
        top_indices, _ = torch.sort(combined)

        token_labels = ['[CLS]' if idx == 0 else f'P{idx.item()}' for idx in top_indices]
        repr_data    = layer_out[top_indices].abs().numpy()
        S, D         = repr_data.shape

        token_coords = np.repeat(np.arange(S), D)
        dim_coords   = np.tile(np.arange(D), S)
        values       = repr_data.flatten()

        mask         = values > 0.05
        token_coords = token_coords[mask]
        dim_coords   = dim_coords[mask]
        values       = values[mask]

        segments = np.stack([
            np.stack([token_coords, dim_coords, np.zeros_like(values)], axis=1),
            np.stack([token_coords, dim_coords, values],                axis=1),
        ], axis=1)

        ax = fig.add_subplot(rows, cols, plot_idx + 1, projection='3d')
        lc = Line3DCollection(segments, colors='blue', alpha=1.0, linewidth=3.0)
        ax.add_collection3d(lc)

        ax.set_xlim(0, S - 1)
        ax.set_ylim(0, D - 1)
        ax.set_zlim(0, values.max() if values.size > 0 else 1)
        ax.set_xticks(range(S))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_title(f"Layer {layer_idx} — pre-MLP", fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_repr_3d_ijepa_top_activations(model, pixel_values, layers=None, top_k=10):
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)

    total_layers = len(model.encoder.layer)
    layers_to_plot = layers if layers is not None else list(range(total_layers))

    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values, output_hidden_states=True)

    # hidden_states: tuple of [B, SeqLen, D] — one per layer + embedding layer
    # index 0 is embedding output, 1..32 are transformer layer outputs
    hidden_states = outputs.hidden_states

    num_plots = len(layers_to_plot)
    cols = 3
    rows = (num_plots + cols - 1) // cols

    fig = plt.figure(figsize=(6 * cols, 5 * rows))

    for plot_idx, layer_idx in enumerate(tqdm(layers_to_plot)):
        # +1 because index 0 is embedding output
        layer_out = hidden_states[layer_idx + 1][0].cpu()  # [SeqLen, D]

        # CLS token is at index 0, patch tokens are 1..
        spatial_out = layer_out[1:]
        token_peak  = spatial_out.abs().max(dim=1).values

        actual_k = min(top_k, token_peak.shape[0])
        _, top_spatial_indices = torch.topk(token_peak, k=actual_k)
        top_spatial_indices = top_spatial_indices + 1  # re-add CLS offset

        cls_index      = torch.tensor([0])
        combined       = torch.cat([cls_index, top_spatial_indices])
        top_indices, _ = torch.sort(combined)

        token_labels = ['[CLS]' if idx == 0 else f'P{idx.item()}' for idx in top_indices]
        repr_data = layer_out[top_indices].abs().numpy()  # [S, D]
        S, D = repr_data.shape

        token_coords = np.repeat(np.arange(S), D)
        dim_coords   = np.tile(np.arange(D), S)
        values       = repr_data.flatten()

        mask         = values > 0.05
        token_coords = token_coords[mask]
        dim_coords   = dim_coords[mask]
        values       = values[mask]

        segments = np.stack([
            np.stack([token_coords, dim_coords, np.zeros_like(values)], axis=1),
            np.stack([token_coords, dim_coords, values],                axis=1),
        ], axis=1)

        ax = fig.add_subplot(rows, cols, plot_idx + 1, projection='3d')
        lc = Line3DCollection(segments, colors='blue', alpha=1.0, linewidth=3.0)
        ax.add_collection3d(lc)

        ax.set_xlim(0, S - 1)
        ax.set_ylim(0, D - 1)
        ax.set_zlim(0, values.max() if values.size > 0 else 1)

        ax.set_xticks(range(S))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_title(f"Layer {layer_idx}", fontsize=14)

    plt.tight_layout()
    plt.show()
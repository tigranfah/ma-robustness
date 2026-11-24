import torch.nn as nn
import numpy as np
from transformers.activations import GELUActivation


def split_generator(generator):
    """
    Split tuple elements in a generator into lists
    """
    return list(map(list, zip(*generator)))


def flatten_model(model,whitelist_keywords=None):
    """
    flatten the modules in a model into a list
    whitelist_keywords: modules containing keywords will not be split
    """

    names,modules = split_generator(model.named_children())

    for i,nm in enumerate(zip(names,modules)):

        n,m = nm

        if whitelist_keywords is not None:
            if any([n.find(each) != -1 for each in whitelist_keywords]):
                continue ## do not split

        if len(list(m.children()))>0:

            new_names,new_modules = split_generator(m.named_children())

            modules += new_modules
            modules[i] = None

            names += [f'{names[i]}::{each}' for each in new_names]
            names[i] = None

    return [each for each in names if each is not None], [each for each in modules if each is not None]


def add_hooks_fcn(model, verbose=False):
    """
    Add hooks to the custom FCN model.
    Captures the output of the activation function in each Block, 
    plus the final Linear layer.
    """
    
    # Flatten the model using your provided utility
    names, modules = flatten_model(model)
    assert len(names) == len(modules)

    layer_ids = []
    
    for i, (name, mod) in enumerate(zip(names, modules)):
        if isinstance(mod, nn.ReLU):
            layer_ids.append(i)
            
        elif isinstance(mod, nn.Linear) and len(name.split("::")) == 2:
            layer_ids.append(i)

    layer_ids = np.asarray(layer_ids)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for each in layer_ids:
        modules[each].register_forward_hook(get_activation(names[each]))

    layer_names = np.sort(np.asarray(names)[layer_ids])

    if verbose:
        print('Adding Hook to', layer_names)

    return model, layer_names.tolist(), activation


def add_hooks_preact_resnet18(model, use_bn, verbose=False):
    """
    Add hooks to preact resnet
    """

    names,modules = flatten_model(model)
    assert len(names) == len(modules)

    ## add hooks to bns only. bn outputs are always passed through relus (even for skip connections)
    norm_module = nn.modules.BatchNorm2d if use_bn else nn.modules.Identity
    layer_ids = np.asarray([i for i,each in enumerate(modules) if (type(each)==norm_module)])

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for each in layer_ids:
        modules[each].register_forward_hook(get_activation(names[each]))

    layer_names = np.sort(np.asarray(names)[layer_ids])

    if verbose:
        print('Adding Hook to',layer_names)

    return model, layer_names, activation


def add_hooks_vit(model, verbose=False):
    names, modules = flatten_model(model)
    assert len(names) == len(modules)

    layer_ids = []
    for i, (name, mod) in enumerate(zip(names, modules)):
        if isinstance(mod, nn.GELU):
            layer_ids.append(i)

    layer_ids = np.asarray(layer_ids)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # hook the INPUT to GELU (pre-activation), slice CLS token
            # activation[name] = input[0][:, 0, :].detach()
            activation[name] = input[0].detach().view(input[0].shape[0], -1)
        return hook

    for each in layer_ids:
        modules[each].register_forward_hook(get_activation(names[each]))

    layer_names = np.sort(np.asarray(names)[layer_ids]).tolist()

    if verbose:
        print('Adding Hook to', layer_names)

    return model, layer_names, activation


class PythiaEmbedWrapper(nn.Module):
    def __init__(self, model, seq_len, hidden_dim):
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [B, T*D] flat hull vector
        embeds = x.view(x.shape[0], self.seq_len, self.hidden_dim)
        return self.model(inputs_embeds=embeds)


def add_hooks_pythia(model):
    names, modules = flatten_model(model)
    layer_ids = [i for i, mod in enumerate(modules) if isinstance(mod, GELUActivation)]
    activation = {}

    def get_activation(name):
        def hook(module, input, output):
            activation[name] = input[0].detach().view(input[0].shape[0], -1)  # [B, T * mlp_dim]
        return hook

    for idx in layer_ids:
        modules[idx].register_forward_hook(get_activation(names[idx]))

    layer_names = [names[i] for i in layer_ids]
    return model, layer_names, activation
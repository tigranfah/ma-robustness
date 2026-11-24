import torch
import random
import numpy as np
import torch.nn as nn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def add_hooks_preact_resnet18(model, config, verbose=False):
    """
    Add hooks to preact resnet
    """

    names,modules = flatten_model(model)
    assert len(names) == len(modules)

    ## add hooks to bns only. bn outputs are always passed through relus (even for skip connections)
    norm_module = ch.nn.modules.BatchNorm2d if config.use_bn else ch.nn.modules.Identity
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
    
    # Iterate through flattened modules to find target layers
    for i, (name, mod) in enumerate(zip(names, modules)):
        # 1. Hook all ReLU activations inside the Blocks
        if isinstance(mod, nn.ReLU):
            layer_ids.append(i)
            
        # 2. Hook the final output layer. 
        # Inside the flattened list, internal linear layers look like 'layers::0::linear'
        # The final linear layer sits at the top level of the Sequential, looking like 'layers::2'
        elif isinstance(mod, nn.Linear) and len(name.split("::")) == 2:
            layer_ids.append(i)

    layer_ids = np.asarray(layer_ids)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Register the forward hooks
    for each in layer_ids:
        modules[each].register_forward_hook(get_activation(names[each]))

    layer_names = np.sort(np.asarray(names)[layer_ids])

    if verbose:
        print('Adding Hook to', layer_names)

    return model, layer_names.tolist(), activation
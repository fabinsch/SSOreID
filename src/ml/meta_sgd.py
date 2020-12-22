from sacred import Experiment
import sacred
import os.path as osp
import os
import numpy as np
import yaml
import time
import functools, operator

import torch
import torch.nn
from torch.autograd import grad
from torch.nn import functional as F

from tracktor.config import get_output_dir, get_tb_dir
import random
from ml.utils import load_dataset, get_ML_settings, get_plotter, save_checkpoint, sample_task

import learn2learn as l2l
from learn2learn.utils import clone_module, clone_parameters
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MetaSGD(l2l.algorithms.MetaSGD):
    def __init__(self, model, lr=1.0, first_order=False, allow_unused=None, allow_nograd=False, lrs=None, lrs_inner=None,
                 global_LR=False, others_neuron_b=None, others_neuron_w=None, template_neuron_b=None,
                 template_neuron_w=None):
        super(l2l.algorithms.MetaSGD, self).__init__()
        self.module = model
        self.module.global_LR = global_LR
        # LR per layer
        # if lrs is None:
        #     lrs = [torch.ones(1).to(device) * lr for p in model.parameters()]
        #     lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])

        self.others_neuron_weight = others_neuron_w
        self.others_neuron_bias = others_neuron_b
        self.template_neuron_weight = template_neuron_w
        self.template_neuron_bias = template_neuron_b

        if global_LR:
            # one global LR
            if lrs is None:
                lrs = [torch.nn.Parameter(torch.ones(1).to(device) * lr)]
            self.lrs = lrs[0]
        else:
            if lrs is None:
                # add LRs for template (outer model)
                lrs = [torch.ones_like(p) * lr for name, p in self.named_parameters()
                       #if 'last' not in name and 'head' not in name]
                       if 'module' not in name
                       ]

                lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
                self.lrs = lrs

                # add LRs for inner model
                #lrs = [torch.ones_like(p) * lr for name, p in self.module.named_parameters()]
                #lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
                #self.module.lrs = lrs

            else:
                #lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
                self.lrs = lrs
                #lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs_inner])
                #self.module.lrs._parameters = lrs_inner

        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Descritpion**
        Akin to `MAML.clone()` but for MetaSGD: it includes a set of learnable fast-adaptation
        learning rates.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MetaSGD(clone_module(self.module),
                       lrs=clone_parameters(self.lrs),
                       lrs_inner=clone_parameters(self.module.lrs),
                       first_order=first_order,
                       allow_unused=allow_unused,
                       allow_nograd=allow_nograd,
                       global_LR=self.module.global_LR,
                       others_neuron_b=self.others_neuron_bias.clone(),
                       others_neuron_w=self.others_neuron_weight.clone(),
                       template_neuron_b=self.template_neuron_bias.clone(),
                       template_neuron_w=self.template_neuron_weight.clone())

    def adapt(self, loss, first_order=None, allow_nograd=False):
        """
        **Descritpion**
        Akin to `MAML.adapt()` but for MetaSGD: it updates the model with the learnable
        per-parameter learning rates.
        """
        if first_order is None:
            first_order = self.first_order

        second_order = not first_order
        #diff_params = [p for p in self.module.parameters() if p.requires_grad]
        diff_params = [p for name, p in self.module.named_parameters() if p.requires_grad and 'lrs' not in name]
        #diff_params = [p for name, p in self.module.named_parameters() if p.requires_grad]
        gradients = grad(loss,
                         diff_params,
                         retain_graph=second_order,
                         create_graph=second_order,
                         allow_unused=self.allow_unused)

        # for i, name in enumerate(self.module.named_parameters()):
        #     if ('lrs') not in name[0]:
        #         if 'last' in name[0]:
        #             if gradients[i].shape == 6:
        #                 print(f"GRAD {name[0]:<30} {gradients[i].item()}")
        #             else:
        #                 for j in range(gradients[i].shape[0]):
        #                     print(f"GRAD {j} {name[0]:<30} {gradients[i][j].mean().item()}")
        #
        #         else:
        #             print(f"GRAD {name[0]:<30} {gradients[i].mean().item()}")
        #     # else:
        #     #     print(f"GRAD {name:<30} None")
        #
        # exit()

        # gradients = []
        # grad_counter = 0
        #
        # # Handles gradients for non-differentiable parameters
        # for param in self.module.parameters():
        #     if param.requires_grad:
        #         gradient = grad_params[grad_counter]
        #         grad_counter += 1
        #     else:
        #         gradient = None
        #     gradients.append(gradient)

        self.module = meta_sgd_update(self.module, self.module.lrs, gradients)

    def init_last(self, ways, train=False):
        if self.module.global_LR:
            last_n = 'last_{}'.format(ways - 1)
            repeated_bias = self.predictor.bias.clone().repeat(ways - 1)
            repeated_weights = self.predictor.weight.clone().repeat(ways - 1, 1)
            for name, layer in self.module.named_modules():
                if name == last_n:
                    for param_key in layer._parameters:
                        p = layer._parameters[param_key]
                        # p.requires_grad = True
                        if param_key == 'weight':
                            p.data = repeated_weights
                        elif param_key == 'bias':
                            p.data = repeated_bias
                    break
        else:
            # LR per Parameter
            last = ways + 1  # always others neuron + ways times duplicated template neuron
            last_n = f"last_{last}"

            # repeat template neuron and LRs
            repeated_bias = self.template_neuron_bias.repeat(ways)
            #repeated_bias = torch.rand(ways).to(device)

            repeated_weights = self.template_neuron_weight.repeat(ways, 1)
            #repeated_weights = torch.rand(ways, 1024).to(device)

            repeated_weights_lr = self.lrs[2].repeat(ways, 1)
            repeated_bias_lr = self.lrs[3].repeat(ways)

            repeated_weights = torch.cat((self.others_neuron_weight, repeated_weights))
            #repeated_weights = torch.cat((torch.rand(1, 1024).to(device), repeated_weights))
            repeated_bias = torch.cat((self.others_neuron_bias, repeated_bias))
            #repeated_bias = torch.cat((torch.rand(1).to(device), repeated_bias))
            repeated_weights_lr = torch.cat((self.lrs[0], repeated_weights_lr))
            repeated_bias_lr = torch.cat((self.lrs[1], repeated_bias_lr))

            # in case of several ways find LRs belonging to Last N, first 4 entries belonging to head
            i = self.module.num_output.index(ways)
            for name, layer in self.module.named_modules():
                if name == last_n:
                    for param_key in layer._parameters:
                        if param_key == 'weight':
                            layer._parameters[param_key] = repeated_weights
                            #self.lrs.append(repeated_weights_lr)
                            self.module.lrs._parameters[f"{4 + (i * 2)}"] = repeated_weights_lr
                        elif param_key == 'bias':
                            layer._parameters[param_key] = repeated_bias
                            #self.lrs.append(repeated_bias_lr)
                            self.module.lrs._parameters[f"{5 + (i * 2)}"] = repeated_bias_lr
                    break



class MetaSGD_noTemplate(l2l.algorithms.MetaSGD):
    def __init__(self, model, lr=1.0, first_order=False, allow_unused=None, allow_nograd=False, lrs=None, global_LR=False):
        super(l2l.algorithms.MetaSGD, self).__init__()
        self.module = model
        self.module.global_LR = global_LR

        if global_LR:
            # one global LR
            if lrs is None:
                lrs = [torch.nn.Parameter(torch.ones(1).to(device) * lr)]
            self.lrs = lrs[0]
        else:
            if lrs is None:

                lrs = [torch.ones_like(p) * lr for name, p in self.named_parameters()]
                lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
                self.lrs = lrs
            else:
                self.lrs = lrs

        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Descritpion**
        Akin to `MAML.clone()` but for MetaSGD: it includes a set of learnable fast-adaptation
        learning rates.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MetaSGD_noTemplate(clone_module(self.module),
                       lrs=clone_parameters(self.lrs),
                       first_order=first_order,
                       allow_unused=allow_unused,
                       allow_nograd=allow_nograd,
                       global_LR=self.module.global_LR,
                       )

    def adapt(self, loss, first_order=None, allow_nograd=False):
        """
        **Descritpion**
        Akin to `MAML.adapt()` but for MetaSGD: it updates the model with the learnable
        per-parameter learning rates.
        """
        if first_order is None:
            first_order = self.first_order

        second_order = not first_order
        #diff_params = [p for p in self.module.parameters() if p.requires_grad]
        diff_params = [p for name, p in self.module.named_parameters() if p.requires_grad and 'lrs' not in name]
        #diff_params = [p for name, p in self.module.named_parameters() if p.requires_grad]
        gradients = grad(loss,
                         diff_params,
                         retain_graph=second_order,
                         create_graph=second_order,
                         allow_unused=self.allow_unused)


        self.module = meta_sgd_update(self.module, self.lrs, gradients)

    def init_last(self, ways, train=False):
        pass


class MetaSGD_noOthers(l2l.algorithms.MetaSGD):
    def __init__(self, model, lr=1.0, first_order=False, allow_unused=None, allow_nograd=False, lrs=None, lrs_inner=None,
                 global_LR=False, template_neuron_b=None, template_neuron_w=None):
        super(l2l.algorithms.MetaSGD, self).__init__()
        self.module = model
        self.module.global_LR = global_LR

        self.template_neuron_weight = template_neuron_w
        self.template_neuron_bias = template_neuron_b

        if global_LR:
            # one global LR
            if lrs is None:
                lrs = [torch.nn.Parameter(torch.ones(1).to(device) * lr)]
            self.lrs = lrs[0]
        else:
            if lrs is None:
                # add LRs for template (outer model)
                lrs = [torch.ones_like(p) * lr for name, p in self.named_parameters()
                       if 'module' not in name
                       ]

                lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
                self.lrs = lrs


            else:

                self.lrs = lrs


        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Descritpion**
        Akin to `MAML.clone()` but for MetaSGD: it includes a set of learnable fast-adaptation
        learning rates.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MetaSGD_noOthers(clone_module(self.module),
                       lrs=clone_parameters(self.lrs),
                       lrs_inner=clone_parameters(self.module.lrs),
                       first_order=first_order,
                       allow_unused=allow_unused,
                       allow_nograd=allow_nograd,
                       global_LR=self.module.global_LR,
                       template_neuron_b=self.template_neuron_bias.clone(),
                       template_neuron_w=self.template_neuron_weight.clone())

    def adapt(self, loss, first_order=None, allow_nograd=False):
        """
        **Descritpion**
        Akin to `MAML.adapt()` but for MetaSGD: it updates the model with the learnable
        per-parameter learning rates.
        """
        if first_order is None:
            first_order = self.first_order

        second_order = not first_order
        diff_params = [p for name, p in self.module.named_parameters() if p.requires_grad and 'lrs' not in name]
        gradients = grad(loss,
                         diff_params,
                         retain_graph=second_order,
                         create_graph=second_order,
                         allow_unused=self.allow_unused)


        self.module = meta_sgd_update(self.module, self.module.lrs, gradients)

    def init_last(self, ways, train=False):
        if self.module.global_LR:
            last_n = 'last_{}'.format(ways - 1)
            repeated_bias = self.predictor.bias.clone().repeat(ways - 1)
            repeated_weights = self.predictor.weight.clone().repeat(ways - 1, 1)
            for name, layer in self.module.named_modules():
                if name == last_n:
                    for param_key in layer._parameters:
                        p = layer._parameters[param_key]
                        # p.requires_grad = True
                        if param_key == 'weight':
                            p.data = repeated_weights
                        elif param_key == 'bias':
                            p.data = repeated_bias
                    break
        else:
            # LR per Parameter
            last = ways
            last_n = f"last_{last}"

            # repeat template neuron and LRs
            repeated_bias = self.template_neuron_bias.repeat(ways)
            repeated_weights = self.template_neuron_weight.repeat(ways, 1)
            repeated_weights_lr = self.lrs[0].repeat(ways, 1)
            repeated_bias_lr = self.lrs[1].repeat(ways)


            # in case of several ways find LRs belonging to Last N, first 4 entries belonging to head
            i = self.module.num_output.index(ways)
            for name, layer in self.module.named_modules():
                if name == last_n:
                    for param_key in layer._parameters:
                        if param_key == 'weight':
                            layer._parameters[param_key] = repeated_weights
                            self.module.lrs._parameters[f"{4 + (i * 2)}"] = repeated_weights_lr
                        elif param_key == 'bias':
                            layer._parameters[param_key] = repeated_bias
                            #self.lrs.append(repeated_bias_lr)
                            self.module.lrs._parameters[f"{5 + (i * 2)}"] = repeated_bias_lr
                    break

def meta_sgd_update(model, lrs=None, grads=None):
    """
    **Description**
    Performs a MetaSGD update on model using grads and lrs.
    The function re-routes the Python object, thus avoiding in-place
    operations.
    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.
    **Arguments**
    * **model** (Module) - The model to update.
    * **lrs** (list) - The meta-learned learning rates used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.
    **Example**
    ~~~python
    meta = l2l.algorithms.MetaSGD(Model(), lr=1.0)
    lrs = [th.ones_like(p) for p in meta.model.parameters()]
    model = meta.clone() # The next two lines essentially implement model.adapt(loss)
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    meta_sgd_update(model, lrs=lrs, grads)
    ~~~
    """

    if grads is not None and lrs is not None:
        if model.global_LR:
            lr = lrs
            for p, g in zip(model.parameters(), grads):
                p.grad = g
                p._lr = lr
        else:
            model_parameters = [p for name, p in model.named_parameters() if p.requires_grad and 'lrs' not in name]
            for p, lr, g in zip(model_parameters, lrs, grads):
                if g is not None:
                    p.grad = g
                    p._lr = lr

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - p._lr * p.grad

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None and buff._lr is not None:
            model._buffers[buffer_key] = buff - buff._lr * buff.grad

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = meta_sgd_update(model._modules[module_key])
    return model

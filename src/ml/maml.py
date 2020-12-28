import torch
import torch.nn
import learn2learn as l2l
from learn2learn.utils import clone_module

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MAML(l2l.algorithms.MAML):
    def __init__(self, model, lr, first_order=False, allow_unused=None, allow_nograd=False,
                 others_neuron_b=None, others_neuron_w=None, template_neuron_b=None, template_neuron_w=None):
        super(l2l.algorithms.MAML, self).__init__()
        self.module = model
        self.lr = lr

        self.others_neuron_weight = others_neuron_w
        self.others_neuron_bias = others_neuron_b
        self.template_neuron_weight = template_neuron_w
        self.template_neuron_bias = template_neuron_b

        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused


    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd,
                    others_neuron_b=self.others_neuron_bias.clone(),
                    others_neuron_w=self.others_neuron_weight.clone(),
                    template_neuron_b=self.template_neuron_bias.clone(),
                    template_neuron_w=self.template_neuron_weight.clone())


    def init_last(self, ways):
        last = ways + 1  # always others neuron + ways times duplicated template neuron
        last_n = f"last_{last}"

        # duplicate template neuron
        repeated_bias = self.template_neuron_bias.repeat(ways)
        repeated_weights = self.template_neuron_weight.repeat(ways, 1)

        # add others neuron
        repeated_weights = torch.cat((self.others_neuron_weight, repeated_weights))
        repeated_bias = torch.cat((self.others_neuron_bias, repeated_bias))
        for name, layer in self.module.named_modules():
            if name == last_n:
                for param_key in layer._parameters:
                    if param_key == 'weight':
                        layer._parameters[param_key] = repeated_weights
                    elif param_key == 'bias':
                        layer._parameters[param_key] = repeated_bias
                break
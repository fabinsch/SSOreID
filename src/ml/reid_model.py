import numpy as np
import torch
import torch.nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class reID_Model(torch.nn.Module):
    def __init__(self, head, predictor, n_list, lr=1e-3):
        super(reID_Model, self).__init__()
        self.head = head
        for n in n_list:
            self.add_module(f"last_{n}", torch.nn.Linear(1024, n).to(device))

        if lr > 0:
            lrs = [torch.ones_like(p) * lr for p in self.parameters()]
            lrs = [torch.normal(mean=lr, std=1e-4) for lr in lrs]
            lrs = torch.nn.ParameterList([torch.nn.Parameter(lr) for lr in lrs])
        self.lrs = lrs  # first 4 entries belonging to head, than 2 for each last N module first weight than bias
        self.num_output = n_list

    def forward(self, x, nways):
        feat = self.head(x)
        last_n = f"last_{nways}"

        # because of variable last name
        for name, layer in self.named_modules():
            if name == last_n:
                x = layer(feat)
        return x

    def remove_last(self, state_dict, num_output):
        r = dict(state_dict)
        for i, n in enumerate(num_output):
            key_weight = f"module.last_{n}.weight"
            key_bias = f"module.last_{n}.bias"
            key_lr_weight = f"module.lrs.{4 + (i * 2)}"
            key_lr_bias = f"module.lrs.{5 + (i * 2)}"
            del r[key_weight]
            del r[key_bias]
            del r[key_lr_weight]
            del r[key_lr_bias]
        return r

    def forward_pass_for_classifier_training(self, learner, features, labels, nways, return_scores=False):
        class_logits = learner(features, nways)
        loss = F.cross_entropy(class_logits, labels.long())
        if return_scores:
            pred_scores = F.softmax(class_logits, -1)
            return pred_scores.detach(), loss
        else:
            return loss

    def accuracy(self, predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        valid_accuracy_without_zero = (predictions == targets).sum().float() / targets.size(0)
        return (valid_accuracy_without_zero, torch.zeros(1), torch.zeros(1), torch.zeros(1))

    def fast_adapt(self, batch, learner, adaptation_steps, shots, ways, reid=None):
        flip_p = reid['ML']['flip_p']
        data, labels = batch
        data, labels = data, labels.to(device)
        n = 1  # consider flip in indices
        if flip_p > 0.0:
            n = 2
            # do all flipping here, put flips at the end
            data = torch.cat((data, data.flip(-1)))
            labels = labels.repeat(2)

        # Separate data into adaptation/evaluation sets
        train_indices = np.zeros(data.size(0), dtype=bool)
        train_indices[np.arange(shots * ways * n) * 2] = True  # true false true false ...
        val_indices = torch.from_numpy(~train_indices)
        train_indices = torch.from_numpy(train_indices)

        train_data, train_labels = data[train_indices], labels[train_indices]
        val_data, val_labels = data[val_indices], labels[val_indices]

        # init last layer with the template weights
        learner.init_last(ways)
        for step in range(adaptation_steps):
            train_predictions, train_loss = self.forward_pass_for_classifier_training(learner, train_data, train_labels,
                                                                                 ways, return_scores=True)
            learner.adapt(train_loss)  # Takes a gradient step on the loss and updates the cloned parameters in place

        predictions, validation_loss = self.forward_pass_for_classifier_training(learner, val_data, val_labels, ways,
                                                                            return_scores=True)
        valid_accuracy = self.accuracy(predictions, val_labels)

        return validation_loss, valid_accuracy
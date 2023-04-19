import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types


def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

def SNIP(inputs, targets, net, keep_ratio, device, minus=False, rand=False):
    inputs, targets = inputs.to(device), targets.to(device)
    net = copy.deepcopy(net)
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.nll_loss( F.log_softmax(outputs, dim=1), targets)
    print(loss.item())
    loss.backward()
    if rand:
        grads = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads.append(torch.randn(layer.weight_mask.grad.size()))
        all_weights = torch.cat([torch.flatten(x) for x in grads])
        num_params_to_keep = int(len(all_weights)*keep_ratio)
        threshold, _ = torch.topk(all_weights, num_params_to_keep, largest=True, sorted=True)
        acceptable_score = threshold[-1]
        keep_masks = []
        for g in grads:
            keep_masks.append((g >= acceptable_score).float())
        return(keep_masks)

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            #grads_abs.append(layer.weight_mask.grad) TODO: abs?
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, largest=not minus, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        if minus:
            keep_masks.append(((g / norm_factor) <= acceptable_score).float())
        else:
            keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    #num_ones = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks]))
    #num_zeros = torch.sum(torch.cat([torch.flatten(x == 0) for x in keep_masks]))
    #num_total = sum([len(x.flatten()) for x in keep_masks])
    #print(keep_ratio, num_ones / num_total, num_zeros, num_ones, num_total)

    return(keep_masks)

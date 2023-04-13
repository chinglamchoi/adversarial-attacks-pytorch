import torch
import torch.nn as nn

from ..attack import Attack
import copy
from snip import SNIP

class SPA(Attack):

    def __init__(elf, model0, device, eps=8/255, alpha=2/255, steps=10, random_start=True, eps_for_division=1e-10, keep_ratio=0.05, minus=False, rand=False):
        super().__init__("SPA", model0)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.device = device
        self.keep_ratio, self.minus, self.rand = keep_ratio, minus, rand

    def apply_prune_mask(self, net, keep_masks):
        handles = []
        prunable_layers = filter(
            lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear), net.modules())
        for layer, keep_mask in zip(prunable_layers, keep_masks):
            assert (layer.weight.shape == keep_mask.shape)
            def hook_factory(keep_mask):
                def hook(grads):
                    return grads * keep_mask
                return hook
            layer.weight.data[keep_mask == 0.] = 0.
            handles.append(layer.weight.register_hook(hook_factory(keep_mask)))
        return handles

    def remove_handles(self, handles):
        for handle in handles:
            handle.remove()

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        keep_masks = SNIP(adv_images, labels, self.model0, self.keep_ratio, self.device, minus=self.minus, rand=self.rand)
        self.model = copy.deepcopy(self.model0).to(self.device)
        handles = self.apply_prune_mask(self.model, keep_masks)

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        self.remove_handles(handles)
        del handles
        del keep_masks
        del self.model

        return adv_images

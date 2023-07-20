import torch
import torch.nn as nn
from torch import Tensor
import torch.autograd as autograd


class AdversarialLoss(nn.Module):
    def __init__(self, D: nn.Module, lambda_gp: float = 10.):
        super().__init__()
        self.D = D
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, realX: Tensor, fakeX: Tensor):
        alpha = torch.rand(1, device=realX.device)
        interX = alpha * realX + (1 - alpha) * fakeX
        interX.requires_grad_()
        d_interX = self.D(interX)
        gradients = autograd.grad(outputs=d_interX, inputs=interX,
                                  grad_outputs=torch.ones_like(d_interX),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.flatten(start_dim=1) + 1e-10
        return torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

    def forward_D(self, realX: Tensor, fakeX: Tensor):
        """ min E[D(G(z))] - E[D(x)] + lambda * gp """
        d_fake = self.D(fakeX)
        d_real = self.D(realX)
        lossD = torch.mean(d_fake) - torch.mean(d_real)
        lossD = lossD + self.lambda_gp * self.gradient_penalty(realX, fakeX)
        return lossD

    def forward_G(self, fakeX: Tensor):
        """ max E[D(G(z))] <=> min E[-D(G(z))] """
        d_fake = self.D(fakeX)
        lossG = -torch.mean(d_fake)
        return lossG

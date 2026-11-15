"""Briques communes aux deux bancs : lois, réseau MLP, distances, mesures."""
import math

import torch
from torch import nn


class MLP(nn.Module):
    """2 -> largeur -> largeur -> largeur -> sortie, activations SiLU.

    Même forme que le `mlp_2d` du dépôt Generative_Wasserstein_Flows.
    Sert de générateur (sortie 2) et de critique (sortie 1, ou 128 pour la MMD apprise).
    """
    def __init__(self, sortie, largeur=128):
        super().__init__()
        dimensions = (2, largeur, largeur, largeur, sortie)
        couches = []
        for i, (a, b) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            couches.append(nn.Linear(a, b))
            if i < 3:
                couches.append(nn.SiLU())
        self.network = nn.Sequential(*couches)

    def forward(self, x):
        return self.network(x)


def rng(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def source(n, tirages, device):
    """Bruit d'entrée mu_0 = N(0, I/3) (même covariance que l'uniforme sur [-1,1]²)."""
    return torch.randn(n, 2, generator=tirages, device=device) / math.sqrt(3)


def cible(n, tirages, device, nom):
    """Huit gaussiennes (rayon 1, écart-type 0,1) ou deux lunes (bruit 0,06)."""
    if nom == "huit_gaussiennes":
        a = torch.arange(8, device=device) * math.pi / 4
        centres = torch.stack((a.cos(), a.sin()), 1)
        groupes = torch.randint(8, (n,), generator=tirages, device=device)
        return centres[groupes] + .1 * torch.randn(n, 2, generator=tirages, device=device)
    groupes = torch.randint(2, (n,), generator=tirages, device=device)
    a = math.pi * torch.rand(n, generator=tirages, device=device)
    x = torch.where(groupes == 0, .8 * a.cos() - .4, .4 - .8 * a.cos())
    y = torch.where(groupes == 0, .8 * a.sin() - .2, .2 - .8 * a.sin())
    return torch.stack((x, y), 1) + .06 * torch.randn(n, 2, generator=tirages, device=device)


def distances_carrees(x, y):
    return (x.square().sum(1, keepdim=True) + y.square().sum(1)[None]
            - 2 * x @ y.T).clamp_min(0)


def demi_mmd2(x, y):
    """(1/2) MMD² avec le noyau de Riesz -||x-y|| (distance d'énergie), U-statistique."""
    xx = (distances_carrees(x, x) + 1e-6).sqrt()
    yy = (distances_carrees(y, y) + 1e-6).sqrt()
    xy = (distances_carrees(x, y) + 1e-6).sqrt()
    nx, ny = len(x), len(y)
    propre_x = (xx.sum() - xx.diagonal().sum()) / (nx * (nx - 1))
    propre_y = (yy.sum() - yy.diagonal().sum()) / (ny * (ny - 1))
    return xy.mean() - .5 * propre_x - .5 * propre_y


@torch.no_grad()
def sw2(x, y):
    """Wasserstein-2 projeté sur 128 directions (x et y de même taille)."""
    a = torch.arange(128) * math.pi / 128
    directions = torch.stack((a.cos(), a.sin()), 1)
    return ((x @ directions.T).sort(0).values
            - (y @ directions.T).sort(0).values).square().mean().sqrt().item()


@torch.no_grad()
def mesures_huit_gaussiennes(points):
    """Proches : part à moins de 0,2 d'un centre (cible ~86,5 %).
    Largeur : écart-type autour du centre le plus proche (cible ~0,1).
    Poids : écart en variation totale aux poids 1/8 des amas (cible 0)."""
    a = torch.arange(8) * math.pi / 4
    centres = torch.stack((a.cos(), a.sin()), 1)
    d, groupes = distances_carrees(points, centres).min(1)
    poids = torch.bincount(groupes, minlength=8).float() / len(points)
    return dict(proches=(d <= .2 ** 2).float().mean().item(),
                largeur=(d.mean() / 2).sqrt().item(),
                erreur_poids=.5 * (poids - .125).abs().sum().item())

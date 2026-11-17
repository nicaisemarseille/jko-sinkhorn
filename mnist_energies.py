"""Nos énergies 2D, portées telles quelles sur les images.

Mêmes formules que experiences/sinkhorn.py et experiences/banc_simple.py.
Seule adaptation : les images sont aplaties puis divisées par sqrt(d), pour que
le coût c(x, y) = ||x - y||² / 2 soit celui de la métrique du terme JKO du dépôt
(transport_cost = moyenne sur les d coordonnées).

Divergences ajoutées à train.py :
  OT_eps           un critique scalaire g ; semi-dual entropique (c-transformée exacte sur le lot)
  Sinkhorn_appris  deux critiques scalaires g (sur les vraies) et s (sur les générées)
  Sinkhorn_direct  aucun critique ; S_eps calculé par Sinkhorn sur chaque lot
  MMD_fixe         aucun critique ; (1/2) MMD² de Riesz (algorithme 1 du papier)
KL_DV, chi2 et MMD (MMD-GAN à noyau appris) existent déjà dans le dépôt.
"""
import math

import torch
from torch import nn

AVEC_CRITIQUE = ("OT_eps", "Sinkhorn_appris")
SANS_CRITIQUE = ("Sinkhorn_direct", "MMD_fixe")
NOS_DIVERGENCES = AVEC_CRITIQUE + SANS_CRITIQUE

# Pire résidu de Sinkhorn direct depuis le dernier affichage (diagnostic, voir train.py).
SUIVI = dict(residu_max=0.0, non_converge=0)


def aplatir(x):
    x = x.flatten(1)
    return x / math.sqrt(x.shape[1])


def cout(x, y):
    return .5 * (x.square().sum(1, keepdim=True) + y.square().sum(1)[None]
                 - 2 * x @ y.T).clamp_min(0)


def logmoyenne(a, dim):
    return torch.logsumexp(a, dim=dim) - math.log(a.shape[dim])


def variation(x, y, g, s, epsilon):
    """s = None : semi-dual de OT_eps(mu, nu). Sinon : S_eps(mu, nu) + OT_eps(nu,nu)/2."""
    g = g.flatten() - g.mean()
    a = logmoyenne((g[None] - cout(x, y)) / epsilon, 1)
    valeur = -epsilon * a.mean() + g.mean()
    if s is not None:
        s = s.flatten() - s.mean()
        cxx = cout(x, x)
        b = logmoyenne((s[None] - cxx) / epsilon, 1)
        q = torch.logsumexp(((s[:, None] + s[None] - cxx) / epsilon).flatten(), 0)
        q = q - 2 * math.log(len(x))
        valeur = valeur + epsilon * (b.mean() - .5 * q)
    return valeur


@torch.no_grad()
def potentiels_exacts(c, epsilon, tolerance, maximum):
    """Sinkhorn log-domaine. Contrairement au 2D, un non-convergé n'arrête pas la nuit :
    on garde les derniers potentiels et on le compte dans SUIVI."""
    g = torch.zeros(c.shape[1], dtype=c.dtype, device=c.device)
    erreur = float("inf")
    for iteration in range(1, maximum + 1):
        f = -epsilon * logmoyenne((g[None] - c) / epsilon, 1)
        g = -epsilon * logmoyenne((f[:, None] - c) / epsilon, 0)
        if iteration % 20 == 0 or iteration == maximum:
            log_lignes = f / epsilon + logmoyenne((g[None] - c) / epsilon, 1)
            log_colonnes = g / epsilon + logmoyenne((f[:, None] - c) / epsilon, 0)
            erreur = max(log_lignes.expm1().abs().max().item(),
                         log_colonnes.expm1().abs().max().item())
            if erreur <= tolerance:
                break
    SUIVI["residu_max"] = max(SUIVI["residu_max"], erreur)
    if erreur > tolerance:
        SUIVI["non_converge"] += 1
    return f, g


def transport_direct(x, y, epsilon, tolerance, maximum):
    c = cout(x, y)
    f, g = potentiels_exacts(c.detach(), epsilon, tolerance, maximum)
    return f.mean() + g.mean() - epsilon * (
        torch.exp((f[:, None] + g[None] - c) / epsilon).mean() - 1)


def demi_mmd2_riesz(x, y):
    def d(a, b):
        return (2 * cout(a, b) + 1e-6).sqrt()
    xx, yy, xy = d(x, x), d(y, y), d(x, y)
    nx, ny = len(x), len(y)
    return (xy.mean() - .5 * (xx.sum() - xx.diagonal().sum()) / (nx * (nx - 1))
            - .5 * (yy.sum() - yy.diagonal().sum()) / (ny * (ny - 1)))


def energie(args, fake_images, real_images, d_real, d_fake):
    """Valeur variationnelle, maximisée par les critiques et minimisée par le générateur."""
    x, y = aplatir(fake_images), aplatir(real_images)
    eps = args.eps_sinkhorn
    if args.divergence == "OT_eps":
        return variation(x, y, d_real[:, 0], None, eps)
    if args.divergence == "Sinkhorn_appris":
        return variation(x, y, d_real[:, 0], d_fake[:, 1], eps)
    if args.divergence == "Sinkhorn_direct":
        tol, nmax = args.tol_sinkhorn, args.iter_sinkhorn
        return (transport_direct(x, y.detach(), eps, tol, nmax)
                - .5 * transport_direct(x, x, eps, tol, nmax))
    if args.divergence == "MMD_fixe":
        return demi_mmd2_riesz(x, y.detach())
    raise ValueError(args.divergence)


class DeuxPotentiels(nn.Module):
    """Deux critiques séparés de même architecture : colonne 0 = g, colonne 1 = s."""
    def __init__(self, g, s):
        super().__init__()
        self.g, self.s = g, s

    def forward(self, x):
        return torch.cat((self.g(x).flatten(1)[:, :1], self.s(x).flatten(1)[:, :1]), 1)

"""Banc simple : JKO paramétré avec cinq énergies, sur deux cibles 2D.

    python banc_simple.py                      # les deux cibles
    python banc_simple.py --cible lunes

Résultat : resultats/banc_simple.pt (points générés), puis figures/banc_simple.png.
Durée : environ 20 minutes par cible sur GPU.

Adapté de la commande 2D du dépôt Generative_Wasserstein_Flows (commit 6633b55) :
même MLP, même pénalité R1 (0,2), même Adam, un pas critique pour un pas générateur.
Changements : source N(0, I/3), 50 pas JKO, noyau de Riesz, ajout de OT epsilon.
Notre énergie MMD est (1/2)MMD² ; la perte MMD du dépôt emploie MMD².
"""
import argparse
import copy
import math
import time
from pathlib import Path

import torch

from commun import MLP, cible, demi_mmd2, distances_carrees, rng, source, sw2

METHODES = ("kl_dv", "chi2", "ot_eps", "mmd_fixed", "mmd_learned")
PARAMETRES = dict(seed=2026, etapes=50, iterations=500, batch=256,
                  largeur=128, tau=1., epsilon=.05, r1=.2,
                  dimension_mmd=128, lr_generateur=2e-4, lr_critique=1e-4,
                  beta1=.5, beta2=.9, eps_adam=1e-8, weight_decay=0.)
RESULTATS = Path(__file__).resolve().parent / "resultats"


def energie(methode, x, y, critique, epsilon):
    """Valeur du problème variationnel (à maximiser en le critique)."""
    if methode == "mmd_fixed":
        return demi_mmd2(x, y), None
    dy = critique(y)
    if methode == "ot_eps":
        # Semi-dual entropique : sup_g  E_mu[g^{c,eps}] + E_nu[g].
        g = dy.flatten()
        g = g - g.mean()
        cout = .5 * distances_carrees(x, y)
        transforme = -epsilon * (torch.logsumexp((g[None] - cout) / epsilon, 1)
                                 - math.log(len(y)))
        return transforme.mean() + g.mean(), dy
    dx = critique(x)
    if methode == "kl_dv":
        # Donsker-Varadhan.
        valeur = -dx.mean() - torch.logsumexp(-dy.flatten(), 0) + math.log(len(y))
    elif methode == "chi2":
        valeur = -dx.mean() - .25 * dy.square().mean() + dy.mean()
    else:
        # MMD-GAN : (1/2)MMD² de Riesz entre les plongements psi(x) et psi(y).
        valeur = demi_mmd2(dx, dy)
    return valeur, dy


def penalite(dy, y):
    """R1 sur les vraies données. En sortie vectorielle (MMD apprise),
    des composantes opposées peuvent s'annuler : pénalité imparfaite."""
    gradient, = torch.autograd.grad(dy.sum(), y, create_graph=True)
    return gradient.square().sum(1).mean()


def entrainer(methode, nom_cible, p, device, afficher=True):
    torch.manual_seed(p["seed"])
    generateur = MLP(2, p["largeur"]).to(device)
    torch.manual_seed(p["seed"] + 1)
    critique = None if methode == "mmd_fixed" else MLP(
        p["dimension_mmd"] if methode == "mmd_learned" else 1, p["largeur"]).to(device)
    adam = dict(betas=(p["beta1"], p["beta2"]), eps=p["eps_adam"], weight_decay=p["weight_decay"])
    optim_t = torch.optim.Adam(generateur.parameters(), lr=p["lr_generateur"], **adam)
    optim_g = None if critique is None else torch.optim.Adam(
        critique.parameters(), lr=p["lr_critique"], **adam)
    # Flux distincts : les tirages du générateur restent appariés entre méthodes.
    zd, yd, zt, yt = [rng(device, p["seed"] + j) for j in (10, 11, 12, 13)]
    precedent = None  # Au premier pas JKO, la référence est l'identité.
    debut = time.perf_counter()
    for k in range(p["etapes"]):
        for _ in range(p["iterations"]):
            if critique is not None:
                # 1. Un pas de montée du critique, générateur figé.
                critique.requires_grad_(True)
                generateur.requires_grad_(False)
                z = source(p["batch"], zd, device)
                y = cible(p["batch"], yd, device, nom_cible).requires_grad_(True)
                with torch.no_grad():
                    x = generateur(z)
                valeur, dy = energie(methode, x, y, critique, p["epsilon"])
                perte_g = -valeur + p["r1"] * penalite(dy, y)
                optim_g.zero_grad(set_to_none=True)
                perte_g.backward()
                optim_g.step()
                critique.requires_grad_(False)
            # 2. Un pas de descente du générateur, critique figé, nouveaux tirages.
            generateur.requires_grad_(True)
            z = source(p["batch"], zt, device)
            y = cible(p["batch"], yt, device, nom_cible)
            x = generateur(z)
            with torch.no_grad():
                reference = z if precedent is None else precedent(z)
            if methode in ("kl_dv", "chi2"):
                valeur = -critique(x).mean()  # Seuls termes dépendant du générateur.
            else:
                valeur, _ = energie(methode, x, y, critique, p["epsilon"])
            # En 2D, moyenne sur points ET coordonnées = E||x - référence||² / 2.
            perte_t = valeur + (x - reference).square().mean() / p["tau"]
            optim_t.zero_grad(set_to_none=True)
            perte_t.backward()
            optim_t.step()
        precedent = copy.deepcopy(generateur).requires_grad_(False).eval()
        if afficher:
            print(f"{nom_cible} | {methode} | JKO {k + 1}/{p['etapes']}", flush=True)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return generateur.eval(), time.perf_counter() - debut


def calculer(nom_cible, device, p=PARAMETRES):
    y = cible(8192, rng("cpu", 21038), "cpu", nom_cible)
    z = source(8192, rng("cpu", 21037), "cpu")
    sortie = dict(cible=y, points={}, sw2={}, secondes={})
    for methode in METHODES:
        modele, secondes = entrainer(methode, nom_cible, p, device)
        with torch.no_grad():
            x = modele.cpu()(z)
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"Points non finis : {methode}")
        sortie["points"][methode] = x
        sortie["sw2"][methode] = sw2(x, y)
        sortie["secondes"][methode] = secondes
        print(f"{methode} : {secondes / 60:.1f} min, SW2 = {sortie['sw2'][methode]:.4f}", flush=True)
    return sortie


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cible", choices=("huit_gaussiennes", "lunes", "les_deux"), default="les_deux")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chemin = RESULTATS / "banc_simple.pt"
    donnees = torch.load(chemin, weights_only=True) if chemin.exists() else dict(parametres=PARAMETRES, cibles={})
    noms = ("huit_gaussiennes", "lunes") if args.cible == "les_deux" else (args.cible,)
    for nom in noms:
        donnees["cibles"][nom] = calculer(nom, device)
        RESULTATS.mkdir(exist_ok=True)
        torch.save(donnees, chemin)
    import figures
    figures.banc_simple()

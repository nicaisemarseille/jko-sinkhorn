"""Redessine les figures finales à partir de resultats/*.pt, sans réentraîner.

    python figures.py

Écrit figures/banc_simple.png et figures/sinkhorn_<cible>.png,
et affiche un tableau de mesures dans le terminal.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from commun import cible, mesures_huit_gaussiennes, rng

sys.stdout.reconfigure(encoding="utf-8")  # Console Windows : afficher χ², ε.
DOSSIER = Path(__file__).resolve().parent
RESULTATS, FIGURES = DOSSIER / "resultats", DOSSIER / "figures"
CIBLES = {"huit_gaussiennes": "Huit gaussiennes", "lunes": "Deux lunes"}
TITRES_SIMPLE = {"kl_dv": "KL (Donsker-Varadhan)", "chi2": "χ²", "ot_eps": "OT ε",
                 "mmd_fixed": "MMD fixe (alg. 1)", "mmd_learned": "MMD-GAN (noyau appris)"}
ENERGIES = {"ot": "OT ε", "sinkhorn": "Sinkhorn appris", "direct": "Sinkhorn direct"}
GENERATEURS = {"mlp": "MLP", "residuel": "Résiduel", "splines": "Splines"}


def nuage(ax, points, titre=None):
    ax.set(xlim=(-1.65, 1.65), ylim=(-1.65, 1.65), xticks=[-1, 0, 1], yticks=[-1, 0, 1])
    ax.set_aspect("equal")
    if titre:
        ax.set_title(titre, fontsize=12)
    if points is None:
        ax.text(.5, .5, "non calculé", transform=ax.transAxes, ha="center", color="gray")
        return
    points = points[:2048]
    ax.scatter(points[:, 0], points[:, 1], s=2, alpha=.4, color="#2563eb")
    dehors = (points.abs() > 1.65).any(1).float().mean().item()
    if dehors > .01:
        ax.text(.02, .02, f"{dehors:.0%} hors cadre", transform=ax.transAxes, fontsize=9)


def ligne_mesures(nom, nom_cible, d):
    texte = f"  {nom:<34} SW2 {d['sw2']:.4f}"
    if nom_cible == "huit_gaussiennes":
        m = mesures_huit_gaussiennes(d["points"])
        texte += f" | proches {m['proches']:6.1%} | largeur {m['largeur']:.3f} | poids {m['erreur_poids']:.3f}"
    if "cosinus_gradient" in d:
        texte += f" | cosinus {d['cosinus_gradient']:+.2f}"
    return texte


def banc_simple():
    """Deux lignes (cibles) x six colonnes (cible + cinq énergies)."""
    donnees = torch.load(RESULTATS / "banc_simple.pt", weights_only=True)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.6), constrained_layout=True)
    print("Banc simple (huit gaussiennes : proches ~86,5 %, largeur ~0,100, poids 0)")
    for ligne, (nom, titre_cible) in enumerate(CIBLES.items()):
        d = donnees["cibles"].get(nom)
        nuage(axes[ligne, 0], cible(2048, rng("cpu", 21038), "cpu", nom), "Cible")
        axes[ligne, 0].set_ylabel(titre_cible, fontsize=13)
        print(f" {titre_cible}")
        for colonne, (methode, titre) in enumerate(TITRES_SIMPLE.items(), 1):
            points = None if d is None else d["points"][methode]
            nuage(axes[ligne, colonne], points, titre)
            if points is not None:
                print(ligne_mesures(titre.replace("\n", " "), nom,
                                    dict(points=points, sw2=d["sw2"][methode])))
    fig.suptitle("Banc simple : 50 pas JKO × 500 itérations, générateur et critiques MLP", fontsize=15)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "banc_simple.png", dpi=150)
    plt.close(fig)


def sinkhorn():
    """Une figure par cible : lignes = énergies, colonnes = cible + trois générateurs."""
    donnees = torch.load(RESULTATS / "sinkhorn.pt", weights_only=True)["conditions"]
    print("Banc Sinkhorn")
    for nom, titre_cible in CIBLES.items():
        fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.6), constrained_layout=True)
        y = cible(2048, rng("cpu", 21038), "cpu", nom)
        print(f" {titre_cible}")
        for ligne, (energie, titre_energie) in enumerate(ENERGIES.items()):
            nuage(axes[ligne, 0], y, "Cible" if ligne == 0 else None)
            axes[ligne, 0].set_ylabel(titre_energie, fontsize=13)
            for colonne, (gen, titre_gen) in enumerate(GENERATEURS.items(), 1):
                d = donnees.get(f"{nom}/{energie}/{gen}")
                nuage(axes[ligne, colonne], None if d is None else d["points"],
                      titre_gen if ligne == 0 else None)
                if d is not None:
                    print(ligne_mesures(f"{titre_energie} / {titre_gen}", nom, d))
        fig.suptitle(f"{titre_cible} : énergies de Sinkhorn × générateurs (20 pas JKO × 500 itérations)",
                     fontsize=14)
        FIGURES.mkdir(exist_ok=True)
        fig.savefig(FIGURES / f"sinkhorn_{nom}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    banc_simple()
    sinkhorn()

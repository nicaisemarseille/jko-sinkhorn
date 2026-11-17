"""Redessine les figures finales à partir de resultats/*.pt, sans réentraîner.

    python figures.py

Écrit figures/banc_simple.png, figures/sinkhorn_<cible>.png (2D) et figures/mnist.png
(à partir de resultats/mnist.json et mnist_echantillons.npz, produits par mnist_kaggle.py resumer),
et affiche un tableau de mesures dans le terminal.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import json

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


# MNIST : deux blocs de lignes (énergie) x colonnes (avec JKO, sans JKO, FID au fil des pas).
MNIST = {"Méthodes du papier (critique réseau, sauf MMD fixe)": {
             "kl_dv": "KL (Donsker-Varadhan)", "chi2": "χ²", "mmd_gan": "MMD-GAN (noyau appris)",
             "w1": "W1 (WGAN-GP)", "mmd_fixe": "MMD fixe (alg. 1)"},
         "Nos énergies de transport (coût ½‖x−y‖² sur les pixels)": {
             "ot_eps0.01": "OT ε=0,01", "ot_eps0.05": "OT ε=0,05",
             "sinkhorn_appris_eps0.01": "Sinkhorn appris ε=0,01", "sinkhorn_appris_eps0.05": "Sinkhorn appris ε=0,05",
             "sinkhorn_direct_eps0.01": "Sinkhorn direct ε=0,01", "sinkhorn_direct_eps0.05": "Sinkhorn direct ε=0,05"}}


def mnist():
    resume = json.loads((RESULTATS / "mnist.json").read_text(encoding="utf-8"))
    images = np.load(RESULTATS / "mnist_echantillons.npz")
    lignes = max(len(b) for b in MNIST.values())
    fig = plt.figure(figsize=(19, 2.75 * lignes + 1), constrained_layout=True)
    print("MNIST (FID finale avec JKO | sans JKO, durée sur un T4)")
    for sous_fig, (titre_bloc, bloc) in zip(fig.subfigures(1, 2, wspace=.04), MNIST.items()):
        axes = sous_fig.subplots(lignes, 3, gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
        sous_fig.suptitle(titre_bloc, fontsize=14)
        for i, (nom, titre) in enumerate(bloc.items()):
            texte = f"  {titre:<26}"
            for j, (jko, entete) in enumerate((("jko", "avec JKO"), ("sans_jko", "sans JKO"))):
                ax, r = axes[i, j], resume.get(f"{nom}/{jko}")
                ax.set(xticks=[], yticks=[])
                if r is None:
                    ax.text(.5, .5, "pas lancé", transform=ax.transAxes, ha="center", color="gray")
                    texte += "            -"
                else:
                    ax.imshow(images[f"{nom}/{jko}"], cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"FID {r['fid'][-1]:.1f}", fontsize=11, loc="right")
                    axes[i, 2].plot(r["fid_pas_jko"], r["fid"], "o-", ms=3, label=entete,
                                    color="#2563eb" if jko == "jko" else "#ea580c")
                    texte += f" {r['fid'][-1]:6.1f} ({r['heures']:.1f} h)"
                if i == 0:
                    ax.set_title(entete, fontsize=12, loc="left")
            axes[i, 0].set_ylabel(titre, fontsize=11)
            c = axes[i, 2]
            c.set(yscale="log", ylim=(1, 400), xlim=(0, 41))
            c.grid(alpha=.3)
            c.tick_params(labelsize=8)
            if i == 0:
                c.set_title("FID au fil des pas JKO", fontsize=12)
                c.legend(handles=[plt.Line2D([], [], color="#2563eb", marker="o", ms=3, label="avec JKO"),
                                  plt.Line2D([], [], color="#ea580c", marker="o", ms=3, label="sans JKO")],
                         fontsize=8, loc="upper right")
            if i == len(bloc) - 1:
                c.set_xlabel("pas (500 itérations chacun)", fontsize=9)
            print(texte)
        for i in range(len(bloc), lignes):
            for ax in axes[i]:
                ax.axis("off")
    fig.suptitle("MNIST : 64 échantillons finaux par run (même bruit z partout), 40 pas × 500 itérations, "
                 "une graine ; FID sur 50 000 échantillons", fontsize=15)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "mnist.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    banc_simple()
    sinkhorn()
    mnist()

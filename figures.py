"""Redessine toutes les figures à partir de resultats/, sans réentraîner.

    python figures.py            # ou : python figures.py 2d / python figures.py mnist

2D : figures/banc_simple.png, figures/sinkhorn_<cible>.png (depuis banc_simple.pt, sinkhorn.pt).
MNIST : figures/mnist_echantillons.png, mnist_mesures.png, mnist_variantes.png (depuis mnist.json
et mnist_echantillons.npz, écrits par mnist_kaggle.py resumer).
Affiche aussi les mesures dans le terminal.
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
TITRES_SIMPLE = {"kl_dv": "KL", "chi2": "χ²", "ot_eps": "OT ε",
                 "mmd_fixed": "MMD, noyau fixe", "mmd_learned": "MMD-GAN"}
ENERGIES = {"ot": "OT ε", "sinkhorn": "Sinkhorn,\nforme variationnelle", "direct": "Sinkhorn"}
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
                    print(ligne_mesures(f"{titre_energie.replace(chr(10), ' ')} / {titre_gen}", nom, d))
        fig.suptitle(f"{titre_cible} : énergies de Sinkhorn × générateurs (20 pas JKO × 500 itérations)",
                     fontsize=14)
        FIGURES.mkdir(exist_ok=True)
        fig.savefig(FIGURES / f"sinkhorn_{nom}.png", dpi=150)
        plt.close(fig)


# ----- MNIST -----
# (titre, run avec JKO, run sans JKO) ; None : pas lancé. Runs décrits dans mnist_kaggle.py.
METHODES = [
    ("MMD-GAN", "mmd_gan_jko", "mmd_gan_sans_jko"),
    ("KL", "kl_jko", "kl_sans_jko"),
    ("WGAN", "wgan_jko", "wgan_sans_jko"),
    ("WGAN + Dukler", "wgan_dukler_jko", "wgan_dukler_sans_jko"),
    ("Sinkhorn, pixels", "sinkhorn_pixels_jko", "sinkhorn_pixels_sans_jko"),
    ("Sinkhorn, encodeur fixe\n(style Drifting)", "sinkhorn_encodeur_fixe_jko", "sinkhorn_encodeur_fixe_sans_jko"),
    ("Sinkhorn, encodeur\nadversarial", "sinkhorn_encodeur_adversarial_jko", "sinkhorn_encodeur_adversarial_sans_jko"),
]
A, P = "sinkhorn_encodeur_adversarial", "sinkhorn_pixels"
VARIANTES = [
    ("Sinkhorn, encodeur adversarial", [
        ("ε = 0,05", f"{A}_jko", f"{A}_sans_jko"),
        ("ε = 0,05, graine 2", f"{A}_graine2_jko", f"{A}_graine2_sans_jko"),
        ("ε = 0,05", f"{A}_tau0.1_jko", None),
        ("ε = 0,05", f"{A}_tau5_jko", None),
        ("ε = 0,01", f"{A}_eps0.01_jko", None),
        ("ε = 0,2", f"{A}_eps0.2_jko", None),
        ("ε = 0,05, pénalité de Dukler", f"{A}_dukler_jko", f"{A}_dukler_sans_jko")]),
    ("Sinkhorn, pixels", [
        ("ε = 0,05", f"{P}_jko", f"{P}_sans_jko"),
        ("ε = 0,01", f"{P}_eps0.01_jko", f"{P}_eps0.01_sans_jko"),
        ("ε = 0,05", f"{P}_tau5_jko", None),
        ("annealing ε : 0,05 → 0,005", f"{P}_annealing_0.05_0.005_jko", f"{P}_annealing_0.05_0.005_sans_jko"),
        ("annealing ε : 0,1 → 0,01", f"{P}_annealing_0.1_0.01_jko", f"{P}_annealing_0.1_0.01_sans_jko"),
        ("ε = 0,05, 400 pas de 50 itérations", f"{P}_400pas_jko", None),
        ("ε = 0,05, 400 pas de 50 itérations", f"{P}_400pas_tau5_jko", None),
        ("ε = 0,05, batch 64", None, f"{P}_batch64_sans_jko")]),
    ("Sinkhorn, autres coûts fixes (ε = 0,05)", [
        ("encodeur fixe (style Drifting)", "sinkhorn_encodeur_fixe_jko", "sinkhorn_encodeur_fixe_sans_jko"),
        ("encodeur fixe, amorcé", "sinkhorn_encodeur_fixe_amorce_jko", None),
        ("encodeur du MMD-GAN, figé", "sinkhorn_encodeur_mmd_fige_jko", "sinkhorn_encodeur_mmd_fige_sans_jko"),
        ("classifieur figé", "sinkhorn_classifieur_fige_jko", "sinkhorn_classifieur_fige_sans_jko"),
        ("classifieur figé, amorcé", "sinkhorn_classifieur_fige_amorce_jko", None),
        ("H⁻¹", "sinkhorn_hm1_jko", "sinkhorn_hm1_sans_jko"),
        ("Dukler (Wasserstein of Wasserstein), batch 64", "sinkhorn_dukler_jko", "sinkhorn_dukler_sans_jko")]),
]
# (clé, titre, bornes de l'axe log) ; None : axe linéaire de 0 à 1.
MESURES = [("fid_10k", "FID", (1, 400)), ("sw_pixels", "sliced Wasserstein, pixels", (5e-4, 6e-2)),
           ("sw_inception", "sliced Wasserstein, Inception", (5e-3, .6)),
           ("precision", "précision", None), ("rappel", "rappel", None)]


def virgule(x, chiffres=None):
    return (f"{x:g}" if chiffres is None else f"{x:.{chiffres}f}").replace(".", ",")


def mosaique(images, n=6):
    t = np.full((n * 29 + 1, n * 29 + 1), 255.)
    for i, im in enumerate(images[:n * n]):
        a, b = divmod(i, n)
        t[1 + 29 * a:29 + 29 * a, 1 + 29 * b:29 + 29 * b] = im
    return t


def legende(r):
    return f"avec JKO, τ = {virgule(r['tau'])}" if r["jko"] else "sans JKO"


def nom_ligne(titre, r):
    """Le titre, plus τ s'il n'est pas 0,5, et le nombre de pas si le run a été arrêté avant la fin."""
    if r["jko"] and r["tau"] != .5:
        titre += f", τ = {virgule(r['tau'])}"
    if r["pas_jko"] < r["pas_jko_prevus"]:
        titre += f" ({r['pas_jko']} pas sur {r['pas_jko_prevus']})"
    return titre


def barres(res, groupes, fichier, legende_jko):
    """Une colonne par mesure, une ligne par condition, deux barres (avec / sans JKO)."""
    ref = json.loads((RESULTATS / "mnist_reference.json").read_text(encoding="utf-8"))
    lignes = []                          # (étiquette, run avec JKO, run sans JKO, en-tête ?)
    for titre_groupe, g in groupes:
        if titre_groupe:
            lignes.append((titre_groupe, None, None, True))
        lignes += [(nom_ligne(t, res[a or s]), a, s, False) for t, a, s in g]
    fig, axes = plt.subplots(1, len(MESURES), figsize=(3.8 * len(MESURES), .36 * len(lignes) + 1.4),
                             sharey=True, constrained_layout=True)
    y = np.arange(len(lignes))[::-1]
    for ax, (cle, titre, bornes) in zip(axes, MESURES):
        for dec, idx, couleur, etiquette in ((.2, 1, "C0", legende_jko), (-.2, 2, "C1", "sans JKO")):
            for yy, l in zip(y + dec, lignes):
                if l[idx]:
                    x = res[l[idx]][cle]
                    ax.barh(yy, x, height=.38, color=couleur, label=etiquette)
                    chiffres = 1 if cle == "fid_10k" else 4 if cle.startswith("sw") else 2
                    ax.text(x * 1.08 if bornes else x + .01, yy, virgule(x, chiffres), va="center", fontsize=7)
        ax.axvline(ref[cle], color="k", ls="--", lw=1, label="vraies images")
        ax.set(title=titre, xscale="log" if bornes else "linear", xlim=bornes or (0, 1))
        ax.grid(axis="x", alpha=.3)
    axes[0].set_yticks(y, [l[0] for l in lignes])
    for tick, l in zip(axes[0].get_yticklabels(), lignes):
        if l[3]:
            tick.set_fontweight("bold")
    poignees = dict(zip(*axes[0].get_legend_handles_labels()[::-1]))   # une entrée par étiquette
    fig.legend(poignees.values(), poignees.keys(), loc="outside lower center", ncol=3)
    fig.savefig(FIGURES / fichier, dpi=110, bbox_inches="tight")
    plt.close(fig)


def mnist():
    """mnist_echantillons.png et mnist_mesures.png (les sept méthodes), mnist_variantes.png (le reste).
    Mesures sur 10 000 images générées contre les 10 000 images de test de MNIST."""
    res = json.loads((RESULTATS / "mnist.json").read_text(encoding="utf-8"))
    images = np.load(RESULTATS / "mnist_echantillons.npz")
    FIGURES.mkdir(exist_ok=True)

    # 1. Échantillons : une colonne par méthode, avec JKO en haut, sans JKO en bas.
    fig, axes = plt.subplots(2, len(METHODES), figsize=(2.6 * len(METHODES), 6.4), constrained_layout=True)
    for j, (t, a, s) in enumerate(METHODES):
        for i, nom in enumerate((a, s)):
            axes[i, j].imshow(mosaique(images[nom]), cmap="gray", vmin=0, vmax=255)
            axes[i, j].axis("off")
            axes[i, j].set_title(f"{t}\n{legende(res[nom])} — FID {virgule(res[nom]['fid_10k'], 1)}", fontsize=10)
    fig.savefig(FIGURES / "mnist_echantillons.png", dpi=90, bbox_inches="tight")
    plt.close(fig)

    # 2. et 3. Mesures.
    barres(res, [(None, [(t.replace("\n", " "), a, s) for t, a, s in METHODES])], "mnist_mesures.png",
           "avec JKO, τ = 0,5")
    barres(res, VARIANTES, "mnist_variantes.png", "avec JKO, τ = 0,5 sauf mention")

    print("MNIST (FID, précision, rappel : avec JKO | sans JKO)")
    for t, a, s in METHODES + [l for _, g in VARIANTES for l in g]:
        texte = f"  {nom_ligne(t.replace(chr(10), ' '), res[a or s]):<52}"
        for n in (a, s):
            texte += (f" | FID {res[n]['fid_10k']:6.1f} P {res[n]['precision']:.2f} R {res[n]['rappel']:.2f}"
                      if n else " | —")
        print(texte)


if __name__ == "__main__":
    quoi = sys.argv[1:] or ["2d", "mnist"]
    if "2d" in quoi:
        banc_simple()
        sinkhorn()
    if "mnist" in quoi:
        mnist()

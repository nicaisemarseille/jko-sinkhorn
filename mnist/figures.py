"""Résumé et figures d'une nuit Kaggle, à partir des sorties récupérées par `kaggle_mnist.py pull`.

    python figures.py --run-id n1

Lit resultats_kaggle/<run-id>/<run>/ (metrics.json, log.txt, final_samples.png) et écrit
resultats/<run-id>.json (FID au fil des pas JKO, temps), figures/echantillons.png et figures/fid.png.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
RUNS = [("mmd_gan_jko", "MMD-GAN, JKO"), ("mmd_gan_sans_jko", "MMD-GAN, sans JKO"),
        ("kl_dv_jko", "KL-DV, JKO"), ("kl_dv_sans_jko", "KL-DV, sans JKO"),
        ("chi2_jko", "χ², JKO"), ("mmd_fixe_jko", "MMD fixe (alg. 1), JKO"),
        ("ot_eps0.05", "OT ε=0,05"), ("ot_eps0.01", "OT ε=0,01"),
        ("sinkhorn_appris_eps0.05", "Sinkhorn appris ε=0,05"),
        ("sinkhorn_appris_eps0.01", "Sinkhorn appris ε=0,01"),
        ("sinkhorn_direct_eps0.05", "Sinkhorn direct ε=0,05"),
        ("sinkhorn_direct_eps0.01", "Sinkhorn direct ε=0,01")]


def secondes(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def resumer(dossier):
    fid = json.loads((dossier / "metrics.json").read_text())["fid"]
    lignes = (dossier / "log.txt").read_text(encoding="utf-8").splitlines()
    points = [(int(m[1]), int(m[2]), secondes(m[3])) for l in lignes
              if (m := re.search(r"JKO step (\d+) \| iter (\d+) .*elapsed ([0-9:.]+)", l))]
    pas = [(t2 - t1) / (i2 - i1) for (k1, i1, t1), (k2, i2, t2) in zip(points, points[1:]) if k1 == k2]
    arret = next(l for l in lignes if "[ARRET]" in l)
    return dict(fid_pas_jko=[k + 1 for k in fid["JKO_step"]], fid=fid["value"],
                fid_finale=fid["value"][-1], secondes_par_iteration=statistics.median(pas),
                heures_total=float(re.search(r"\((\d+\.\d+) h", arret)[1]))


def main(run_id):
    source = ROOT / "resultats_kaggle" / run_id
    resume = {nom: resumer(source / nom) for nom, _ in RUNS}
    (ROOT / "resultats").mkdir(exist_ok=True)
    (ROOT / "resultats" / f"{run_id}.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
    (ROOT / "figures").mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(15, 12), constrained_layout=True)
    for ax, (nom, titre) in zip(axes.flat, RUNS):
        image = mpimg.imread(source / nom / "final_samples.png")
        ax.imshow(image[:8 * 30 + 2], cmap="gray")  # 64 premiers échantillons (8 lignes de 30 px)
        ax.set_title(f"{titre}\nFID finale {resume[nom]['fid_finale']:.1f}", fontsize=13)
        ax.axis("off")
    fig.suptitle("MNIST : 64 échantillons finaux par run (même bruit z partout), "
                 "40 pas JKO × 500 itérations", fontsize=15)
    fig.savefig(ROOT / "figures" / "echantillons.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, groupe in zip(axes, (RUNS[:6], RUNS[6:])):
        for nom, titre in groupe:
            ax.plot(resume[nom]["fid_pas_jko"], resume[nom]["fid"], "o-", label=titre)
        ax.set(yscale="log", xlabel="pas JKO", ylabel="FID (échelle log)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=9)
    axes[0].set_title("Méthodes du papier (critique sur les images)")
    axes[1].set_title("Nos énergies de transport (coût sur les pixels)")
    fig.savefig(ROOT / "figures" / "fid.png", dpi=110)
    plt.close(fig)
    for nom, titre in RUNS:
        r = resume[nom]
        print(f"{titre:26s} FID {r['fid_finale']:6.1f} | {r['secondes_par_iteration']:.2f} s/it | "
              f"{r['heures_total']:.2f} h")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    main(ap.parse_args().run_id)

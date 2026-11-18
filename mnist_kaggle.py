"""MNIST : lance nos runs sur Kaggle, les récupère et les résume pour figures.py.

    python mnist_kaggle.py plan                    # quels runs sur quel noyau, quel compte
    python mnist_kaggle.py essai mmd_gan_jko,...   # version minuscule en local, pour vérifier
    python mnist_kaggle.py push [--noyaux 0,1]     # un noyau = deux GPU T4, chacun enchaîne ses runs
    python mnist_kaggle.py status
    python mnist_kaggle.py pull                    # -> resultats_kaggle/<run>/ (non versionné)
    python mnist_kaggle.py resumer                 # -> resultats/mnist.json, mnist_echantillons.npz

Tous les runs : code du papier (gwf/, voir README), architecture ResNet de leur Table 1, 40 pas JKO
de 500 itérations, batch 256, tau = 0,5, graine 2026, sauf mention dans le nom du run. Chaque run
finit par les mesures de mnist_mesures.py sur 10 000 images.

Prérequis : `pip install kaggle truststore`, un dossier par compte dans
~/.kaggle-accounts/<compte>/kaggle.json (comptes sans GPU à exclure : KAGGLE_SANS_GPU=c1,c2).
Le noyau i va sur le compte i modulo le nombre de comptes. Kaggle ne fait tourner que deux noyaux
GPU à la fois par compte : s'il y a plus de deux noyaux par compte, lancer par paquets (--noyaux).
Le CLI Kaggle passe par truststore (antivirus local qui intercepte TLS).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GWF = ROOT / "gwf"
SANS_GPU = tuple(c for c in os.environ.get("KAGGLE_SANS_GPU", "").split(",") if c)
CLI = ("import sys, truststore; truststore.inject_into_ssl(); from kaggle.cli import main; "
       "sys.argv = ['kaggle'] + sys.argv[1:]; main()")

# ----- Les runs -----
BASE = ["--dataset", "mnist28", "--image_size", "28", "--num_channels", "1",
        "--model_name", "resnet_MMDGAN", "--nz", "128", "--batch_size", "256",
        "--lambda2", "1", "--JKO_steps", "40", "--inner_iterations_first", "500",
        "--inner_iterations", "500", "--print_every", "100", "--save_image_every", "5",
        "--evaluate_every", "10", "--fid", "--mesures", "--seed", "2026"]   # tau = lambda2 / 2
# Critiques du papier (MMD-GAN : noyau sur un encodeur appris ; KL : f-divergence ; WGAN : W1).
CRITIQUE_MMD = ["--embed_dim", "128", "--gp_type", "interpolated", "--lbda_gp", "10"]
CRITIQUE_KL = ["--embed_dim", "1", "--gp_type", "sjko", "--lbda_gp", "0.2"]
CRITIQUE_WGAN = ["--embed_dim", "1", "--gp_type", "WGAN", "--lbda_gp", "10"]
CRITIQUE_WGAN_DUKLER = ["--embed_dim", "1", "--gp_type", "dukler", "--lbda_gp", "10"]
SANS_CRITIQUE = ["--embed_dim", "1", "--lbda_gp", "0"]
# Sinkhorn : l'encodeur adversarial a l'architecture et la pénalité du critique du MMD-GAN.
ADVERSARIAL = ["--espace", "encodeur_adversarial", *CRITIQUE_MMD]
ADVERSARIAL_DUKLER = ["--espace", "encodeur_adversarial", "--embed_dim", "128", "--gp_type", "dukler",
                      "--lbda_gp", "10", "--dukler_cible", "0"]
TAU5, TAU01 = ["--lambda2", "10"], ["--lambda2", "0.2"]
PAS_400 = ["--JKO_steps", "400", "--inner_iterations", "50", "--evaluate_every", "40",
           "--save_image_every", "50"]                       # 400 pas de 50 itérations
CRITIQUE_MMD_FIGE = "outputs/MMD/mnist28/mmd_gan_jko/critique.pt"   # écrit par mmd_gan_jko


def run(nom, divergence, *args, jko=True, budget=None, attendre=None):
    """budget : heures au plus pour ce run ; attendre : fichier produit par un autre run."""
    return dict(nom=nom + ("_jko" if jko else "_sans_jko"), budget=budget, attendre=attendre,
                args=BASE + ["--divergence", divergence, "--JKO" if jko else "--no-JKO", *args])


def sinkhorn(nom, *args, eps=0.05, **kw):
    return run("sinkhorn_" + nom, "Sinkhorn", *SANS_CRITIQUE, "--eps_sinkhorn", str(eps), *args, **kw)


def avec_sans(f):
    """Un noyau : GPU 0 avec JKO, GPU 1 sans JKO."""
    return [[f(True)], [f(False)]]


# Un noyau = [runs du GPU 0, runs du GPU 1].
NOYAUX = [
    # Méthodes du papier, et Sinkhorn dans l'espace du critique final du MMD-GAN avec JKO, figé
    [[run("mmd_gan", "MMD", *CRITIQUE_MMD, budget=6.0),
      sinkhorn("encodeur_mmd_fige", "--espace", "encodeur_mmd_fige", "--encodeur_ckpt", CRITIQUE_MMD_FIGE,
               attendre=CRITIQUE_MMD_FIGE)],
     [run("mmd_gan", "MMD", *CRITIQUE_MMD, jko=False, budget=6.0),
      sinkhorn("encodeur_mmd_fige", "--espace", "encodeur_mmd_fige", "--encodeur_ckpt", CRITIQUE_MMD_FIGE,
               jko=False, attendre=CRITIQUE_MMD_FIGE)]],
    avec_sans(lambda j: run("kl", "KL_DV", *CRITIQUE_KL, jko=j)),
    avec_sans(lambda j: run("wgan", "Wasserstein-1", *CRITIQUE_WGAN, jko=j)),
    avec_sans(lambda j: run("wgan_dukler", "Wasserstein-1", *CRITIQUE_WGAN_DUKLER, jko=j)),
    # Sinkhorn, encodeur adversarial
    avec_sans(lambda j: sinkhorn("encodeur_adversarial", *ADVERSARIAL, jko=j)),
    avec_sans(lambda j: sinkhorn("encodeur_adversarial_graine2", *ADVERSARIAL, "--seed", "2027", jko=j)),
    [[sinkhorn("encodeur_adversarial_tau0.1", *ADVERSARIAL, *TAU01)],
     [sinkhorn("encodeur_adversarial_tau5", *ADVERSARIAL, *TAU5)]],
    [[sinkhorn("encodeur_adversarial_eps0.01", *ADVERSARIAL, eps=0.01)],
     [sinkhorn("encodeur_adversarial_eps0.2", *ADVERSARIAL, eps=0.2)]],
    avec_sans(lambda j: sinkhorn("encodeur_adversarial_dukler", *ADVERSARIAL_DUKLER, jko=j)),
    # Sinkhorn, pixels
    [[sinkhorn("pixels"), sinkhorn("pixels_eps0.01", eps=0.01), sinkhorn("pixels_tau5", *TAU5)],
     [sinkhorn("pixels", jko=False), sinkhorn("pixels_eps0.01", eps=0.01, jko=False),
      sinkhorn("pixels_batch64", "--batch_size", "64", jko=False)]],
    [[sinkhorn("pixels_annealing_0.05_0.005", "--eps_final", "0.005"),
      sinkhorn("pixels_annealing_0.1_0.01", "--eps_final", "0.01", eps=0.1),
      sinkhorn("pixels_400pas", *PAS_400)],
     [sinkhorn("pixels_annealing_0.05_0.005", "--eps_final", "0.005", jko=False),
      sinkhorn("pixels_annealing_0.1_0.01", "--eps_final", "0.01", eps=0.1, jko=False),
      sinkhorn("pixels_400pas_tau5", *PAS_400, *TAU5)]],
    # Sinkhorn, autres coûts fixes
    [[sinkhorn("encodeur_fixe", "--espace", "encodeur_fixe"),
      sinkhorn("encodeur_fixe_amorce", "--espace", "encodeur_fixe", "--amorce", "1000"),
      sinkhorn("hm1", "--espace", "hm1")],
     [sinkhorn("encodeur_fixe", "--espace", "encodeur_fixe", jko=False),
      sinkhorn("classifieur_fige_amorce", "--espace", "classifieur_fige", "--amorce", "1000"),
      sinkhorn("hm1", "--espace", "hm1", jko=False)]],
    [[sinkhorn("classifieur_fige", "--espace", "classifieur_fige"),
      sinkhorn("dukler", "--espace", "dukler", "--batch_size", "64")],          # batch 64 : trop lent à 256
     [sinkhorn("classifieur_fige", "--espace", "classifieur_fige", jko=False),
      sinkhorn("dukler", "--espace", "dukler", "--batch_size", "64", jko=False)]],
]
RUNS = [r for noyau in NOYAUX for chaine in noyau for r in chaine]
assert len({r["nom"] for r in RUNS}) == len(RUNS)
BUDGET_HEURES = 10.5   # session Kaggle limitée à 12 h

# ----- Kaggle -----
WRAPPER = '''# Généré par mnist_kaggle.py -- ne pas modifier.
import base64, io, json, os, shutil, subprocess, sys, threading, time, zipfile
WORK, REPO = "/kaggle/working", "/tmp/gwf"
os.makedirs(os.path.join(WORK, "out"), exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode("{payload}"))) as zf:
    zf.extractall(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-fid"])
import torch
n_gpu = torch.cuda.device_count()
env = dict(python=sys.version.split()[0], torch=torch.__version__, n_gpu=n_gpu,
           gpus=[torch.cuda.get_device_name(i) for i in range(n_gpu)])
print("ENVIRONNEMENT " + json.dumps(env), flush=True)
json.dump(env, open(os.path.join(WORK, "out", "environnement.json"), "w"), indent=2)
CHAINES, BUDGET, DEBUT = {chaines!r}, {budget!r}, time.time()
MARGE = 0.3  # heures gardées pour les mesures de fin et la copie des sorties

def lancer(i, r, gpu, budget):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), FID_BASE_DIR="/tmp/fid_%d" % i)
    sortie = open(os.path.join(WORK, "out", r["nom"] + ".console.txt"), "w")
    cmd = [sys.executable, "-W", "ignore", "train.py", *r["args"], "--exp", r["nom"],
           "--data_root", "/tmp/data_%d" % i, "--budget_heures", "%.3f" % budget]
    return subprocess.Popen(cmd, cwd=REPO, env=e, stdout=sortie, stderr=subprocess.STDOUT)

codes = {{}}

def chaine(i, runs, gpu):
    for j, r in enumerate(runs):
        if r.get("attendre"):  # fichier produit par un autre run du même noyau
            f = os.path.join(REPO, r["attendre"])
            while not os.path.exists(f) and time.time() - DEBUT < 3600 * (BUDGET - 2):
                time.sleep(30)
            if not os.path.exists(f):
                codes[r["nom"]] = "fichier attendu absent"
                continue
        reste = BUDGET - (time.time() - DEBUT) / 3600 - MARGE
        budget = reste if j == len(runs) - 1 or not r.get("budget") else min(r["budget"], reste)
        print("LANCE", r["nom"], "gpu", gpu, "budget %.2f h" % budget, flush=True)
        codes[r["nom"]] = lancer(i, r, gpu, budget).wait()
        print("FINI", r["nom"], codes[r["nom"]], flush=True)

if n_gpu >= len(CHAINES):
    fils = [threading.Thread(target=chaine, args=(i, c, i)) for i, c in enumerate(CHAINES)]
    for t in fils:
        t.start()
    for t in fils:
        t.join()
else:  # un seul GPU : tout à la suite
    chaine(0, [r for c in CHAINES for r in c], 0)
json.dump(dict(codes=codes), open(os.path.join(WORK, "out", "codes.json"), "w"), indent=2)
if os.path.isdir(os.path.join(REPO, "outputs")):
    shutil.copytree(os.path.join(REPO, "outputs"), os.path.join(WORK, "out", "outputs"), dirs_exist_ok=True)
sys.exit(0 if codes and all(c == 0 for c in codes.values()) else 1)
'''


def comptes():
    return [d.name for d in sorted((Path.home() / ".kaggle-accounts").iterdir())
            if (d / "kaggle.json").is_file() and d.name not in SANS_GPU]


def plan(choix=None):
    c = comptes()
    return [dict(i=i, id=f"{c[i % len(c)]}/jko-mnist-{i}", compte=c[i % len(c)], chaines=n,
                 runs=[r for ch in n for r in ch])
            for i, n in enumerate(NOYAUX) if choix is None or i in choix]


def kaggle(compte, *args):
    env = dict(os.environ, KAGGLE_CONFIG_DIR=str(Path.home() / ".kaggle-accounts" / compte),
               PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-c", CLI, *args], capture_output=True, text=True, env=env)


def payload():
    """gwf/ (code du papier patché + nos modules) zippé dans le script du noyau."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in GWF.rglob("*"):
            rel = f.relative_to(GWF)
            if f.is_file() and not ({".git", "outputs", "__pycache__", "fid_cache", "data"} & set(rel.parts)):
                zf.write(f, rel.as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def push(choix):
    pl = payload()
    for k in plan(choix):
        d = ROOT / "kaggle_jobs" / str(k["i"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "script.py").write_text(WRAPPER.format(payload=pl, chaines=k["chaines"], budget=BUDGET_HEURES),
                                     encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": k["id"], "title": k["id"].split("/")[1], "code_file": "script.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "enable_tpu": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "kernel_sources": [], "competition_sources": []}, indent=2))
        r = kaggle(k["compte"], "kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4")
        sortie = (r.stdout + r.stderr).strip().splitlines()
        mauvais = r.returncode != 0 or any("error" in l.lower() for l in sortie)
        print(f"{'ECHEC' if mauvais else 'lancé'} {k['id']}  {[x['nom'] for x in k['runs']]}"
              f"  | {sortie[-1] if sortie else ''}", flush=True)


def status(choix):
    for k in plan(choix):
        r = kaggle(k["compte"], "kernels", "status", k["id"])
        lignes = (r.stdout or r.stderr).strip().splitlines()
        print(f"{k['id']:45s} {lignes[-1] if lignes else '?'}", flush=True)


def pull(choix):
    """Sorties regroupées par run dans resultats_kaggle/<run>/ ; log Kaggle, codes de retour et
    versions dans resultats_kaggle/_noyaux/<i>/."""
    for k in plan(choix):
        tmp = ROOT / "resultats_kaggle" / f"_tmp{k['i']}"
        tmp.mkdir(parents=True, exist_ok=True)
        r = kaggle(k["compte"], "kernels", "output", k["id"], "-p", str(tmp))
        for x in k["runs"]:
            dest = ROOT / "resultats_kaggle" / x["nom"]
            dest.mkdir(parents=True, exist_ok=True)
            for f in tmp.rglob("*"):
                if f.is_file() and (x["nom"] in f.parts or f.name.startswith(x["nom"] + ".")):
                    shutil.copy2(f, dest / f.name)
        noyau = ROOT / "resultats_kaggle" / "_noyaux" / str(k["i"])
        noyau.mkdir(parents=True, exist_ok=True)
        for f in tmp.rglob("*"):
            if f.is_file() and (f.suffix == ".log" or f.name in ("codes.json", "environnement.json")):
                shutil.copy2(f, noyau / f.name)
        n = len([p for p in tmp.rglob("*") if p.is_file()])
        shutil.rmtree(tmp)
        print(f"noyau {k['i']}: {n} fichiers  {(r.stderr or '').strip()[-200:]}", flush=True)


def resumer(source):
    """Pour chaque run : mesures finales, FID au fil des pas, durée, et 36 images finales."""
    import numpy as np
    resume, images = {}, {}
    for r in RUNS:
        d = Path(source) / r["nom"]
        if not (d / "mesures.json").exists():
            print("manquant :", r["nom"])
            continue
        m = json.loads((d / "mesures.json").read_text())
        fid = json.loads((d / "metrics.json").read_text())["fid"]
        arret = re.search(r"\[ARRET\] dernier pas JKO (\d+) \((\d+\.\d+) h",
                          (d / "log.txt").read_text(encoding="utf-8"))
        config = json.loads((d / "config.json").read_text())
        m.update(fid_pas_jko=[s + 1 for s in fid["JKO_step"]], fid=fid["value"], heures=float(arret[2]),
                 pas_jko=int(arret[1]) + 1, pas_jko_prevus=config["JKO_steps"], jko=config["JKO"],
                 tau=config["lambda2"] / 2)
        resume[r["nom"]] = m
        images[r["nom"]] = np.load(d / "echantillons_10k.npz")["images"][:36, 0]
    sortie = ROOT / "resultats"
    sortie.mkdir(exist_ok=True)
    (sortie / "mnist.json").write_text(json.dumps(resume, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(sortie / "mnist_echantillons.npz", **images)
    print(f"{len(resume)} runs résumés -> resultats/mnist.json, resultats/mnist_echantillons.npz")


COURT = ["--JKO_steps", "2", "--inner_iterations_first", "20", "--inner_iterations", "20",
         "--evaluate_every", "1", "--print_every", "10", "--batch_size", "16", "--no-mesures"]


def essai(noms, data_root):
    """Chaque run demandé, en version minuscule (2 pas de 20 itérations, batch 16), dans gwf/."""
    import time
    par_nom = {r["nom"]: r for r in RUNS}
    for nom in noms:
        a = [x for x in par_nom[nom]["args"] if x != "--fid"] + COURT
        if "--amorce" in a:
            a[a.index("--amorce") + 1] = "30"
        t = time.time()
        p = subprocess.run([sys.executable, "-W", "ignore", "train.py", *a, "--exp", "essai_" + nom,
                            "--data_root", str(Path(data_root).resolve())],
                           cwd=GWF, capture_output=True, text=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        print(f"{nom:45s} code {p.returncode}  {time.time() - t:.0f} s", flush=True)
        if p.returncode:
            print(p.stderr[-2500:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "essai", "push", "status", "pull", "resumer"])
    ap.add_argument("runs", nargs="?", help="pour essai : noms de runs séparés par des virgules")
    ap.add_argument("--noyaux", help="numéros de noyaux séparés par des virgules (tous par défaut)")
    ap.add_argument("--source", default=ROOT / "resultats_kaggle", help="pour resumer")
    ap.add_argument("--data_root", default=ROOT / "data" / "mnist", help="pour essai")
    a = ap.parse_args()
    choix = [int(i) for i in a.noyaux.split(",")] if a.noyaux else None
    if a.action == "plan":
        for k in plan(choix):
            print(k["id"], [[x["nom"] for x in c] for c in k["chaines"]])
    elif a.action == "essai":
        essai(a.runs.split(","), a.data_root)
    elif a.action == "resumer":
        resumer(a.source)
    else:
        {"push": push, "status": status, "pull": pull}[a.action](choix)

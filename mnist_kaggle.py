"""MNIST : nos énergies dans le code du papier, lancées sur des comptes Kaggle, puis résumées.

    python mnist_kaggle.py plan    --run-id n1     # quels runs sur quels comptes
    python mnist_kaggle.py push    --run-id n1     # un noyau T4 x2 par paire de runs
    python mnist_kaggle.py status  --run-id n1
    python mnist_kaggle.py pull    --run-id n1     # -> resultats_kaggle/n1/<run>/ (non versionné)
    python mnist_kaggle.py resumer                 # resultats_kaggle/ -> resultats/mnist.json et
                                                   #   resultats/mnist_echantillons.npz (lus par figures.py)

Campagnes déjà faites : n1 (toutes les énergies, avec JKO, + deux témoins sans JKO du papier),
n2 (témoins sans JKO pour Sinkhorn, et W1 avec / sans JKO).

Prérequis : gwf/ préparé à côté de ce fichier (voir README), `pip install kaggle truststore`,
un dossier par compte dans ~/.kaggle-accounts/<compte>/kaggle.json. Comptes sans GPU à exclure :
KAGGLE_SANS_GPU=compte1,compte2. Chaque noyau embarque gwf/ dans un zip et lance ses deux runs en
parallèle, un par GPU. Le CLI Kaggle passe par truststore (antivirus local qui intercepte TLS).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GWF = ROOT / "gwf"
SANS_GPU = tuple(c for c in os.environ.get("KAGGLE_SANS_GPU", "").split(",") if c)
COMPTES = [d.name for d in sorted((Path.home() / ".kaggle-accounts").iterdir())
           if (d / "kaggle.json").is_file() and d.name not in SANS_GPU]
CLI = ("import sys, truststore; truststore.inject_into_ssl(); from kaggle.cli import main; "
       "sys.argv = ['kaggle'] + sys.argv[1:]; main()")

# Réglages communs : architecture ResNet-MMD de leur Table 1 (MNIST), tau = 0,5 (lambda2 = 1).
BASE = ["--dataset", "mnist28", "--image_size", "28", "--num_channels", "1",
        "--model_name", "resnet_MMDGAN", "--nz", "128", "--batch_size", "256",
        "--lambda2", "1", "--JKO_steps", "40", "--inner_iterations_first", "500",
        "--inner_iterations", "500", "--print_every", "100", "--save_image_every", "5",
        "--evaluate_every", "10", "--fid"]
CRITIQUE_F = ["--embed_dim", "1", "--gp_type", "sjko", "--lbda_gp", "0.2"]
CRITIQUE_MMD = ["--embed_dim", "128", "--gp_type", "interpolated", "--lbda_gp", "10"]
SANS = ["--embed_dim", "1", "--lbda_gp", "0"]


def run(nom, divergence, *extra, jko=True):
    return dict(nom=nom, args=BASE + ["--divergence", divergence, "--JKO" if jko else "--no-JKO", *extra])


# Six noyaux : on apparie un run lent et un run rapide.
NUIT = [
    [run("sinkhorn_appris_eps0.05", "Sinkhorn_appris", *CRITIQUE_F, "--eps_sinkhorn", "0.05"),
     run("sinkhorn_direct_eps0.05", "Sinkhorn_direct", *SANS, "--eps_sinkhorn", "0.05")],
    [run("sinkhorn_appris_eps0.01", "Sinkhorn_appris", *CRITIQUE_F, "--eps_sinkhorn", "0.01"),
     run("sinkhorn_direct_eps0.01", "Sinkhorn_direct", *SANS, "--eps_sinkhorn", "0.01")],
    [run("mmd_gan_jko", "MMD", *CRITIQUE_MMD),
     run("mmd_gan_sans_jko", "MMD", *CRITIQUE_MMD, jko=False)],
    [run("kl_dv_jko", "KL_DV", *CRITIQUE_F),
     run("chi2_jko", "chi2", *CRITIQUE_F)],
    [run("ot_eps0.05", "OT_eps", *CRITIQUE_F, "--eps_sinkhorn", "0.05"),
     run("ot_eps0.01", "OT_eps", *CRITIQUE_F, "--eps_sinkhorn", "0.01")],
    [run("mmd_fixe_jko", "MMD_fixe", *SANS),
     run("kl_dv_sans_jko", "KL_DV", *CRITIQUE_F, jko=False)],
]
# Campagne n2 : témoins sans JKO pour les énergies de Sinkhorn, et W1 (Kantorovich-Rubinstein,
# critique 1-lipschitz par pénalité WGAN-GP) avec et sans JKO. Mêmes réglages que n1.
CRITIQUE_W1 = ["--embed_dim", "1", "--gp_type", "WGAN", "--lbda_gp", "10"]
NUIT2 = [
    [run("sinkhorn_appris_eps0.01_sans_jko", "Sinkhorn_appris", *CRITIQUE_F, "--eps_sinkhorn", "0.01", jko=False),
     run("sinkhorn_direct_eps0.01_sans_jko", "Sinkhorn_direct", *SANS, "--eps_sinkhorn", "0.01", jko=False)],
    [run("sinkhorn_appris_eps0.05_sans_jko", "Sinkhorn_appris", *CRITIQUE_F, "--eps_sinkhorn", "0.05", jko=False),
     run("sinkhorn_direct_eps0.05_sans_jko", "Sinkhorn_direct", *SANS, "--eps_sinkhorn", "0.05", jko=False)],
    [run("w1_jko", "Wasserstein-1", *CRITIQUE_W1),
     run("w1_sans_jko", "Wasserstein-1", *CRITIQUE_W1, jko=False)],
]
CAMPAGNES = {"n1": NUIT, "n2": NUIT2}

COURT = ["--JKO_steps", "2", "--inner_iterations_first", "20", "--inner_iterations", "20",
         "--evaluate_every", "1", "--print_every", "10"]
FUMEE = [[dict(nom=r["nom"] + "_fumee", args=r["args"] + COURT) for r in (NUIT[0][0], NUIT[2][0])]]

BUDGET_HEURES = 10.5  # session Kaggle limitée à 12 h

WRAPPER = '''# Généré par mnist_kaggle.py -- ne pas modifier.
import base64, io, json, os, shutil, subprocess, sys, time, zipfile
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
RUNS, BUDGET = {runs!r}, {budget!r}

def lancer(i, r, gpu, budget):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), FID_BASE_DIR="/tmp/fid_%d" % i)
    sortie = open(os.path.join(WORK, "out", r["nom"] + ".console.txt"), "w")
    cmd = [sys.executable, "-W", "ignore", "train.py", *r["args"], "--exp", r["nom"],
           "--data_root", "/tmp/data_%d" % i, "--budget_heures", str(budget)]
    return subprocess.Popen(cmd, cwd=REPO, env=e, stdout=sortie, stderr=subprocess.STDOUT)

codes = []
if n_gpu >= len(RUNS):
    procs = [lancer(i, r, i, BUDGET) for i, r in enumerate(RUNS)]
    codes = [p.wait() for p in procs]
else:
    for i, r in enumerate(RUNS):
        codes.append(lancer(i, r, 0, BUDGET / len(RUNS)).wait())
json.dump(dict(codes=codes, runs=[r["nom"] for r in RUNS]),
          open(os.path.join(WORK, "out", "codes.json"), "w"))
if os.path.isdir(os.path.join(REPO, "outputs")):
    shutil.copytree(os.path.join(REPO, "outputs"), os.path.join(WORK, "out", "outputs"), dirs_exist_ok=True)
sys.exit(max(codes) if codes else 1)
'''


def payload():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in GWF.rglob("*"):
            rel = f.relative_to(GWF)
            if f.is_file() and not ({".git", "outputs", "__pycache__", "fid_cache"} & set(rel.parts)):
                zf.write(f, rel.as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def plan(fumee, run_id="n1"):
    noyaux = FUMEE if fumee else CAMPAGNES[run_id]
    assert len(noyaux) <= len(COMPTES), (len(noyaux), COMPTES)
    comptes = COMPTES[::-1] if fumee else COMPTES  # la fumée prend un compte en second slot
    return [dict(compte=comptes[i], nom=f"k{i}", runs=runs) for i, runs in enumerate(noyaux)]


def kaggle(compte, *args):
    env = dict(os.environ, KAGGLE_CONFIG_DIR=str(Path.home() / ".kaggle-accounts" / compte),
               PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-c", CLI, *args], capture_output=True, text=True, env=env)


def kid(k, run_id):
    return f"{k['compte']}/jko-mnist-{run_id}-{k['nom']}"


def push(run_id, fumee):
    pl = payload()
    for k in plan(fumee, run_id):
        d = ROOT / "kaggle_jobs" / run_id / k["nom"]
        d.mkdir(parents=True, exist_ok=True)
        budget = 0.5 if fumee else BUDGET_HEURES
        (d / "script.py").write_text(WRAPPER.format(payload=pl, runs=k["runs"], budget=budget), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": kid(k, run_id), "title": kid(k, run_id).split("/")[1], "code_file": "script.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "enable_tpu": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "kernel_sources": [], "competition_sources": []}, indent=2))
        r = kaggle(k["compte"], "kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4")
        sortie = (r.stdout + r.stderr).strip().splitlines()
        mauvais = r.returncode != 0 or any("error" in l.lower() for l in sortie)
        print(f"{'ECHEC' if mauvais else 'lancé'} {kid(k, run_id)}  {[x['nom'] for x in k['runs']]}"
              f"  | {sortie[-1] if sortie else ''}", flush=True)


def status(run_id, fumee):
    for k in plan(fumee, run_id):
        r = kaggle(k["compte"], "kernels", "status", kid(k, run_id))
        lignes = (r.stdout or r.stderr).strip().splitlines()
        print(f"{kid(k, run_id):55s} {lignes[-1] if lignes else '?'}", flush=True)


def pull(run_id, fumee):
    for k in plan(fumee, run_id):
        tmp = ROOT / "resultats_kaggle" / run_id / ("_" + k["nom"])
        tmp.mkdir(parents=True, exist_ok=True)
        r = kaggle(k["compte"], "kernels", "output", kid(k, run_id), "-p", str(tmp))
        # Regroupe par run : log, métriques, images, console.
        for x in k["runs"]:
            dest = ROOT / "resultats_kaggle" / run_id / x["nom"]
            dest.mkdir(parents=True, exist_ok=True)
            for f in tmp.rglob("*"):
                if f.is_file() and (x["nom"] in f.parts or f.name.startswith(x["nom"] + ".")):
                    shutil.copy2(f, dest / f.name)
        # Ce qui concerne le noyau entier (log Kaggle, codes de retour, versions) -> _noyaux/<k>/.
        noyau = ROOT / "resultats_kaggle" / run_id / "_noyaux" / k["nom"]
        noyau.mkdir(parents=True, exist_ok=True)
        for f in tmp.rglob("*"):
            if f.is_file() and (f.suffix == ".log" or f.name in ("codes.json", "environnement.json")):
                shutil.copy2(f, noyau / f.name)
        n = len([p for p in tmp.rglob("*") if p.is_file()])
        shutil.rmtree(tmp)
        print(f"{k['nom']}: {n} fichiers  {(r.stderr or '').strip()[-200:]}", flush=True)


# Résumé pour figures.py : (campagne, run) de chaque condition, avec et sans terme proximal JKO.
CONDITIONS = {
    "kl_dv": (("n1", "kl_dv_jko"), ("n1", "kl_dv_sans_jko")),
    "chi2": (("n1", "chi2_jko"), None),
    "mmd_fixe": (("n1", "mmd_fixe_jko"), None),
    "mmd_gan": (("n1", "mmd_gan_jko"), ("n1", "mmd_gan_sans_jko")),
    "w1": (("n2", "w1_jko"), ("n2", "w1_sans_jko")),
    "ot_eps0.01": (("n1", "ot_eps0.01"), None),
    "ot_eps0.05": (("n1", "ot_eps0.05"), None),
    "sinkhorn_appris_eps0.01": (("n1", "sinkhorn_appris_eps0.01"), ("n2", "sinkhorn_appris_eps0.01_sans_jko")),
    "sinkhorn_appris_eps0.05": (("n1", "sinkhorn_appris_eps0.05"), ("n2", "sinkhorn_appris_eps0.05_sans_jko")),
    "sinkhorn_direct_eps0.01": (("n1", "sinkhorn_direct_eps0.01"), ("n2", "sinkhorn_direct_eps0.01_sans_jko")),
    "sinkhorn_direct_eps0.05": (("n1", "sinkhorn_direct_eps0.05"), ("n2", "sinkhorn_direct_eps0.05_sans_jko")),
}


def resumer(source):
    import re
    import statistics
    import numpy as np
    import matplotlib.image as mpimg

    def secondes(t):
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    resume, images = {}, {}
    for nom, variantes in CONDITIONS.items():
        for jko, cle in zip(("jko", "sans_jko"), variantes):
            if cle is None:
                continue
            d = Path(source) / cle[0] / cle[1]
            fid = json.loads((d / "metrics.json").read_text())["fid"]
            lignes = (d / "log.txt").read_text(encoding="utf-8").splitlines()
            pts = [(int(m[1]), int(m[2]), secondes(m[3])) for l in lignes
                   if (m := re.search(r"JKO step (\d+) \| iter (\d+) .*elapsed ([0-9:.]+)", l))]
            vit = [(t2 - t1) / (i2 - i1) for (k1, i1, t1), (k2, i2, t2) in zip(pts, pts[1:]) if k1 == k2]
            arret = next(l for l in lignes if "[ARRET]" in l)
            resume[f"{nom}/{jko}"] = dict(
                campagne=cle[0], run=cle[1], fid_pas_jko=[k + 1 for k in fid["JKO_step"]], fid=fid["value"],
                secondes_par_iteration=round(statistics.median(vit), 3),
                heures=float(re.search(r"\((\d+\.\d+) h", arret)[1]))
            # 64 premiers échantillons finaux (8 lignes de 30 px), en niveaux de gris 8 bits.
            img = mpimg.imread(d / "final_samples.png")[:8 * 30 + 2]
            images[f"{nom}/{jko}"] = (img[..., 0] * 255).round().astype(np.uint8)
    (ROOT / "resultats").mkdir(exist_ok=True)
    (ROOT / "resultats" / "mnist.json").write_text(json.dumps(resume, indent=1, ensure_ascii=False),
                                                   encoding="utf-8")
    np.savez_compressed(ROOT / "resultats" / "mnist_echantillons.npz", **images)
    print(f"{len(resume)} runs résumés -> resultats/mnist.json, resultats/mnist_echantillons.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "push", "status", "pull", "resumer"])
    ap.add_argument("--run-id")
    ap.add_argument("--fumee", action="store_true")
    ap.add_argument("--source", default=ROOT / "resultats_kaggle", help="pour resumer")
    a = ap.parse_args()
    if a.action == "resumer":
        resumer(a.source)
    elif a.action == "plan":
        for k in plan(a.fumee, a.run_id):
            print(kid(k, a.run_id), [x["nom"] for x in k["runs"]])
    else:
        {"push": push, "status": status, "pull": pull}[a.action](a.run_id, a.fumee)

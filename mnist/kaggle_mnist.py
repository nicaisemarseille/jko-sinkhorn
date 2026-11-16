"""Nos énergies 2D sur MNIST, réparties sur les comptes Kaggle (~/.kaggle-accounts/<compte>/kaggle.json).

Prérequis : gwf/ préparé à côté de ce fichier (voir README), `pip install kaggle truststore`.

    python kaggle_mnist.py plan   --run-id n1
    python kaggle_mnist.py push   --run-id n1            # la nuit : 6 noyaux, 2 runs chacun
    python kaggle_mnist.py push   --run-id f1 --fumee    # essai court de bout en bout (1 noyau)
    python kaggle_mnist.py status --run-id n1
    python kaggle_mnist.py pull   --run-id n1            # -> resultats_kaggle/n1/<run>/

Chaque noyau est un script privé (GPU T4 x2, internet activé pour pytorch-fid, MNIST et les
poids Inception). Il embarque gwf/ (leur dépôt + nos énergies) dans un zip et lance ses deux
runs en parallèle, un par GPU. Les caches FID vont dans /tmp pour ne pas gonfler la sortie.
Le CLI Kaggle passe par truststore (l'antivirus local intercepte TLS).
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
# Comptes à exclure (sans GPU), séparés par des virgules : KAGGLE_SANS_GPU=compte1,compte2
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
COURT = ["--JKO_steps", "2", "--inner_iterations_first", "20", "--inner_iterations", "20",
         "--evaluate_every", "1", "--print_every", "10"]
FUMEE = [[dict(nom=r["nom"] + "_fumee", args=r["args"] + COURT) for r in (NUIT[0][0], NUIT[2][0])]]

BUDGET_HEURES = 10.5  # session Kaggle limitée à 12 h

WRAPPER = '''# Généré par images/kaggle_mnist.py -- ne pas modifier.
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


def plan(fumee):
    noyaux = FUMEE if fumee else NUIT
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
    for k in plan(fumee):
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
    for k in plan(fumee):
        r = kaggle(k["compte"], "kernels", "status", kid(k, run_id))
        lignes = (r.stdout or r.stderr).strip().splitlines()
        print(f"{kid(k, run_id):55s} {lignes[-1] if lignes else '?'}", flush=True)


def pull(run_id, fumee):
    for k in plan(fumee):
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
        n = len([p for p in tmp.rglob("*") if p.is_file()])
        print(f"{k['nom']}: {n} fichiers  {(r.stderr or '').strip()[-200:]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "push", "status", "pull"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--fumee", action="store_true")
    a = ap.parse_args()
    if a.action == "plan":
        for k in plan(a.fumee):
            print(kid(k, a.run_id), [x["nom"] for x in k["runs"]])
    else:
        {"push": push, "status": status, "pull": pull}[a.action](a.run_id, a.fumee)

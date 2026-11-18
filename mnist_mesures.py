"""Mesures de fin de run (copié en gwf/mesures_nous.py), sur 10 000 images générées contre les 10 000 images de test MNIST.

  fid_10k           FID (Inception, ajustement gaussien)
  sw_pixels         Sliced Wasserstein-2 sur les pixels (images aplaties / sqrt(d), la métrique du JKO)
  sw_inception      Sliced Wasserstein-2 dans l'espace Inception (l'idée de MIND, sans hypothèse gaussienne)
  precision, rappel k plus proches voisins, k = 3, dans l'espace Inception (Kynkäänniemi et al. 2019,
                    qui utilisent VGG-16 ; ici Inception, comme la FID)
  densite, couverture   k = 5 (Naeem et al. 2020)
  classes           proportions des 10 chiffres selon le classifieur figé, écart (variation totale)
                    aux proportions réelles, confiance moyenne du classifieur

Les 10 000 images sont aussi sauvegardées (uint8), pour recalculer n'importe quelle mesure plus tard.
Utilisable seul : python mesures_nous.py echantillons_10k.npz --data_root ../data/mnist
"""
import json
import math
import os

import numpy as np
import torch

N = 10_000


@torch.no_grad()
def generer(netT, noise_sampler, device, n=N):
    netT.eval()
    morceaux, total = [], 0
    while total < n:
        x = netT(noise_sampler.sample().to(device)).float().clamp(-1, 1)
        morceaux.append(x.cpu())
        total += len(x)
    return torch.cat(morceaux)[:n]


def vers_uint8(x):
    return ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()


def depuis_uint8(a):
    return torch.from_numpy(a).float() / 127.5 - 1


def test_mnist(data_root, n=N):
    from torchvision import datasets
    ds = datasets.MNIST(data_root, train=False, download=True)
    x = ds.data[:n].float().div(127.5).sub(1).unsqueeze(1)   # même normalisation que leur loader
    return x, ds.targets[:n]


@torch.no_grad()
def inception(x, device, lot=250):
    from pytorch_fid.inception import InceptionV3
    m = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    sorties = []
    for i in range(0, len(x), lot):
        b = ((x[i:i + lot].to(device) + 1) / 2).repeat(1, 3, 1, 1)
        sorties.append(m(b)[0].flatten(1).double().cpu())
    return torch.cat(sorties)


@torch.no_grad()
def sliced_w2(a, b, device, n_dir=1000, graine=0):
    g = torch.Generator().manual_seed(graine)
    d = torch.randn(a.shape[1], n_dir, generator=g, dtype=torch.float64)
    d = (d / d.norm(dim=0, keepdim=True)).to(device)
    pa = torch.sort(a.to(device) @ d, 0).values
    pb = torch.sort(b.to(device) @ d, 0).values
    return (pa - pb).square().mean().sqrt().item()


@torch.no_grad()
def distances(a, b, device, lot=1000):
    """Matrice des distances euclidiennes, par blocs (len(a) x len(b))."""
    b = b.to(device)
    return torch.cat([torch.cdist(a[i:i + lot].to(device), b) for i in range(0, len(a), lot)])


@torch.no_grad()
def rayons(f, k, device):
    d = distances(f, f, device)
    return d.topk(k + 1, dim=1, largest=False).values[:, k]   # k-ième voisin, soi-même exclu


@torch.no_grad()
def voisinages(fr, fg, device):
    r3, r5 = rayons(fr, 3, device), rayons(fr, 5, device)
    g3 = rayons(fg, 3, device)
    dgr = distances(fg, fr, device)                       # générées x réelles
    precision = (dgr <= r3[None]).any(1).float().mean().item()
    rappel = (dgr.T <= g3[None]).any(1).float().mean().item()
    densite = (dgr <= r5[None]).float().sum(1).div(5).mean().item()
    couverture = (dgr.min(0).values <= r5).float().mean().item()
    return dict(precision=precision, rappel=rappel, densite=densite, couverture=couverture)


@torch.no_grad()
def classes(x, etiquettes_reelles, device):
    import encodeurs_figes_nous
    m = encodeurs_figes_nous.charger_classifieur(device)
    p = torch.cat([m(x[i:i + 1000].to(device)).softmax(1).cpu() for i in range(0, len(x), 1000)])
    prop = torch.bincount(p.argmax(1), minlength=10).double() / len(x)
    vraie = torch.bincount(etiquettes_reelles, minlength=10).double() / len(etiquettes_reelles)
    return dict(proportions_classes=prop.tolist(), ecart_classes=0.5 * (prop - vraie).abs().sum().item(),
                confiance=p.max(1).values.mean().item())


def mesurer(x_gen, data_root, device, cache=None):
    """x_gen : (N, 1, 28, 28) dans [-1, 1]. cache : fichier .pt pour les caractéristiques réelles."""
    from pytorch_fid.fid_score import calculate_frechet_distance
    x_vrai, etiquettes = test_mnist(data_root)
    if cache and os.path.exists(cache):
        f_vrai = torch.load(cache)
    else:
        f_vrai = inception(x_vrai, device)
        if cache:
            torch.save(f_vrai, cache)
    f_gen = inception(x_gen, device)
    mu1, mu2 = f_gen.mean(0).numpy(), f_vrai.mean(0).numpy()
    s1, s2 = np.cov(f_gen.numpy(), rowvar=False), np.cov(f_vrai.numpy(), rowvar=False)
    d = x_gen[0].numel()
    r = dict(fid_10k=float(calculate_frechet_distance(mu1, s1, mu2, s2)),
             sw_pixels=sliced_w2(x_gen.flatten(1).double() / math.sqrt(d),
                                 x_vrai.flatten(1).double() / math.sqrt(d), device),
             sw_inception=sliced_w2(f_gen, f_vrai, device))
    r.update(voisinages(f_vrai.float(), f_gen.float(), device))
    r.update(classes(x_gen, etiquettes, device))
    return r


def fin_de_run(netT, noise_sampler, device, exp_path, data_root, cache):
    x = generer(netT, noise_sampler, device)
    np.savez_compressed(os.path.join(exp_path, "echantillons_10k.npz"), images=vers_uint8(x))
    r = mesurer(depuis_uint8(vers_uint8(x)), data_root, device, cache)
    with open(os.path.join(exp_path, "mesures.json"), "w") as f:
        json.dump(r, f, indent=2)
    return r


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("fichier", help="echantillons_10k.npz, ou 'vraies' pour la référence train/test")
    ap.add_argument("--data_root", default="../data/mnist")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.fichier == "vraies":
        from torchvision import datasets
        ds = datasets.MNIST(a.data_root, train=True, download=True)
        x = ds.data[:N].float().div(127.5).sub(1).unsqueeze(1)
    else:
        x = depuis_uint8(np.load(a.fichier)["images"])
    print(json.dumps(mesurer(x, a.data_root, dev), indent=2))

"""Les deux réseaux figés de MNIST (copié en gwf/encodeurs_figes_nous.py).

Classifieur : petit classifieur MNIST. Sert d'encodeur figé (« classifieur figé », avant-dernière
couche, 128 dimensions) et à compter les chiffres générés dans les mesures. Entraîné une fois,
poids dans classifieur_mnist.pt :
    python encodeurs_figes_nous.py --data_root ../data/mnist

EncodeurMasque : « encodeur fixe (style Drifting) ». Drifting (Deng et al. 2026) calcule sa perte
dans un réseau pré-entraîné sans étiquettes, à chaque étage, et additionne les pertes. Version
MNIST : un petit autoencodeur masqué (on cache la moitié des carrés 4x4, il doit reconstruire
l'image entière). Trois étages : 32 x 14 x 14, 64 x 7 x 7, 128 x 4 x 4. Entraîné au début de
chaque run (environ une minute sur T4, graine fixe), puis figé.
Images 28x28 dans [-1, 1], comme le générateur.
"""
import argparse
import os

import torch
from torch import nn
import torch.nn.functional as F

POIDS_CLASSIFIEUR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifieur_mnist.pt")


class Classifieur(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.f1 = nn.Linear(128 * 3 * 3, 128)
        self.f2 = nn.Linear(128, 10)

    def caracteristiques(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)   # 14
        x = F.max_pool2d(F.relu(self.c2(x)), 2)   # 7
        x = F.max_pool2d(F.relu(self.c3(x)), 2)   # 3
        return F.relu(self.f1(x.flatten(1)))

    def forward(self, x):
        return self.f2(self.caracteristiques(x))


def charger_classifieur(device):
    m = Classifieur().to(device)
    m.load_state_dict(torch.load(POIDS_CLASSIFIEUR, map_location=device))
    return m.eval().requires_grad_(False)


class EncodeurMasque(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.GELU())
        self.e2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.GELU())
        self.e3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.GELU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, 2, 1), nn.GELU(),
                                 nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
                                 nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh())

    def etages(self, x):
        h1 = self.e1(x)
        h2 = self.e2(h1)
        return [h1, h2, self.e3(h2)]

    def forward(self, x):
        return self.dec(self.etages(x)[-1])


def masquer(x, taille=4, proportion=.5):
    b, _, n, _ = x.shape
    garde = (torch.rand(b, 1, n // taille, n // taille, device=x.device) > proportion).float()
    garde = garde.repeat_interleave(taille, 2).repeat_interleave(taille, 3)
    return x * garde - (1 - garde)          # fond = -1 (noir)


def entrainer_encodeur(data_sampler, device, iterations=3000, graine=0, journal=None):
    # Batch 256 : bloqué sur « tout noir » (erreur 0,44) jusqu'à ~2000 itérations, puis 0,05.
    etat = torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(graine)
    enc = EncodeurMasque().to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=2e-3)
    for i in range(iterations):
        x = data_sampler.sample().to(device)
        perte = (enc(masquer(x)) - x).square().mean()
        opt.zero_grad(set_to_none=True)
        perte.backward()
        opt.step()
        if journal and (i + 1) % 500 == 0:
            journal(f"[ENCODEUR] itération {i + 1}, erreur de reconstruction {perte.item():.4f}")
    torch.set_rng_state(etat[0])
    if etat[1] is not None:
        torch.cuda.set_rng_state_all(etat[1])
    return enc.eval().requires_grad_(False)


if __name__ == "__main__":   # entraîne le classifieur
    from torchvision import datasets, transforms
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../data/mnist")
    ap.add_argument("--epoques", type=int, default=3)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    app = torch.utils.data.DataLoader(datasets.MNIST(a.data_root, True, tf, download=True),
                                      batch_size=128, shuffle=True, num_workers=2)
    test = torch.utils.data.DataLoader(datasets.MNIST(a.data_root, False, tf, download=True),
                                       batch_size=1000)
    m = Classifieur().to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for e in range(a.epoques):
        m.train()
        for x, y in app:
            x, y = x.to(dev), y.to(dev)
            x = x + 0.1 * torch.randn_like(x)  # un peu de bruit : robuste aux images floues
            loss = F.cross_entropy(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            juste = sum((m(x.to(dev)).argmax(1) == y.to(dev)).sum().item() for x, y in test)
        print(f"époque {e}: précision test {juste / 100:.2f} %", flush=True)
    torch.save(m.state_dict(), POIDS_CLASSIFIEUR)

"""Banc Sinkhorn : trois énergies x trois générateurs, sur deux cibles 2D.

    python sinkhorn.py valider                 # contrôles mathématiques (POT), ~1 min
    python sinkhorn.py calculer                # les 18 conditions (plusieurs heures)
    python sinkhorn.py calculer --cible lunes --energie direct --generateur splines

Résultat : resultats/sinkhorn.pt, puis figures/sinkhorn_<cible>.png.
Une condition déjà présente dans resultats/sinkhorn.pt n'est pas recalculée.

Énergies (coût c = ||x-y||²/2, epsilon = 0,05) :
  ot        OT_eps(mu, nu), semi-dual, un potentiel neuronal g.
  sinkhorn  S_eps(mu, nu) = OT_eps(mu,nu) - OT_eps(mu,mu)/2 - OT_eps(nu,nu)/2,
            formule à deux potentiels neuronaux g et s (preuve C du rapport).
  direct    S_eps calculé exactement sur chaque lot par l'algorithme de Sinkhorn,
            sans aucun réseau critique (gradient d'enveloppe).
La constante -OT_eps(nu,nu)/2 est omise : elle ne dépend pas du générateur.
Pas de pénalité R1 ici, 20 pas JKO de 500 itérations.
"""
import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from commun import MLP, cible, distances_carrees, rng, source, sw2

ENERGIES = ("ot", "sinkhorn", "direct")
GENERATEURS = ("mlp", "residuel", "splines")
PARAMETRES = dict(seed=2026, etapes=20, iterations=500, batch=256, tau=1.,
                  epsilon=.05, lr_generateur=2e-4, lr_potentiels=1e-4,
                  betas=(.5, .9), eps_adam=1e-8, pas_potentiels=1,
                  tolerance_sinkhorn=1e-3, iterations_sinkhorn_max=1200)
RESULTATS = Path(__file__).resolve().parent / "resultats"


# ----------------------------------------------------------------------------
# Énergies
# ----------------------------------------------------------------------------

def cout(x, y):
    return .5 * distances_carrees(x, y)


def logmoyenne(a, dim):
    return torch.logsumexp(a, dim=dim) - math.log(a.shape[dim])


def variation(x, y, g, s, epsilon):
    """Objectif variationnel évalué aux potentiels g (sur y) et s (sur x).

    s = None : semi-dual de OT_eps(mu, nu).
    s donné  : S_eps(mu, nu) + OT_eps(nu,nu)/2, formule à deux potentiels.
    Les doubles sommes d'autotransport incluent les diagonales.
    """
    g = g.flatten() - g.mean()  # Chaque potentiel a une jauge additive.
    a = logmoyenne((g[None] - cout(x, y)) / epsilon, 1)
    valeur = -epsilon * a.mean() + g.mean()
    if s is not None:
        s = s.flatten() - s.mean()
        cxx = cout(x, x)
        b = logmoyenne((s[None] - cxx) / epsilon, 1)
        q = torch.logsumexp(((s[:, None] + s[None] - cxx) / epsilon).flatten(), 0)
        q = q - 2 * math.log(len(x))
        valeur = valeur + epsilon * (b.mean() - .5 * q)
    return valeur


@torch.no_grad()
def potentiels_exacts(c, epsilon, tolerance=1e-3, maximum=1200):
    """Sinkhorn en log-domaine ; arrêt sur l'erreur relative des deux marginales."""
    g = torch.zeros(c.shape[1], dtype=c.dtype, device=c.device)
    erreur = float("inf")
    for iteration in range(1, maximum + 1):
        f = -epsilon * logmoyenne((g[None] - c) / epsilon, 1)
        g = -epsilon * logmoyenne((f[:, None] - c) / epsilon, 0)
        if iteration % 20 == 0:
            log_lignes = f / epsilon + logmoyenne((g[None] - c) / epsilon, 1)
            log_colonnes = g / epsilon + logmoyenne((f[:, None] - c) / epsilon, 0)
            erreur = max(log_lignes.expm1().abs().max().item(),
                         log_colonnes.expm1().abs().max().item())
            if erreur <= tolerance:
                return f, g, erreur, iteration
    raise RuntimeError(f"Sinkhorn non convergé : résidu={erreur:.3g}, max={maximum}")


def dual_complet(c, f, g, epsilon):
    # Potentiels détachés : dérivée d'enveloppe par rapport aux positions.
    return f.mean() + g.mean() - epsilon * (
        torch.exp((f[:, None] + g[None] - c) / epsilon).mean() - 1)


def transport_direct(x, y, epsilon, tolerance=1e-3, maximum=1200):
    c = cout(x, y)
    f, g, erreur, iteration = potentiels_exacts(c.detach(), epsilon, tolerance, maximum)
    return dual_complet(c, f, g, epsilon), erreur, iteration


def energie_directe(x, y, epsilon, tolerance=1e-3, maximum=1200, debiaisee=True):
    xy, erreur, iteration = transport_direct(x, y, epsilon, tolerance, maximum)
    if debiaisee:
        xx, erreur_x, iteration_x = transport_direct(x, x, epsilon, tolerance, maximum)
        return xy - .5 * xx, max(erreur, erreur_x), max(iteration, iteration_x)
    return xy, erreur, iteration


# ----------------------------------------------------------------------------
# Générateurs
# ----------------------------------------------------------------------------

class Residuel(nn.Module):
    """Quatre blocs u -> u + h(u)/4, h : 2-128-128-2 SiLU. Presque l'identité au départ."""
    def __init__(self):
        super().__init__()
        self.blocs = nn.ModuleList([
            nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128),
                          nn.SiLU(), nn.Linear(128, 2)) for _ in range(4)])
        for bloc in self.blocs:
            nn.init.normal_(bloc[-1].weight, std=1e-3)
            nn.init.zeros_(bloc[-1].bias)

    def forward(self, z):
        for bloc in self.blocs:
            z = z + .25 * bloc(z)
        return z


class Splines(nn.Module):
    """Flot normalisant : six couplages à splines rationnelles quadratiques.

    Chaque couplage garde une coordonnée fixe et transforme l'autre par une
    fonction monotone à 8 morceaux, dont les paramètres sont donnés par un petit
    réseau 1-64-64-23 de la coordonnée fixe. Queues identité au-delà de +/-4.
    Initialisation : l'identité. Spline : nflows (Durkan et al., 2019).
    """
    def __init__(self):
        super().__init__()
        self.reseaux = nn.ModuleList([
            nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64),
                          nn.SiLU(), nn.Linear(64, 23)) for _ in range(6)])
        for reseau in self.reseaux:
            nn.init.zeros_(reseau[-1].weight)
            nn.init.zeros_(reseau[-1].bias)
            with torch.no_grad():
                reseau[-1].bias[16:] = math.log(math.expm1(1 - 1e-3))

    def transformation(self, z, inverse=False):
        from nflows.transforms.splines.rational_quadratic import unconstrained_rational_quadratic_spline
        logdet = z.new_zeros(len(z))
        indices = reversed(range(6)) if inverse else range(6)
        for indice in indices:
            fixe = indice % 2
            parametres = self.reseaux[indice](z[:, fixe:fixe + 1])
            v, ld = unconstrained_rational_quadratic_spline(
                z[:, 1 - fixe], parametres[:, :8], parametres[:, 8:16],
                parametres[:, 16:], inverse=inverse, tails="linear", tail_bound=4.)
            z = torch.stack((z[:, 0] if fixe == 0 else v,
                             z[:, 1] if fixe == 1 else v), 1)
            logdet = logdet + ld
        return z, logdet

    def forward(self, z):
        return self.transformation(z)[0]


def creer_generateur(nom):
    return {"mlp": lambda: MLP(2, 128), "residuel": Residuel, "splines": Splines}[nom]()


# ----------------------------------------------------------------------------
# Entraînement
# ----------------------------------------------------------------------------

def entrainer(nom_cible, energie, generateur_nom, p=PARAMETRES, device="cuda"):
    torch.manual_seed(p["seed"])
    generateur = creer_generateur(generateur_nom).to(device)
    torch.manual_seed(p["seed"] + 1)
    g = MLP(1, 128).to(device) if energie != "direct" else None
    torch.manual_seed(p["seed"] + 2)
    s = MLP(1, 128).to(device) if energie == "sinkhorn" else None
    optim_t = torch.optim.Adam(generateur.parameters(), lr=p["lr_generateur"],
                               betas=p["betas"], eps=p["eps_adam"])
    potentiels = [] if g is None else list(g.parameters()) + ([] if s is None else list(s.parameters()))
    optim_d = torch.optim.Adam(potentiels, lr=p["lr_potentiels"], betas=p["betas"],
                               eps=p["eps_adam"]) if potentiels else None
    flux = [rng(device, p["seed"] + j) for j in (10, 11, 12, 13)]
    precedent = None  # Au premier pas JKO, la référence est l'identité.
    debut = time.perf_counter()
    for k in range(p["etapes"]):
        for _ in range(p["iterations"]):
            if g is not None:
                # 1. Montée des potentiels, générateur figé.
                generateur.requires_grad_(False)
                g.requires_grad_(True)
                if s is not None:
                    s.requires_grad_(True)
                for _ in range(p["pas_potentiels"]):
                    z = source(p["batch"], flux[0], device)
                    y = cible(p["batch"], flux[1], device, nom_cible)
                    with torch.no_grad():
                        x = generateur(z)
                    valeur = variation(x, y, g(y), None if s is None else s(x), p["epsilon"])
                    optim_d.zero_grad(set_to_none=True)
                    (-valeur).backward()
                    optim_d.step()
                g.requires_grad_(False)
                if s is not None:
                    s.requires_grad_(False)
            # 2. Descente du générateur, nouveaux tirages.
            generateur.requires_grad_(True)
            z = source(p["batch"], flux[2], device)
            y = cible(p["batch"], flux[3], device, nom_cible)
            x = generateur(z)
            if energie == "direct":
                valeur, _, _ = energie_directe(x, y, p["epsilon"], p["tolerance_sinkhorn"],
                                               p["iterations_sinkhorn_max"])
            else:
                valeur = variation(x, y, g(y), None if s is None else s(x), p["epsilon"])
            with torch.no_grad():
                reference = z if precedent is None else precedent(z)
            perte = valeur + (x - reference).square().sum(1).mean() / (2 * p["tau"])
            if not torch.isfinite(perte):
                raise FloatingPointError(f"Perte non finie : {nom_cible}, {energie}, {generateur_nom}, JKO {k}")
            optim_t.zero_grad(set_to_none=True)
            perte.backward()
            optim_t.step()
        precedent = copy.deepcopy(generateur).requires_grad_(False).eval()
        print(f"{nom_cible} | {energie} | {generateur_nom} | JKO {k + 1}/{p['etapes']} | "
              f"{time.perf_counter() - debut:.0f}s", flush=True)
    secondes = time.perf_counter() - debut

    with torch.no_grad():
        points = generateur(source(8192, rng(device, 21037), device)).cpu()
    y = cible(8192, rng("cpu", 21038), "cpu", nom_cible)
    if not torch.isfinite(points).all():
        raise FloatingPointError("Échantillons non finis")
    sortie = dict(points=points, sw2=sw2(points, y), secondes=secondes)
    if g is not None:
        # Diagnostic ponctuel : gradient appris vs gradient de Sinkhorn exact,
        # sur 128 points générés (1 = même direction, 0 = orthogonaux).
        x = points[:128].to(device).detach().clone().requires_grad_(True)
        yv = y[:128].to(device)
        v = variation(x, yv, g(yv), None if s is None else s(x), p["epsilon"])
        vrai, _, _ = energie_directe(x.double(), yv.double(), p["epsilon"], 1e-5, 6000,
                                     debiaisee=energie == "sinkhorn")
        gv, = torch.autograd.grad(v, x)
        ge, = torch.autograd.grad(vrai, x)
        sortie["cosinus_gradient"] = float(nn.functional.cosine_similarity(
            gv.flatten(), ge.flatten().float(), dim=0))
    return sortie


# ----------------------------------------------------------------------------
# Contrôles mathématiques
# ----------------------------------------------------------------------------

def valider():
    """Compare valeurs et gradients à un calcul primal indépendant (POT), en double précision."""
    import ot
    from commun import cible as tirer_cible
    torch.manual_seed(491)
    epsilon = .05
    for nom in ("huit_gaussiennes", "lunes"):
        x = source(16, rng("cpu", 45), "cpu").double().requires_grad_(True)
        y = tirer_cible(19, rng("cpu", 46), "cpu", nom).double()
        references = []
        for a, b in ((x, y), (x, x)):
            c = cout(a.detach(), b.detach()).numpy()
            pa, pb = np.full(len(a), 1 / len(a)), np.full(len(b), 1 / len(b))
            plan = ot.sinkhorn(pa, pb, c, reg=epsilon, method="sinkhorn_log",
                               numItermax=30000, stopThr=1e-13)
            valeur = (plan * c).sum() + epsilon * (plan * np.log(
                np.maximum(plan, 1e-300) / (pa[:, None] * pb[None]))).sum()
            references.append((valeur, torch.tensor(plan)))
        primal = references[0][0] - .5 * references[1][0]
        grad_ref = (references[0][1][..., None] * (x.detach()[:, None] - y[None])).sum(1)
        grad_ref -= (references[1][1][..., None] * (x.detach()[:, None] - x.detach()[None])).sum(1)
        # Sinkhorn direct.
        direct, _, _ = energie_directe(x, y, epsilon, 1e-11, 30000)
        grad_direct, = torch.autograd.grad(direct, x)
        # Formule à deux potentiels, aux potentiels optimaux.
        _, g, _, _ = potentiels_exacts(cout(x.detach(), y), epsilon, 1e-12, 30000)
        p1, p2, _, _ = potentiels_exacts(cout(x.detach(), x.detach()), epsilon, 1e-12, 30000)
        v = variation(x, y, g, .5 * (p1 + p2), epsilon)
        grad_v, = torch.autograd.grad(v, x)
        # Le supremum est aussi retrouvé en optimisant librement g et s.
        gv, sv = nn.Parameter(torch.zeros_like(g)), nn.Parameter(torch.zeros_like(p1))
        lbfgs = torch.optim.LBFGS([gv, sv], max_iter=1200, tolerance_grad=1e-11,
                                  tolerance_change=1e-14, line_search_fn="strong_wolfe")
        def fermeture():
            lbfgs.zero_grad()
            perte = -variation(x.detach(), y, gv, sv, epsilon)
            perte.backward()
            return perte
        lbfgs.step(fermeture)
        libre = variation(x.detach(), y, gv, sv, epsilon)
        erreurs = dict(valeur_direct=abs(direct.item() - primal),
                       gradient_direct=(grad_direct - grad_ref).abs().max().item(),
                       valeur_formule=abs(v.item() - primal),
                       gradient_formule=(grad_v - grad_ref).abs().max().item(),
                       sup_libre=abs(libre.item() - primal))
        print(nom, {k: f"{e:.1e}" for k, e in erreurs.items()})
        assert max(erreurs.values()) < 2e-6, erreurs
    # S_eps(mu, mu) = 0 et gradient nul.
    x = (.3 * torch.randn(18, 2, dtype=torch.float64)).requires_grad_(True)
    divergence = (transport_direct(x, x.detach(), epsilon, 1e-11, 30000)[0]
                  - .5 * transport_direct(x, x, epsilon, 1e-11, 30000)[0]
                  - .5 * transport_direct(x.detach(), x.detach(), epsilon, 1e-11, 30000)[0])
    gradient, = torch.autograd.grad(divergence, x)
    assert abs(divergence.item()) < 1e-8 and gradient.abs().max().item() < 1e-7
    print("Tous les contrôles passent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("valider", "calculer"))
    parser.add_argument("--cible", choices=("huit_gaussiennes", "lunes"))
    parser.add_argument("--energie", choices=ENERGIES)
    parser.add_argument("--generateur", choices=GENERATEURS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "valider":
        valider()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chemin = RESULTATS / "sinkhorn.pt"
        donnees = torch.load(chemin, weights_only=True) if chemin.exists() else dict(parametres=PARAMETRES, conditions={})
        for nom in [args.cible] if args.cible else ("huit_gaussiennes", "lunes"):
            for energie in [args.energie] if args.energie else ENERGIES:
                for gen in [args.generateur] if args.generateur else GENERATEURS:
                    cle = f"{nom}/{energie}/{gen}"
                    if cle in donnees["conditions"]:
                        print(f"{cle} déjà calculé, ignoré.")
                        continue
                    donnees["conditions"][cle] = entrainer(nom, energie, gen, device=device)
                    RESULTATS.mkdir(exist_ok=True)
                    torch.save(donnees, chemin)
        import figures
        figures.sinkhorn()

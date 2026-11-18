"""Sinkhorn sur MNIST, branché dans le train.py du papier (copié en gwf/energies_nous.py).

Une seule divergence ajoutée, « Sinkhorn » : S_eps(mu, nu) calculée par itérations de Sinkhorn
sur chaque batch, sans critique (même formule qu'en 2D, sinkhorn.py). Le coût est
c(x, y) = ½||phi(x) - phi(y)||², et --espace choisit phi :

  pixels                 phi(x) = x / sqrt(d) : le coût du terme JKO du papier
  encodeur_adversarial   phi a l'architecture du critique du MMD-GAN et est appris pour
                         maximiser S_eps (exactement comme le noyau appris du MMD-GAN)
  encodeur_fixe          autoencodeur masqué appris sans étiquettes puis figé, lu à trois
                         échelles, une S_eps par échelle (style Drifting ; mnist_encodeurs_figes.py)
  encodeur_mmd_fige      critique final d'un MMD-GAN déjà entraîné (--encodeur_ckpt), figé
  classifieur_fige       avant-dernière couche d'un petit classifieur MNIST, figé
  hm1                    norme de Sobolev négative H^-1 (linéarisation de W2 entre images)
  dukler                 « Wasserstein of Wasserstein » (Dukler et al. 2019) : chaque image est une
                         masse sur la grille 28x28, et c(x, y) est la divergence de Sinkhorn entre
                         les deux masses (Sinkhorn par convolution). Pas de réseau.

Les encodeurs sont normalisés sur la sphère. Le terme JKO reste ½||x - y||² sur les pixels.
La pénalité de Dukler du WGAN (« WGAN + Dukler ») est penalite_dukler plus bas.
"""
import math

import torch

NOS_DIVERGENCES = ("Sinkhorn",)
ESPACES = ("pixels", "encodeur_adversarial", "encodeur_fixe", "encodeur_mmd_fige",
           "classifieur_fige", "hm1", "dukler")
PHI = {}

# Pire résidu de Sinkhorn depuis le dernier affichage (diagnostic, voir train.py).
SUIVI = dict(residu_max=0.0, non_converge=0)


def aplatir(x):
    x = x.flatten(1)
    return x / math.sqrt(x.shape[1])


def cout(x, y):
    return .5 * (x.square().sum(1, keepdim=True) + y.square().sum(1)[None]
                 - 2 * x @ y.T).clamp_min(0)


def logmoyenne(a, dim):
    return torch.logsumexp(a, dim=dim) - math.log(a.shape[dim])


@torch.no_grad()
def potentiels_exacts(c, epsilon, tolerance, maximum):
    """Sinkhorn log-domaine. Un non-convergé n'arrête pas le run : on garde les derniers
    potentiels et on le compte dans SUIVI."""
    g = torch.zeros(c.shape[1], dtype=c.dtype, device=c.device)
    erreur = float("inf")
    for iteration in range(1, maximum + 1):
        f = -epsilon * logmoyenne((g[None] - c) / epsilon, 1)
        g = -epsilon * logmoyenne((f[:, None] - c) / epsilon, 0)
        if iteration % 20 == 0 or iteration == maximum:
            log_lignes = f / epsilon + logmoyenne((g[None] - c) / epsilon, 1)
            log_colonnes = g / epsilon + logmoyenne((f[:, None] - c) / epsilon, 0)
            erreur = max(log_lignes.expm1().abs().max().item(),
                         log_colonnes.expm1().abs().max().item())
            if erreur <= tolerance:
                break
    SUIVI["residu_max"] = max(SUIVI["residu_max"], erreur)
    if erreur > tolerance:
        SUIVI["non_converge"] += 1
    return f, g


def transport_cout(c, epsilon, tolerance, maximum):
    """OT_eps à partir de la matrice de coût ; gradient par le théorème de l'enveloppe."""
    f, g = potentiels_exacts(c.detach(), epsilon, tolerance, maximum)
    return f.mean() + g.mean() - epsilon * (
        torch.exp((f[:, None] + g[None] - c) / epsilon).mean() - 1)


def transport(x, y, epsilon, tolerance, maximum):
    return transport_cout(cout(x, y), epsilon, tolerance, maximum)


def sphere(f):
    return f / f.norm(dim=1, keepdim=True).clamp_min(1e-8)


# ----- espace hm1 -----
def poids_hm1(n, device):
    k = torch.arange(n, device=device, dtype=torch.float32)
    lam = 4 * torch.sin(math.pi * k / n) ** 2              # valeurs propres du laplacien discret
    return (lam[:, None] + lam[None] + 4 * math.sin(math.pi / n) ** 2).rsqrt()


def hm1(images):
    return torch.fft.ifft2(torch.fft.fft2(images) * PHI["poids_hm1"]).real


# ----- espace dukler : transport entre les pixels de deux images -----
def masses(images, delta=1e-2):
    """Image dans [-1, 1] -> masse de probabilité sur la grille (plancher delta, comme Dukler),
    et encre moyenne (la normalisation fait perdre la quantité d'encre, on la garde à part)."""
    m = (images[:, 0] + 1) / 2 + delta
    total = m.flatten(1).sum(1)
    return m / total[:, None, None], total / m[0].numel()


def noyau_grille(n, eta, device):
    t = torch.arange(n, device=device, dtype=torch.float32) / (n - 1)
    return torch.exp(-(t[:, None] - t[None]) ** 2 / eta)   # séparable : K_2D v = K v K


@torch.no_grad()
def potentiels_grille(a, b, noyau, eta, iterations):
    """Sinkhorn par convolution entre P paires de masses (P, n, n), coût |p - q|² sur [0, 1]²."""
    v = torch.ones_like(b)
    for _ in range(iterations):
        u = a / (noyau @ v @ noyau).clamp_min(1e-35)
        v = b / (noyau @ u @ noyau).clamp_min(1e-35)
    return eta * u.clamp_min(1e-35).log(), eta * v.clamp_min(1e-35).log()


def ot_grille(a, b):
    """OT_eta(a_i, b_j) pour toutes les paires : matrice (len(a), len(b)). Gradient par le théorème
    de l'enveloppe : potentiels figés, seules les masses portent le gradient."""
    na, nb, n = len(a), len(b), a.shape[-1]
    aa = a[:, None].expand(na, nb, n, n).reshape(-1, n, n)
    bb = b[None].expand(na, nb, n, n).reshape(-1, n, n)
    f, g = potentiels_grille(aa.detach(), bb.detach(), PHI["noyau_grille"], PHI["eta"], PHI["iter_grille"])
    return ((aa * f).flatten(1).sum(1) + (bb * g).flatten(1).sum(1)).view(na, nb)


def ot_grille_diag(a):
    f, g = potentiels_grille(a.detach(), a.detach(), PHI["noyau_grille"], PHI["eta"], PHI["iter_grille"])
    return (a * f).flatten(1).sum(1) + (a * g).flatten(1).sum(1)


def cout_dukler(x_images, y_images):
    """c(x, y) = S_eta(x, y) + ½ (encre x - encre y)², remis à l'échelle du coût en pixels."""
    a, ea = masses(x_images)
    b, eb = masses(y_images)
    s = ot_grille(a, b) - .5 * ot_grille_diag(a)[:, None] - .5 * ot_grille_diag(b)[None]
    return (s.clamp_min(0) + .5 * (ea[:, None] - eb[None]) ** 2) * PHI["echelle_dukler"]


# ----- branchement dans train.py -----
def sans_critique(args):
    """Seul l'encodeur adversarial s'entraîne comme un critique."""
    return args.divergence == "Sinkhorn" and args.espace != "encodeur_adversarial"


@torch.no_grad()
def preparer(args, device, data_sampler):
    """Construit l'encodeur figé éventuel (une fois, avant l'entraînement)."""
    if args.espace == "classifieur_fige":
        import encodeurs_figes_nous
        PHI["enc"] = encodeurs_figes_nous.charger_classifieur(device).caracteristiques
    elif args.espace == "encodeur_mmd_fige":
        import copy
        from networks import get_model
        a = copy.copy(args)
        a.model_name, a.embed_dim, a.use_sn = "resnet_MMDGAN", 128, False
        d = get_model(a)[1].to(device)
        d.load_state_dict(torch.load(args.encodeur_ckpt, map_location=device))
        PHI["enc"] = d.eval().requires_grad_(False)
    elif args.espace == "encodeur_fixe":
        import encodeurs_figes_nous
        with torch.enable_grad():   # preparer est sous no_grad
            PHI["enc"] = encodeurs_figes_nous.entrainer_encodeur(data_sampler, device,
                                                                 journal=PHI.get("journal"))
    elif args.espace == "hm1":
        x = data_sampler.sample().to(device)
        PHI["poids_hm1"] = poids_hm1(x.shape[-1], device)
        # Échelle fixée une fois : même coût moyen entre vraies images qu'en pixels,
        # pour que epsilon garde le même sens.
        p, h = aplatir(x), aplatir(hm1(x))
        PHI["echelle_hm1"] = math.sqrt(cout(p, p).mean().item() / cout(h, h).mean().item())
    elif args.espace == "dukler":
        x = data_sampler.sample().to(device)[:64]
        PHI.update(noyau_grille=noyau_grille(x.shape[-1], args.eta_grille, device),
                   eta=args.eta_grille, iter_grille=args.iter_grille, echelle_dukler=1.0)
        p = aplatir(x)   # même règle d'échelle que hm1
        PHI["echelle_dukler"] = cout(p, p).mean().item() / cout_dukler(x, x).mean().item()


def coordonnees(args, images, sortie_critique=None):
    if args.espace == "pixels":
        return aplatir(images)
    if args.espace == "encodeur_adversarial":
        return sphere(sortie_critique)
    if args.espace == "hm1":
        return aplatir(hm1(images)) * PHI["echelle_hm1"]
    return sphere(PHI["enc"](images).flatten(1))       # encodeur_mmd_fige, classifieur_fige


def energie(args, fake_images, real_images, d_real, d_fake, critique=False):
    """S_eps entre images générées et vraies ; maximisée par l'encodeur adversarial,
    minimisée par le générateur."""
    eps, tol, nmax = args.eps_sinkhorn, args.tol_sinkhorn, args.iter_sinkhorn
    if args.espace == "dukler":
        return (transport_cout(cout_dukler(fake_images, real_images), eps, tol, nmax)
                - .5 * transport_cout(cout_dukler(fake_images, fake_images), eps, tol, nmax))
    if args.espace == "encodeur_fixe":   # une S_eps par échelle, additionnées
        valeur = 0.
        for hx, hy in zip(PHI["enc"].etages(fake_images), PHI["enc"].etages(real_images)):
            x, y = sphere(hx.flatten(1)), sphere(hy.flatten(1)).detach()
            valeur = valeur + transport(x, y, eps, tol, nmax) - .5 * transport(x, x, eps, tol, nmax)
        return valeur
    x = coordonnees(args, fake_images, d_fake)
    y = coordonnees(args, real_images, d_real)
    if not critique:
        y = y.detach()
    valeur = transport(x, y, eps, tol, nmax) - .5 * transport(x, x, eps, tol, nmax)
    if critique:  # l'encodeur voit aussi -OT(nu, nu)/2, qui dépend de phi
        valeur = valeur - .5 * transport(y, y, eps, tol, nmax)
    return valeur


def penalite_dukler(netD, real_images, fake_images, alpha=1.0, delta=1e-2, cible=1.0):
    """Pénalité de gradient de Dukler et al. (ICML 2019) : la norme du gradient du critique est
    mesurée dans la géométrie W2 de l'espace des images, ||grad f||²_W = grad f^T L(X) grad f, avec
    L(X) le laplacien de la grille des pixels pondéré par l'intensité de X (plus alpha 11^T pour la
    masse totale). Calcul par différences finies, l'équivalent de leurs convolutions.
    cible 1 : WGAN + Dukler. cible 0 : pénalité centrée en zéro, pour l'encodeur adversarial."""
    b = real_images.size(0)
    a = torch.rand([b] + [1] * (real_images.dim() - 1), device=real_images.device)
    xi = (a * real_images + (1 - a) * fake_images).detach().requires_grad_(True)
    g, = torch.autograd.grad(netD(xi).sum(), xi, create_graph=True)
    m = (xi.detach() + 1) / 2 + delta                        # intensité >= delta
    wx, wy = (m[..., 1:] + m[..., :-1]) / 2, (m[..., 1:, :] + m[..., :-1, :]) / 2
    dx, dy = g[..., 1:] - g[..., :-1], g[..., 1:, :] - g[..., :-1, :]
    n2 = ((wx * dx ** 2).flatten(1).sum(1) + (wy * dy ** 2).flatten(1).sum(1)
          + alpha * g.flatten(1).sum(1) ** 2)
    return ((n2 + 1e-12).sqrt() - cible) ** 2

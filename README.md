## Contenu

| Fichier | Rôle |
|---|---|
| `commun.py` | 2D : lois (source, cibles), MLP, MMD de Riesz, mesures |
| `banc_simple.py` | 2D, banc 1 : cinq énergies, générateur MLP |
| `sinkhorn.py` | 2D, banc 2 : trois énergies de transport × trois générateurs, plus les contrôles mathématiques |
| `mnist_energies.py` | MNIST : les mêmes énergies, à brancher dans le code du papier |
| `mnist_gwf.patch` | MNIST : nos modifications du `train.py` du papier |
| `mnist_kaggle.py` | MNIST : lance les runs sur Kaggle, les récupère, les résume dans `resultats/` |
| `figures.py` | Redessine toutes les figures depuis `resultats/` sans réentraîner |
| `resultats/` | Points générés finaux en 2D (`*.pt`) ; FID, temps et échantillons MNIST |
| `figures/` | Les figures finales |

## Relancer

```bash
python figures.py              # toutes les figures, quelques secondes
python sinkhorn.py valider     # contrôles contre POT, ~1 min
python banc_simple.py          # ~40 min sur GPU
python sinkhorn.py calculer    # plusieurs heures ; ignore les conditions déjà dans resultats/sinkhorn.pt
```

MNIST (code du papier patché dans `gwf/`, non versionné ; calcul sur Kaggle) :

```bash
git clone https://github.com/Paulcauch/Generative_Wasserstein_Flows.git gwf
cd gwf && git checkout 6633b55 && git apply ../mnist_gwf.patch && cp ../mnist_energies.py energies_nous.py && cd ..
python mnist_kaggle.py push --run-id n1    # puis status, pull ; pareil pour n2
python mnist_kaggle.py resumer             # -> resultats/mnist.json et mnist_echantillons.npz
```

Dépendances : `torch`, `matplotlib`, `numpy`, plus `nflows` (générateur à splines) et `pot`
(seulement pour `valider`) ; pour Kaggle, `kaggle` et `truststore`. Le venv
`Documents/estimation_densite/.venv` a tout ce qu'il faut pour la 2D.

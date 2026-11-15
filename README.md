## Contenu

| Fichier | Rôle |
|---|---|
| `commun.py` | Lois (source, cibles), MLP, MMD de Riesz, mesures |
| `banc_simple.py` | Banc 1 : cinq énergies, générateur MLP |
| `sinkhorn.py` | Banc 2 : trois énergies de transport × trois générateurs, plus les contrôles mathématiques |
| `figures.py` | Redessine les figures depuis `resultats/` sans réentraîner |
| `resultats/*.pt` | Points générés finaux (8 192 par condition) et réglages |
| `figures/` | Les trois figures finales |

## Relancer

```bash
python figures.py              # figures + tableau de mesures, quelques secondes
python sinkhorn.py valider     # contrôles contre POT, ~1 min
python banc_simple.py          # ~40 min sur GPU
python sinkhorn.py calculer    # plusieurs heures ; ignore les conditions déjà dans resultats/sinkhorn.pt
```

Dépendances : `torch`, `matplotlib`, `numpy`, plus `nflows` (générateur à splines) et `pot`
(seulement pour `valider`). Le venv `Documents/estimation_densite/.venv` a tout ce qu'il faut.

# Modèle image et fusion texte + image — ViT et CLIP

Deux modèles qui regardent l'image : un Vision Transformer affiné sur l'image
seule, et une fusion des embeddings CLIP du texte et de l'image surmontée d'une
régression logistique. Deux tâches, protocole A et protocole B sur deux
événements tenus à l'écart. Seed 42.

## 1. Sources et protocole

Références : PROJET.md §§3–6 ; `src/vit.py` ; `src/clip_fusion.py` ;
`src/images.py` ; `src/fetch_images.py`. Les splits viennent de `src/data.py`,
les métriques et les intervalles bootstrap de `src/evaluate.py` : aucun calcul
de score n'est réimplémenté ici. Les prédictions sont
`results/predictions/{task}_{vit,clip_fusion}_{A,B}_{fold}.csv`, les métriques
les JSON de même nom sous `results/metrics/`.

Les deux modèles sont entraînés sur les **paires** et non sur les tweets
dédupliqués (`text_only=False`, cf. `src/data.py` §74) : l'image fait partie de
l'entrée, donc un tweet accompagné de deux images fournit deux exemples
distincts. Les populations d'évaluation sont inchangées et identiques à celles
des modèles texte : 1 534 paires en test pour la tâche 1, 955 pour la tâche 2.

Protocole B : `hurricane_maria` (un ouragan, type de catastrophe également
présent dans l'entraînement) et `california_wildfires` (le seul incendie, type
absent de l'entraînement), les deux cibles déjà retenues pour BERTweet, afin que
les écarts soient directement comparables. Le côté « in-domain » de chaque écart
est le modèle du protocole A restreint aux mêmes paires test, jamais un autre
découpage.

## 2. Images

Les images ne sont pas versionnées. `python src/fetch_images.py` récupère
l'archive `CrisisMMD_v2.0` (1 902 053 684 octets) depuis l'instantané du
18 décembre 2024 de l'Internet Archive — le serveur officiel de QCRI ne répond
plus — en requêtes `Range` parallèles et reprenables, n'extrait que le
sous-arbre `data_image`, puis ouvre et décode intégralement chacune des 12 708
images référencées par les splits, en réutilisant le contrôleur de
`src/prepare_data.py`. Rien n'est extrait avant que la taille assemblée
corresponde exactement à celle de l'instantané, et une image illisible est une
erreur, jamais une ligne silencieusement retirée.

Résultat consigné dans `results/tables/image_fetch_report.json` :
12 708 / 12 708 images valides, sha256 de l'archive
`351b9be6123a5d5074407d792bfdddc1f470d4909f65ddd429b584d24607d43f`.

## 3. Configuration des runs

**ViT (`src/vit.py`).** Checkpoint `google/vit-base-patch16-224-in21k`, tête de
classification initialisée à neuf. La recette d'entraînement est **celle de
BERTweet, à l'identique** : batch 32, AdamW à 2e-5, warmup linéaire sur 10 % des
pas puis décroissance jusqu'à 0, 5 epochs au maximum, early stopping avec une
patience de 2 sur le F1 macro dev, loss pondérée par les poids « balanced » de
scikit-learn, meilleur état sur dev restauré avant l'inférence. Ce choix est
délibéré : il rend l'écart entre modèle texte et modèle image attribuable à la
modalité et non au réglage. Aucune augmentation d'image n'est appliquée, le
prétraitement se limite au redimensionnement et à la normalisation du
processeur du checkpoint.

**CLIP fusion (`src/clip_fusion.py`).** Checkpoint
`openai/clip-vit-base-patch32`, **encodeurs gelés** : CLIP n'est jamais
entraîné. Le texte et l'image donnent chacun un vecteur de 512 dimensions,
normalisés L2 puis concaténés en 1 024. Le classifieur est exactement celui du
baseline TF-IDF — `class_weight='balanced'`, `solver='lbfgs'`, `penalty='l2'`,
`max_iter=2000`, `tol=1e-4`, `random_state=42`, grille
`C ∈ {0.01, 0.1, 1, 10, 100}` sélectionnée sur dev au F1 macro. La comparaison
TF-IDF → fusion isole donc le changement de features, pas le classifieur.

Les embeddings des 12 708 images et des 11 447 textes distincts des six splits
sont calculés une fois (environ quatre minutes sur GPU Apple) et relus ensuite
depuis `cache/` : les six runs et l'ablation ne repayent pas CLIP.

## 4. Contrôles de fuite

Les scores de la fusion sont nettement au-dessus des modèles texte ; ils ont
donc été contrôlés avant d'être publiés, puisqu'un score inattendu est d'abord
un soupçon de fuite.

| Contrôle | F1 macro test (tâche 1) | Attendu si le protocole est sain |
|---|---|---|
| Référence (fusion, C=1) | 0.922 | — |
| Labels d'entraînement mélangés | 0.539 | effondrement au hasard |
| Features de test désalignées | 0.496 | effondrement au hasard |
| Moitié image désalignée à l'entraînement | 0.862 | retour au niveau du texte seul |

Chevauchements mesurés entre train et test : 0 identifiant de tweet, 0 chemin
d'image, 0 texte identique. En comparant les images **par contenu** (empreinte
des octets), 28 lignes de test sur 1 534 (1,8 %) en tâche 1 et 8 sur 955 (0,8 %)
en tâche 2 réutilisent une photo déjà vue à l'entraînement : trop peu pour
expliquer l'écart observé. Les trois contrôles se comportent comme attendu, donc
les scores sont retenus tels quels.

## 5. Protocole A — résultats

F1 macro sur le test officiel, avec l'intervalle de confiance bootstrap à 95 %
(1 000 rééchantillonnages, seed 42) produit par `src/evaluate.py`. Les trois
autres modèles sont rappelés pour situer les deux nôtres.

| Tâche | Modèle | F1 macro | IC 95 % | F1 pondéré | Accuracy |
|---|---|---|---|---|---|
| 1 · Informative (1 534 paires) | TF-IDF | 0.786 | [0.763, 0.807] | 0.817 | 0.825 |
| 1 · Informative | BERTweet (texte) | 0.859 | [0.840, 0.878] | 0.876 | 0.875 |
| 1 · Informative | **ViT (image seule)** | **0.849** | [0.829, 0.866] | 0.865 | 0.864 |
| 1 · Informative | **CLIP fusion (texte + image)** | **0.922** | [0.908, 0.936] | 0.931 | 0.931 |
| 1 · Informative | LLM zero-shot | 0.647 | [0.623, 0.675] | 0.655 | 0.649 |
| 2 · Humanitarian (955 paires) | TF-IDF | 0.634 | [0.565, 0.694] | 0.744 | 0.740 |
| 2 · Humanitarian | BERTweet (texte) | 0.697 | [0.644, 0.745] | 0.808 | 0.802 |
| 2 · Humanitarian | **ViT (image seule)** | **0.743** | [0.645, 0.816] | 0.828 | 0.828 |
| 2 · Humanitarian | **CLIP fusion (texte + image)** | **0.820** | [0.756, 0.867] | 0.880 | 0.879 |
| 2 · Humanitarian | LLM zero-shot | 0.554 | [0.485, 0.610] | 0.659 | 0.686 |

Trois observations.

**La fusion est le meilleur modèle des cinq, sur les deux tâches.** Son
intervalle de confiance ne recouvre celui d'aucun modèle texte : +0.063 de F1
macro sur BERTweet en tâche 1, +0.123 en tâche 2. Pour la question du projet,
regarder l'image apporte donc une information que le texte seul ne porte pas.

**L'image seule suffit presque, et dépasse le texte pour catégoriser l'aide.**
En tâche 1 le ViT (0.849) reste légèrement sous BERTweet (0.859), intervalles
largement recouvrants. En tâche 2 il le dépasse (0.743 contre 0.697) : décider
*quel type* d'aide concerne un tweet — dégât d'infrastructure, personne
affectée, effort de secours — se lit mieux sur la photo que dans 140 caractères.

**Les encodeurs gelés de CLIP battent le ViT affiné** (0.884 contre 0.849 en
image seule, cf. §7). Ce n'est pas contradictoire : CLIP est pré-entraîné sur
400 millions de paires image-texte du web, bien plus proches de photos de
Twitter que les photos d'ImageNet-21k, et une sonde linéaire sur 9 601 exemples
surapprend moins qu'un affinage complet de 86 millions de paramètres. Le ViT
atteint d'ailleurs son meilleur F1 dev dès la 3ᵉ epoch puis se dégrade, malgré
une loss d'entraînement qui continue de descendre (0.15 → 0.06) : c'est du
surapprentissage, contenu ici par l'early stopping.

Détail par classe en tâche 2, qui montre où se joue l'écart :

| Classe | ViT | CLIP fusion | Support |
|---|---|---|---|
| `affected_individuals` | 0.500 | 0.609 | 9 |
| `infrastructure_and_utility_damage` | 0.816 | 0.892 | 81 |
| `rescue_volunteering_or_donation_effort` | 0.687 | 0.837 | 126 |
| `other_relevant_information` | 0.853 | 0.862 | 235 |
| `not_humanitarian` | 0.859 | 0.902 | 504 |

La fusion progresse surtout sur `rescue_volunteering_or_donation_effort`
(+0.150) : un appel aux dons et une photo de sauvetage se ressemblent peu, mais
le texte dit « donate » quand l'image montre un canot. `affected_individuals`
reste la classe la plus fragile pour les deux modèles, avec 9 exemples de test
seulement : aucune conclusion solide ne peut en être tirée.

## 6. Protocole B — généralisation à une catastrophe inédite

Chaque ligne compare **les mêmes paires test** sous deux modèles : celui du
protocole A, qui a vu l'événement à l'entraînement, et celui du protocole B, qui
ne l'a jamais vu. L'écart est la perte imputable à la nouveauté de la
catastrophe.

| Tâche | Modèle | Événement tenu à l'écart | Paires | In-domain | Inédit | Écart |
|---|---|---|---|---|---|---|
| 1 · Informative | ViT | hurricane maria | 359 | 0.816 | 0.807 | +0.009 |
| 1 · Informative | ViT | california wildfires | 179 | 0.828 | 0.584 | +0.244 |
| 1 · Informative | CLIP fusion | hurricane maria | 359 | 0.936 | 0.910 | +0.026 |
| 1 · Informative | CLIP fusion | california wildfires | 179 | 0.812 | 0.755 | +0.056 |
| 2 · Humanitarian | ViT | hurricane maria | 232 | 0.607 | 0.583 | +0.024 |
| 2 · Humanitarian | ViT | california wildfires | 78 | 0.517 | 0.453 | +0.065 |
| 2 · Humanitarian | CLIP fusion | hurricane maria | 232 | 0.857 | 0.893 | −0.036 |
| 2 · Humanitarian | CLIP fusion | california wildfires | 78 | 0.574 | 0.539 | +0.035 |

Pour mémoire, sur les mêmes populations : BERTweet −0.002 et +0.035 en tâche 1,
+0.074 et +0.106 en tâche 2 ; TF-IDF +0.216 et −0.048 en tâche 1.

**Retirer un ouragan ne coûte presque rien** (écarts de −0.036 à +0.026) : trois
ouragans sur sept événements restent à l'entraînement, et Maria leur ressemble.
**Retirer le seul incendie coûte cher au modèle image** : le ViT perd 0.244 de
F1 macro, la plus grosse dégradation de tous les modèles du projet. C'est
cohérent avec ce qu'il apprend — un modèle qui ne connaît que des inondations et
des décombres n'a aucune représentation de la fumée et des flammes. La fusion
encaisse beaucoup mieux (−0.056) parce que le texte, lui, contient toujours
« wildfire », « fire » ou « evacuation ».

Autrement dit : l'image aide le plus en domaine connu, et c'est précisément le
texte qui rattrape le modèle quand la catastrophe est d'un type inédit. Les deux
modalités ne se recouvrent pas, elles se complètent.

## 7. Ablation de la fusion

Mêmes splits, même classifieur, même grille de C : seules les features changent
(`results/tables/clip_fusion_ablation.csv`, option `--ablation`).

| Tâche | Texte seul (512) | Image seule (512) | Fusion (1 024) |
|---|---|---|---|
| 1 · Informative | 0.864 | 0.884 | **0.922** |
| 2 · Humanitarian | 0.693 | 0.768 | **0.820** |

La fusion dépasse la meilleure des deux modalités de +0.038 et +0.052. Le gain
n'est donc pas un simple effet de dimension : les deux moitiés portent des
informations différentes. Le contrôle du §4 va dans le même sens — désaligner la
moitié image à l'entraînement ramène exactement au niveau du texte seul.

## 8. Limites

- **CLIP tronque le texte à 77 tokens.** Quelques tweets longs perdent leur fin ;
  aucun n'est écarté, et aucune mesure de l'effet n'a été faite.
- **Deux événements seulement en protocole B**, contre sept pour le TF-IDF. Les
  écarts rapportés ne sont pas une moyenne sur toutes les catastrophes, et
  `california_wildfires` ne compte que 78 paires test en tâche 2 — un intervalle
  de confiance très large.
- **Une seule seed (42)** et un seul jeu d'hyperparamètres par modèle. L'écart
  ViT/BERTweet en tâche 1 (0.010) est bien plus petit que la variation attendue
  entre deux seeds : il ne faut pas le lire comme un classement.
- **Aucune augmentation d'image ni recherche d'hyperparamètres pour le ViT.** Le
  choix de reprendre la recette de BERTweet sert la comparaison, pas la
  performance maximale du modèle image.
- **`affected_individuals` compte 9 exemples de test.** Son F1 est instable et
  pèse pourtant pour un cinquième du F1 macro de la tâche 2.
- Le `ViTImageProcessor` rapide de `transformers` requiert `torchvision`, absent
  des dépendances ; le repli PIL est utilisé. Les tenseurs produits sont les
  mêmes, seul le prétraitement est un peu plus lent.

## 9. Reproduction

```bash
python src/fetch_images.py                 # 1,8 Go, puis validation des 12 708 images
python src/clip_fusion.py informative --ablation
python src/clip_fusion.py humanitarian --ablation
python src/vit.py informative
python src/vit.py humanitarian
for e in hurricane_maria california_wildfires; do
  for t in informative humanitarian; do
    python src/clip_fusion.py $t --event $e
    python src/vit.py $t --event $e
  done
done
python src/plot_metrics.py vit clip_fusion
python -m unittest discover -s tests -v
```

Le premier run de `clip_fusion` calcule le cache d'embeddings (environ quatre
minutes) ; les suivants prennent quelques secondes. Chaque run de `vit.py` prend
de 20 à 45 minutes sur GPU Apple M5. Les figures sont sous `figures/` et
`figures/{vit,clip_fusion}_B/`, les matrices en CSV sous
`results/tables/{vit,clip_fusion}_B_confusion_matrices/`.

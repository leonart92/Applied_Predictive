# TF-IDF — rapport du protocole B

Deux tâches × sept événements : 14 expériences hors-domaine réellement entraînées. Seed 42.

## Méthode et populations

Tous les splits proviennent de get_split(task, "B", event, text_only=True). Aucun filtrage manuel des données. Le TF-IDF (unigrammes/bigrammes, défauts sklearn) apprend son vocabulaire et son IDF sur le train sans cible. Régression logistique L2 balanced, lbfgs, max_iter=2000, tol=1e-4, random_state=42, un thread. Grille C=[0.01,0.1,1,10,100] ; maximum de F1 macro dev, égalités exactes au plus petit C. Les configurations sont écrites avant prédiction cible ; aucun refit train+dev. Aucune non-convergence.

Le modèle A n’est pas réentraîné ni réexécuté : seuls ses CSV existants sont restreints à l’événement et alignés sur le test B par tweet_id, image_id, event_name et y_true. Les prédictions et probabilités numériques sont conservées exactement. Les fichiers A restent inchangés, contrôle SHA-256.

**test_all_target conserve toutes les paires tweet-image**, y compris les images multiples des tweets du train cible. text_only=True ne déduplique que le train d’apprentissage. C’est la population définie par les modules partagés et l’unité paire du projet. Le score complémentaire est pondéré par les paires, pas par les tweets distincts. Son bootstrap ordinaire ne corrige pas la corrélation entre les images d’un même tweet. Aucun changement des modules partagés.

## Effectifs par tâche et événement

| Tâche | Événement | Train textes | Dev | Test officiel | Complément paires | Complément tweets |
|---|---|---:|---:|---:|---:|---:|
| informative | hurricane_harvey | 6217 | 1160 | 393 | 3157 | 2882 |
| informative | hurricane_irma | 6446 | 1231 | 324 | 2782 | 2513 |
| informative | hurricane_maria | 6288 | 1205 | 359 | 3101 | 2732 |
| informative | mexico_earthquake | 7552 | 1425 | 127 | 1114 | 1016 |
| informative | iraq_iran_earthquake | 7996 | 1515 | 58 | 500 | 413 |
| informative | california_wildfires | 7491 | 1420 | 179 | 1195 | 1134 |
| informative | srilanka_floods | 7768 | 1482 | 94 | 859 | 710 |
| humanitarian | hurricane_harvey | 3967 | 775 | 248 | 1954 | 1767 |
| humanitarian | hurricane_irma | 4052 | 757 | 215 | 1848 | 1667 |
| humanitarian | hurricane_maria | 3801 | 726 | 232 | 2228 | 1966 |
| humanitarian | mexico_earthquake | 4881 | 920 | 79 | 585 | 539 |
| humanitarian | iraq_iran_earthquake | 5122 | 969 | 23 | 227 | 193 |
| humanitarian | california_wildfires | 4927 | 927 | 78 | 511 | 485 |
| humanitarian | srilanka_floods | 4828 | 914 | 80 | 726 | 599 |

## Résultats sur les mêmes lignes du test officiel

Scores sur [0,1] ; IC 95 % du F1 macro entre crochets. Gap = F1 macro A − F1 macro hors-domaine. Les IC individuels ne sont pas des IC du gap.

| Tâche | Événement | C | F1 dev | F1 A [IC95] | F1 hors-domaine [IC95] | Gap | F1 pondéré hors-domaine | Accuracy hors-domaine |
|---|---|---:|---:|---|---|---:|---:|---:|
| informative | hurricane_harvey | 10 | 0.7813 | 0.7567 [0.7034, 0.8086] | 0.7332 [0.6762, 0.7824] | +0.0235 | 0.8133 | 0.8270 |
| informative | hurricane_irma | 10 | 0.7641 | 0.7177 [0.6516, 0.7761] | 0.6353 [0.5704, 0.7013] | +0.0824 | 0.7486 | 0.7870 |
| informative | hurricane_maria | 1 | 0.7503 | 0.8184 [0.7796, 0.8577] | 0.6022 [0.5501, 0.6544] | +0.2161 | 0.6145 | 0.6602 |
| informative | mexico_earthquake | 1 | 0.7784 | 0.6106 [0.5189, 0.7022] | 0.5815 [0.4934, 0.6727] | +0.0290 | 0.6560 | 0.7087 |
| informative | iraq_iran_earthquake | 1 | 0.7727 | 0.5650 [0.4221, 0.7212] | 0.5279 [0.4021, 0.6712] | +0.0371 | 0.7140 | 0.7414 |
| informative | california_wildfires | 1 | 0.7726 | 0.5328 [0.4509, 0.6281] | 0.5811 [0.4916, 0.6766] | -0.0483 | 0.7773 | 0.7821 |
| informative | srilanka_floods | 1 | 0.7477 | 0.8977 [0.8156, 0.9622] | 0.6024 [0.5014, 0.6954] | +0.2953 | 0.6462 | 0.6170 |
| humanitarian | hurricane_harvey | 100 | 0.6149 | 0.5734 [0.5148, 0.6228] | 0.5244 [0.4645, 0.5770] | +0.0490 | 0.7098 | 0.7137 |
| humanitarian | hurricane_irma | 100 | 0.6119 | 0.5447 [0.4855, 0.5966] | 0.4806 [0.4116, 0.5368] | +0.0640 | 0.6324 | 0.6465 |
| humanitarian | hurricane_maria | 100 | 0.6035 | 0.5851 [0.5280, 0.6386] | 0.3960 [0.3252, 0.4661] | +0.1891 | 0.4939 | 0.4957 |
| humanitarian | mexico_earthquake | 100 | 0.5718 | 0.4580 [0.3464, 0.5506] | 0.4170 [0.2942, 0.5124] | +0.0410 | 0.5709 | 0.5696 |
| humanitarian | iraq_iran_earthquake | 100 | 0.6492 | 0.5232 [0.2389, 0.7223] | 0.3429 [0.1231, 0.4762] | +0.1804 | 0.4161 | 0.5217 |
| humanitarian | california_wildfires | 100 | 0.6294 | 0.4331 [0.3286, 0.5284] | 0.4195 [0.3126, 0.5221] | +0.0136 | 0.6484 | 0.6282 |
| humanitarian | srilanka_floods | 100 | 0.6137 | 0.4359 [0.1961, 0.5961] | 0.4314 [0.2260, 0.5442] | +0.0045 | 0.8657 | 0.8125 |

## Score complémentaire sur toutes les paires cibles

Ce tableau est distinct de la comparaison A/B ; ses populations incluent des tweets issus du train et dev officiels, jamais vus par le modèle hors-domaine. Le test officiel est inclus dans ce complément : les deux scores ne sont pas indépendants.

| Tâche | Événement | F1 macro complément [IC95] |
|---|---|---|
| informative | hurricane_harvey | 0.6941 [0.6758, 0.7128] |
| informative | hurricane_irma | 0.6482 [0.6260, 0.6700] |
| informative | hurricane_maria | 0.6147 [0.5959, 0.6331] |
| informative | mexico_earthquake | 0.6253 [0.5934, 0.6593] |
| informative | iraq_iran_earthquake | 0.5955 [0.5389, 0.6503] |
| informative | california_wildfires | 0.6425 [0.6097, 0.6759] |
| informative | srilanka_floods | 0.6166 [0.5845, 0.6502] |
| humanitarian | hurricane_harvey | 0.5439 [0.5242, 0.5617] |
| humanitarian | hurricane_irma | 0.5275 [0.5083, 0.5463] |
| humanitarian | hurricane_maria | 0.4119 [0.3907, 0.4345] |
| humanitarian | mexico_earthquake | 0.5512 [0.4692, 0.6192] |
| humanitarian | iraq_iran_earthquake | 0.4391 [0.3768, 0.4938] |
| humanitarian | california_wildfires | 0.4402 [0.4013, 0.4789] |
| humanitarian | srilanka_floods | 0.4019 [0.3573, 0.4402] |

## Métriques, limites et livrables

Toutes les métriques passent par src/evaluate.py. IC95 : 1000 bootstrap percentiles ordinaires, seed42. Les classes sont fixes pour chaque tâche, même lorsqu’une classe est absente d’un événement/tirage ; division par zéro = 0. Le support et les scores par classe sont enregistrés dans chaque JSON et dans results/tables/tfidf_B_per_class.csv. Les faibles supports, surtout affected_individuals, limitent la stabilité des F1 macro. Les IC ne représentent pas la variabilité entre seeds. Un gap négatif ne prouve pas une supériorité générale du modèle hors-domaine. Les événements sont comparés séparément, aucune moyenne non pondérée n’est présentée comme score global.

Les sélections à la borne C=100 sont conservées ; aucune extension de grille après consultation du test.

- Notebook exécuté : notebooks/02b_tfidf_protocol_b.ipynb.
- Prédictions hors-domaine : results/predictions/{task}_tfidf_B_{event}.csv.
- Prédictions A réutilisées : results/predictions/in_domain/{task}_tfidf_B_{event}.csv.
- Complément : results/predictions/all_target/{task}_tfidf_B_{event}.csv.
- JSON correspondants sous results/metrics/, avec les mêmes sous-dossiers.
- Matrices PNG : figures/tfidf_B/{in_domain,out_of_domain,all_target}/ ; matrices CSV : results/tables/tfidf_B_confusion_matrices/.
- Tableaux : tfidf_B_summary.csv, tfidf_B_validation.csv (70 candidats), tfidf_B_population_counts.csv, tfidf_B_per_class.csv.
- Configurations verrouillées : results/tables/tfidf_B_configs/ ; inventaire : tfidf_B_artifacts.json ; contrôles : tfidf_B_execution_checks.json.
- Graphiques des gaps : figures/tfidf_B/{task}_tfidf_B_generalization_gap.png.

Contrôles intégrés : 14 folds exécutés, cible absente du train/dev, aucun tweet cible vu en apprentissage, populations A/B exactes, probabilités A conservées, 42 CSV/JSON et matrices cohérents, empreintes A/sources/splits inchangées. Les tests automatiques complémentaires et la vérification d’exécution du notebook sont consignés dans tfidf_B_final_checks.json.

Pour reproduire : kernel .venv vierge puis Run All. Les sorties B sont réécrites dans leurs chemins propres ; aucun résultat A, split ou fichier de préparation n’est modifié. Aucun BERTweet, ViT ou CLIP n’est lancé.

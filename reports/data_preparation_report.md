# Rapport de préparation des données — DATA4001

Rapport reconstruit le 15 novembre 2026 à partir des sources, scripts et résultats sauvegardés. Aucun entraînement ni nouvelle préparation des splits. Les chiffres affichés proviennent des fichiers cités ; les contrôles historiques ne sont pas présentés comme de nouvelles exécutions.

## 1. Périmètre et sources

CrisisMMD v2.0, deux tâches : informative et humanitarian. Seuls les six TSV officiels agreed_lab sont utilisés : `data/crisismmd_datasplit_agreed_label/task_{task}_text_img_agreed_lab_{train,dev,test}.tsv`. La documentation des colonnes est le Readme.txt local des splits. Les images sont stockées sous `data/CrisisMMD_v2.0/data_image/`, hors Git.

Sources de reconstruction : [prepare_data.py](../src/prepare_data.py), `results/tables/preparation_summary.csv`, `validation_report.json`, `reference_verification.json`, `event_class_counts.csv`, `split_overlaps.csv`, `duplicate_pairs.csv` et `excluded_images.csv`.

## 2. Unité, schéma et choix de la cible

L’unité conservée dans les CSV est la paire (tweet_id, image_id). Aucune déduplication des splits de préparation. Les images multiples du train restent présentes ; dev et test conservent les lignes officielles. La déduplication du train des modèles texte est effectuée ultérieurement par `get_split(..., text_only=True)` dans src/data.py.

Les neuf colonnes sont conservées : `event_name`, `tweet_id`, `image_id`, `tweet_text`, `image`, `label`, `label_text`, `label_image`, `label_text_image`. Les identifiants sont des chaînes de caractères.

La cible est `label`, étiquette de la tâche. Sur les six sources, `label`, `label_text` et `label_image` concordent ligne par ligne. `label_text_image` vaut Positive et représente l’accord entre modalités, pas une classe de catastrophe. Le script contrôle ces propriétés avant écriture. Aucun label issu d’une autre version du dataset n’est ajouté.

## 3. Préparation implémentée

1. Lecture UTF-8 des TSV avec `csv.DictReader(..., quoting=csv.QUOTE_NONE)` afin de préserver les guillemets initiaux du texte brut, sans nettoyage ni normalisation du texte.
2. Contrôle du schéma, des champs obligatoires, de l’accord des annotations et des classes autorisées.
3. Résolution des images sous data/CrisisMMD_v2.0 : la colonne image de sortie contient un chemin relatif à la racine du dépôt.
4. Vérification Pillow avec verify(), puis réouverture et décodage complet avec load(). Les images tronquées ne sont pas tolérées.
5. Exclusion uniquement des images absentes ou rejetées comme corrompues. Les erreurs de permission et limites de protection ne sont pas assimilées silencieusement à une corruption.
6. Écriture dans l’ordre officiel et relecture exacte des valeurs ; contrôle des paires répétées et de la conservation des images multiples.
7. Comptages événement × classe × split depuis les CSV relus et génération des journaux.

Les encodages imparfaits déjà présents dans les tweets sont conservés. Les caractères, liens, mentions et retours à la ligne ne sont pas corrigés. Le format CSV peut échapper les guillemets ; cela ne modifie pas la valeur restituée par un lecteur CSV.

## 4. Effectifs et exclusions

| task | split | initial_rows | retained_rows | missing_images | corrupted_images | removed_rows | duplicate_pair_excess | initial_multi_image_tweets | retained_multi_image_tweets |
|---|---|---|---|---|---|---|---|---|---|
| informative | train | 9601 | 9601 | 0 | 0 | 0 | 0 | 772 | 772 |
| informative | dev | 1573 | 1573 | 0 | 0 | 0 | 0 | 0 | 0 |
| informative | test | 1534 | 1534 | 0 | 0 | 0 | 0 | 0 | 0 |
| humanitarian | train | 6126 | 6126 | 0 | 0 | 0 | 0 | 545 | 545 |
| humanitarian | dev | 998 | 998 | 0 | 0 | 0 | 0 | 0 | 0 |
| humanitarian | test | 955 | 955 | 0 | 0 | 0 | 0 | 0 | 0 |

Total conservé : **20787 paires**. Images manquantes : **0 lignes** ; corrompues : **0 lignes**. Le rapport de validation sauvegardé indique **12708 chemins d’images distincts** référencés. Les tâches partagent des images : additionner leurs nombres de lignes ne donne pas un nombre d’images uniques.

`excluded_images.csv` et `duplicate_pairs.csv` contiennent leur en-tête seulement : aucun rejet ni doublon de paire n’y est enregistré. Les nombres de tweets multi-images du train, avant/après préparation, figurent dans le tableau ci-dessus.

## 5. Comparaison aux références du projet

| task | split | pairs | distinct_tweet_ids |
|---|---|---|---|
| informative | train | 9601 | 8293 |
| informative | dev | 1573 | 1573 |
| informative | test | 1534 | 1534 |
| informative | total | 12708 | 11400 |
| humanitarian | train | 6126 | 5263 |
| humanitarian | dev | 998 | 998 |
| humanitarian | test | 955 | 955 |
| humanitarian | total | 8079 | 7216 |

Ces effectifs correspondent exactement à PROJET.md §3 : train informative 8 293 textes / 9 601 paires, train humanitarian 5 263 textes / 6 126 paires ; totaux respectifs 11 400 / 12 708 et 7 216 / 8 079. Les six sommes de lignes sont cohérentes avec les sources et aucune exclusion n’explique un écart, puisqu’il n’y en a pas.

## 6. Distributions de classes

| task | label | train_pairs | dev_pairs | test_pairs | total_distinct_tweets |
|---|---|---|---|---|---|
| informative | informative | 6345 | 1056 | 1030 | 7632 |
| informative | not_informative | 3256 | 517 | 504 | 3768 |
| humanitarian | affected_individuals | 71 | 9 | 9 | 88 |
| humanitarian | infrastructure_and_utility_damage | 612 | 80 | 81 | 657 |
| humanitarian | rescue_volunteering_or_donation_effort | 912 | 149 | 126 | 1037 |
| humanitarian | other_relevant_information | 1279 | 239 | 235 | 1666 |
| humanitarian | not_humanitarian | 3252 | 521 | 504 | 3768 |

Les références des classes humanitaires portent sur les textes distincts : 88, 657, 1 037, 1 666 et 3 768 dans l’ordre affected_individuals, infrastructure_and_utility_damage, rescue_volunteering_or_donation_effort, other_relevant_information, not_humanitarian. Elles concordent avec la colonne total_distinct_tweets. Les supports test sont 9, 81, 126, 235 et 504. Le projet ne fournit pas de référence informative par classe.

[event_class_counts.csv](../results/tables/event_class_counts.csv) contient 147 combinaisons, zéros compris : 7 événements × 3 splits × 2 classes informatives et 7 × 3 × 5 classes humanitaires. Chaque total par tâche/split correspond au CSV nettoyé. Aucune classe attendue ne manque à l’échelle d’un split complet.

## 7. Validation et traçabilité

Le rapport sauvegardé validation_report.json répertorie les contrôles de schéma, labels, champs obligatoires, images, relecture CSV, préservation de l’ordre et de la multiplicité, et réconciliation des effectifs. Il enregistre aussi Pillow 13.0.0 et les empreintes SHA-256 des six TSV et six CSV. Ces douze empreintes ont été vérifiées lors de la génération.

split_overlaps.csv rapporte zéro tweet_id et zéro paire communs entre les splits d’une même tâche. execution_checks.json conserve la preuve historique de deux préparations identiques ; aucun nouveau passage de préparation ni nouveau test d’image n’est revendiqué par cette reconstruction documentaire.

## 8. Modules communs et reproduction

`src/data.py` charge les CSV et construit les protocoles A/B. `src/evaluate.py` centralise métriques et bootstrap. Les notebooks consomment les CSV ; ils n’utilisent pas les annotations de la version non-agreed.

Depuis la racine, lorsque les données externes ont été téléchargées et placées aux chemins documentés :

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/prepare_data.py
```

Cette commande réécrit les splits : elle est fournie pour une future reproduction, et n’a pas été exécutée pour reconstruire ce rapport. Les modèles texte utilisent ensuite get_split(task, "A", "official", text_only=True).

## 9. Limites de la reconstruction

Le texte historique exact du rapport disparu n’est pas disponible. Cette version est une reconstruction vérifiée, pas une affirmation de restauration octet pour octet. Les résultats et manifestes existants ne sont pas réécrits. Les détails non enregistrés, tels que durées précises, machine d’entraînement ou décisions de groupe non validées, ne sont pas inventés.

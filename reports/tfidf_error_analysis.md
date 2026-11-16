# Analyse des erreurs de la baseline TF-IDF — DATA4001

## Périmètre et méthode

Analyse des prédictions existantes, sans entraînement, inférence, modification de split, de métrique ou d’hyperparamètre. Les vrais labels sont ceux des CSV nettoyés agreed_lab. Les jointures vérifient exactement tweet_id, image_id, event_name et y_true. Les 30 ensembles de prédictions/métriques lus (2 A et 28 sous-ensembles B in/out) sont contrôlés en mémoire avec la fonction commune compute_metrics. Aucun all_target ne sert à la comparaison A/B. Les probabilités correspondent aux colonnes des classes prédites ; elles ne sont pas nécessairement calibrées.

Les dix exemples sont une sélection qualitative raisonnée : cinq par tâche, six A et quatre B, six événements, plusieurs directions de confusion et niveaux de confiance. Ils ne constituent pas un échantillon statistique représentatif des causes. Toutes les erreurs, y compris celles non sélectionnées, restent dans results/tables/tfidf_error_analysis_all_errors.csv ; les transitions de toutes les paires sont également conservées.

## 1. Protocole A — observations quantitatives

### informative

269 erreurs sur 1534 lignes ; F1 macro 0.7855.

| label | precision | recall | f1-score | support |
|---|---|---|---|---|
| not_informative | 0.8133 | 0.6052 | 0.6940 | 504 |
| informative | 0.8283 | 0.9320 | 0.8771 | 1030 |

Principales confusions dirigées :

| y_true | y_pred | count | support | share_of_errors | share_of_true_class |
|---|---|---|---|---|---|
| not_informative | informative | 199 | 504 | 0.7398 | 0.3948 |
| informative | not_informative | 70 | 1030 | 0.2602 | 0.0680 |

### humanitarian

248 erreurs sur 955 lignes ; F1 macro 0.6342.

| label | precision | recall | f1-score | support |
|---|---|---|---|---|
| affected_individuals | 0.3750 | 0.3333 | 0.3529 | 9 |
| infrastructure_and_utility_damage | 0.5667 | 0.6296 | 0.5965 | 81 |
| rescue_volunteering_or_donation_effort | 0.6211 | 0.7937 | 0.6969 | 126 |
| other_relevant_information | 0.6726 | 0.8043 | 0.7326 | 235 |
| not_humanitarian | 0.8771 | 0.7222 | 0.7922 | 504 |

Principales confusions dirigées :

| y_true | y_pred | count | support | share_of_errors | share_of_true_class |
|---|---|---|---|---|---|
| not_humanitarian | other_relevant_information | 74 | 504 | 0.2984 | 0.1468 |
| not_humanitarian | rescue_volunteering_or_donation_effort | 41 | 504 | 0.1653 | 0.0813 |
| other_relevant_information | not_humanitarian | 23 | 235 | 0.0927 | 0.0979 |
| not_humanitarian | infrastructure_and_utility_damage | 22 | 504 | 0.0887 | 0.0437 |
| rescue_volunteering_or_donation_effort | not_humanitarian | 14 | 126 | 0.0565 | 0.1111 |
| infrastructure_and_utility_damage | other_relevant_information | 12 | 81 | 0.0484 | 0.1481 |

En informative, not_informative a le F1 et le rappel les plus faibles. Les faux positifs de la classe informative dominent. En humanitarian, affected_individuals est la classe la plus fragile, mais seulement 9 exemples test sont disponibles : 3 correctement retrouvés, 6 manqués. Les erreurs de not_humanitarian vers other_relevant_information et rescue sont les confusions les plus nombreuses en valeur absolue ; cela ne signifie pas que cette classe ait le plus mauvais taux d’erreur.

## 2. Protocole B — généralisation sur le test officiel

Les F1 A et B ci-dessous portent sur exactement les mêmes paires de chaque événement. Un gap positif favorise A ; un gap négatif favorise B. Les IC95 par modèle figurent dans le CSV détaillé, sans être interprétés comme un IC du gap.

| task | event_name | n_test | f1_macro_in_domain | f1_macro_out_of_domain | generalization_gap | A_correct_B_wrong | A_wrong_B_correct |
|---|---|---|---|---|---|---|---|
| informative | hurricane_harvey | 393 | 0.7567 | 0.7332 | 0.0235 | 13 | 7 |
| informative | hurricane_irma | 324 | 0.7177 | 0.6353 | 0.0824 | 18 | 7 |
| informative | hurricane_maria | 359 | 0.8184 | 0.6022 | 0.2161 | 88 | 31 |
| informative | mexico_earthquake | 127 | 0.6106 | 0.5815 | 0.0290 | 4 | 1 |
| informative | iraq_iran_earthquake | 58 | 0.5650 | 0.5279 | 0.0371 | 3 | 0 |
| informative | california_wildfires | 179 | 0.5328 | 0.5811 | -0.0483 | 13 | 5 |
| informative | srilanka_floods | 94 | 0.8977 | 0.6024 | 0.2953 | 30 | 1 |
| humanitarian | hurricane_harvey | 248 | 0.5734 | 0.5244 | 0.0490 | 20 | 11 |
| humanitarian | hurricane_irma | 215 | 0.5447 | 0.4806 | 0.0640 | 20 | 5 |
| humanitarian | hurricane_maria | 232 | 0.5851 | 0.3960 | 0.1891 | 74 | 5 |
| humanitarian | mexico_earthquake | 79 | 0.4580 | 0.4170 | 0.0410 | 7 | 4 |
| humanitarian | iraq_iran_earthquake | 23 | 0.5232 | 0.3429 | 0.1804 | 6 | 6 |
| humanitarian | california_wildfires | 78 | 0.4331 | 0.4195 | 0.0136 | 7 | 8 |
| humanitarian | srilanka_floods | 80 | 0.4359 | 0.4314 | 0.0045 | 11 | 1 |

### Classes qui se dégradent le plus (support non nul)

| task | event_name | label | support | f1_in_domain | f1_out_of_domain | f1_drop | recall_in_domain | recall_out_of_domain |
|---|---|---|---|---|---|---|---|---|
| humanitarian | iraq_iran_earthquake | affected_individuals | 4 | 0.6000 | 0.0000 | 0.6000 | 0.7500 | 0.0000 |
| humanitarian | iraq_iran_earthquake | other_relevant_information | 5 | 0.5455 | 0.0000 | 0.5455 | 0.6000 | 0.0000 |
| informative | hurricane_maria | not_informative | 165 | 0.8083 | 0.4505 | 0.3578 | 0.8303 | 0.3030 |
| humanitarian | hurricane_maria | not_humanitarian | 129 | 0.8446 | 0.5028 | 0.3418 | 0.8217 | 0.3488 |
| informative | srilanka_floods | informative | 20 | 0.8444 | 0.5263 | 0.3181 | 0.9500 | 1.0000 |
| humanitarian | hurricane_maria | other_relevant_information | 58 | 0.8000 | 0.4885 | 0.3115 | 0.8966 | 0.9138 |
| informative | srilanka_floods | not_informative | 74 | 0.9510 | 0.6786 | 0.2725 | 0.9189 | 0.5135 |
| humanitarian | hurricane_maria | infrastructure_and_utility_damage | 21 | 0.6286 | 0.3571 | 0.2714 | 0.5238 | 0.2381 |
| humanitarian | srilanka_floods | infrastructure_and_utility_damage | 3 | 0.8000 | 0.6000 | 0.2000 | 0.6667 | 1.0000 |
| humanitarian | iraq_iran_earthquake | rescue_volunteering_or_donation_effort | 1 | 0.5000 | 0.3333 | 0.1667 | 1.0000 | 1.0000 |
| humanitarian | hurricane_harvey | infrastructure_and_utility_damage | 20 | 0.6000 | 0.4444 | 0.1556 | 0.6000 | 0.4000 |
| humanitarian | hurricane_irma | infrastructure_and_utility_damage | 17 | 0.6047 | 0.4571 | 0.1475 | 0.7647 | 0.4706 |

Sur Maria, not_informative perd fortement en rappel et not_humanitarian est davantage envoyé vers other_relevant_information. Le F1 de other_relevant_information baisse aussi alors que son rappel augmente légèrement : la précision se dégrade, donc il ne faut pas décrire cette baisse de F1 comme un effondrement du rappel.

Sur srilanka_floods, le modèle B informative prédit informative pour des messages non informatifs : le rappel de informative augmente, mais son F1 diminue sous l’effet des faux positifs. En humanitarian Irak-Iran, affected_individuals et other_relevant_information ont un rappel nul sous B, sur seulement 4 et 5 exemples. Les 23 lignes totales et les IC larges imposent une lecture prudente.

California est un contre-exemple à une dégradation systématique du F1 macro : il est meilleur sous B que sous A. Ce gain ne correspond pas à moins d’erreurs : elles passent de 31 à 39 (13 nouveaux échecs contre 5 corrections). Le rappel de not_informative passe de 10,34 % à 27,59 %, tandis que celui de informative recule de 96,67 % à 88,00 %. Le F1 macro pondère également les classes, contrairement au nombre brut d’erreurs : ces deux observations sont compatibles. Les transitions A faux/B correct sont rapportées pour tous les événements, et E03 en montre un cas. E05 reste une erreur sous B en California malgré ce gain moyen.

### Principales confusions B pour chaque tâche et événement

**informative / hurricane_harvey**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_informative | informative | 51 | 97 | 0.5258 |
| informative | not_informative | 17 | 296 | 0.0574 |

**informative / hurricane_irma**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_informative | informative | 61 | 84 | 0.7262 |
| informative | not_informative | 8 | 240 | 0.0333 |

**informative / hurricane_maria**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_informative | informative | 115 | 165 | 0.6970 |
| informative | not_informative | 7 | 194 | 0.0361 |

**informative / mexico_earthquake**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_informative | informative | 33 | 43 | 0.7674 |
| informative | not_informative | 4 | 84 | 0.0476 |

**informative / iraq_iran_earthquake**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_informative | informative | 10 | 12 | 0.8333 |
| informative | not_informative | 5 | 46 | 0.1087 |

**informative / california_wildfires**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_informative | informative | 21 | 29 | 0.7241 |
| informative | not_informative | 18 | 150 | 0.1200 |

**informative / srilanka_floods**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_informative | informative | 36 | 74 | 0.4865 |

**humanitarian / hurricane_harvey**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_humanitarian | other_relevant_information | 19 | 121 | 0.1570 |
| other_relevant_information | not_humanitarian | 10 | 55 | 0.1818 |
| not_humanitarian | rescue_volunteering_or_donation_effort | 9 | 121 | 0.0744 |

**humanitarian / hurricane_irma**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_humanitarian | other_relevant_information | 32 | 77 | 0.4156 |
| other_relevant_information | not_humanitarian | 9 | 94 | 0.0957 |
| infrastructure_and_utility_damage | other_relevant_information | 9 | 17 | 0.5294 |

**humanitarian / hurricane_maria**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_humanitarian | other_relevant_information | 81 | 129 | 0.6279 |
| infrastructure_and_utility_damage | other_relevant_information | 15 | 21 | 0.7143 |
| rescue_volunteering_or_donation_effort | other_relevant_information | 9 | 22 | 0.4091 |

**humanitarian / mexico_earthquake**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_humanitarian | rescue_volunteering_or_donation_effort | 11 | 43 | 0.2558 |
| not_humanitarian | other_relevant_information | 7 | 43 | 0.1628 |
| other_relevant_information | not_humanitarian | 4 | 11 | 0.3636 |

**humanitarian / iraq_iran_earthquake**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| other_relevant_information | not_humanitarian | 5 | 5 | 1.0000 |
| affected_individuals | rescue_volunteering_or_donation_effort | 3 | 4 | 0.7500 |
| affected_individuals | not_humanitarian | 1 | 4 | 0.2500 |

**humanitarian / california_wildfires**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_humanitarian | rescue_volunteering_or_donation_effort | 8 | 47 | 0.1702 |
| infrastructure_and_utility_damage | not_humanitarian | 4 | 13 | 0.3077 |
| not_humanitarian | other_relevant_information | 4 | 47 | 0.0851 |

**humanitarian / srilanka_floods**

| y_true | y_pred | count | support | share_of_true_class |
|---|---|---|---|---|
| not_humanitarian | other_relevant_information | 9 | 75 | 0.1200 |
| not_humanitarian | infrastructure_and_utility_damage | 4 | 75 | 0.0533 |
| other_relevant_information | not_humanitarian | 1 | 1 | 1.0000 |

## 3. Dix exemples vérifiables et catégories qualitatives

Les constats décrivent ce qui est lisible dans le texte. Les hypothèses ne démontrent ni le mécanisme TF-IDF ni une erreur d’annotation. Aucune cause n’est imposée lorsqu’elle ne se déduit pas du texte. Une image réelle est disponible pour chaque exemple sélectionné et peut être affichée dans le notebook ; les commentaires ci-dessous ne prétendent pas avoir établi son contenu ou refait l’annotation multimodale.

### E01 — informative, protocole A

- Tweet : `921578610624000004` ; image : `921578610624000004_0` ; événement : `hurricane_maria`.
- Vraie classe : `informative` ; prédite : `not_informative` ; confiance : **0.8549**.
- Comparaison : A → `not_informative` (p=0.8549), B → `not_informative` (p=0.5445).
- Image locale : `data/CrisisMMD_v2.0/data_image/hurricane_maria/21_10_2017/921578610624000004_0.jpg` ; fichier présent : True.

Texte original (aucune normalisation) :

<pre>my dad collected so much stuff for puerto rico https://t.co/sCArZhXgnl</pre>

**Catégorie :** Texte ambigu ou trop court; Texte qui ne contient pas assez d’informations pour identifier la classe.

**Observation :** Le texte mentionne une collecte par le père pour Puerto Rico, sans détailler les objets ni les destinataires.

**Hypothèse, non démontrée :** Un message informel et peu explicite peut masquer la dimension d’aide ; le rôle des mots géographiques n’est pas établi.

**Intérêt de la sélection :** Faux négatif assez confiant, forme informelle d’aide.

### E02 — informative, protocole A

- Tweet : `923475560029974528` ; image : `923475560029974528_0` ; événement : `hurricane_maria`.
- Vraie classe : `not_informative` ; prédite : `informative` ; confiance : **0.9912**.
- Comparaison : A → `informative` (p=0.9912), B → `informative` (p=0.9897).
- Image locale : `data/CrisisMMD_v2.0/data_image/hurricane_maria/26_10_2017/923475560029974528_0.jpg` ; fichier présent : True.

Texte original (aucune normalisation) :

<pre>RT: Fred Bentler: https://t.co/AfwpXQRTxx https://t.co/HmgJWKzYKx</pre>

**Catégorie :** Texte qui ne contient pas assez d’informations pour identifier la classe.

**Observation :** Le texte visible contient essentiellement RT, un nom et deux liens.

**Hypothèse, non démontrée :** Le texte seul ne décrit pas la catastrophe ni une action humanitaire ; cette forte probabilité erronée ne prouve pas une cause lexicale particulière.

**Intérêt de la sélection :** Faux positif très confiant avec presque aucun contenu exploitable.

### E03 — informative, protocole A

- Tweet : `905664032949993472` ; image : `905664032949993472_0` ; événement : `hurricane_harvey`.
- Vraie classe : `informative` ; prédite : `not_informative` ; confiance : **0.5002**.
- Comparaison : A → `not_informative` (p=0.5002), B → `informative` (p=0.7765).
- Image locale : `data/CrisisMMD_v2.0/data_image/hurricane_harvey/7_9_2017/905664032949993472_0.jpg` ; fichier présent : True.

Texte original (aucune normalisation) :

<pre>Is it me or do #Harvey and #Irma seem hell-bent on destroying climate-denial red states? https://t.co/wMbeUFKAiI</pre>

**Catégorie :** Texte ambigu ou trop court.

**Observation :** Question rhétorique sur Harvey, Irma et les États climatosceptiques ; la cible fournie est informative.

**Hypothèse, non démontrée :** La frontière entre commentaire politique et information utile est ambiguë ici ; aucun label n’est changé. B classe ce cas correctement, contrairement à A.

**Intérêt de la sélection :** Décision proche du seuil et contre-exemple A faux/B correct.

### E04 — informative, protocole B

- Tweet : `871037159326830592` ; image : `871037159326830592_0` ; événement : `srilanka_floods`.
- Vraie classe : `not_informative` ; prédite : `informative` ; confiance : **0.9793**.
- Comparaison : A → `not_informative` (p=0.8400), B → `informative` (p=0.9793).
- Image locale : `data/CrisisMMD_v2.0/data_image/srilanka_floods/3_6_2017/871037159326830592_0.jpg` ; fichier présent : True.

Texte original (aucune normalisation) :

<pre>Outskirts of Sadrith Mora #ESO #Morrowind https://t.co/2DFL1maTZ6</pre>

**Catégorie :** Vocabulaire spécifique à une catastrophe ou à un lieu.

**Observation :** Le texte cite Sadrith Mora, #ESO et #Morrowind, mais est rattaché à l’événement srilanka_floods.

**Hypothèse, non démontrée :** Le nom Mora apparaît dans un contexte de jeu vidéo ; une association lexicale hors contexte est plausible, sans preuve qu’il a piloté la décision du modèle.

**Intérêt de la sélection :** Faux positif hors-domaine très confiant, A correct.

### E05 — informative, protocole B

- Tweet : `920348743316180993` ; image : `920348743316180993_0` ; événement : `california_wildfires`.
- Vraie classe : `informative` ; prédite : `not_informative` ; confiance : **0.5545**.
- Comparaison : A → `informative` (p=0.6344), B → `not_informative` (p=0.5545).
- Image locale : `data/CrisisMMD_v2.0/data_image/california_wildfires/17_10_2017/920348743316180993_0.jpg` ; fichier présent : True.

Texte original (aucune normalisation) :

<pre>How will the California wildfires affect your next glass of wine? Details tonight on Nbc 15 News.@mynbc15 https://t.co/JsHIO4X8q6</pre>

**Catégorie :** Texte qui ne contient pas assez d’informations pour identifier la classe.

**Observation :** Le tweet demande comment les incendies affecteront le vin et renvoie à un reportage, sans donner la réponse.

**Hypothèse, non démontrée :** La portée informative peut dépendre du contenu lié ; le texte seul ne permet pas de vérifier cette portée. Ce cas échoue sous B malgré une amélioration moyenne sur California.

**Intérêt de la sélection :** Erreur de confiance modérée dans un événement où B améliore le F1 global.

### E06 — humanitarian, protocole A

- Tweet : `905221184819781632` ; image : `905221184819781632_1` ; événement : `hurricane_harvey`.
- Vraie classe : `affected_individuals` ; prédite : `not_humanitarian` ; confiance : **0.8793**.
- Comparaison : A → `not_humanitarian` (p=0.8793), B → `not_humanitarian` (p=0.8473).
- Image locale : `data/CrisisMMD_v2.0/data_image/hurricane_harvey/6_9_2017/905221184819781632_1.jpg` ; fichier présent : True.

Texte original (aucune normalisation) :

<pre>Hope was just 10 days old when her family was rescued following #Harvey. (Photo by Noelle Mills) https://t.co/PTU3hHEWl2</pre>

**Catégorie :** Non attribuée à partir du texte.

**Observation :** Une enfant de dix jours et sa famille sont explicitement décrites comme secourues après Harvey.

**Hypothèse, non démontrée :** Le texte contient des indices humanitaires explicites ; la prédiction not_humanitarian reste inexpliquée ici. Ce cas contredit une explication générale par manque d’information.

**Intérêt de la sélection :** Échec difficile sur la classe rare affected_individuals, sans explication forcée.

### E07 — humanitarian, protocole A

- Tweet : `912013825863557121` ; image : `912013825863557121_0` ; événement : `hurricane_maria`.
- Vraie classe : `infrastructure_and_utility_damage` ; prédite : `other_relevant_information` ; confiance : **0.5857**.
- Comparaison : A → `other_relevant_information` (p=0.5857), B → `other_relevant_information` (p=0.9298).
- Image locale : `data/CrisisMMD_v2.0/data_image/hurricane_maria/24_9_2017/912013825863557121_0.jpg` ; fichier présent : True.

Texte original (aucune normalisation) :

<pre>Hurricane Maria damages Dominica’s main hospital, leaves ‘war zone’ conditions https://t.co/rYammNVB8y https://t.co/meAHPR6Zet</pre>

**Catégorie :** Confusion entre deux classes proches.

**Observation :** Le texte mentionne des dégâts sur le principal hôpital de Dominica et des conditions de zone de guerre.

**Hypothèse, non démontrée :** Le dommage à une infrastructure et l’information générale sur la catastrophe coexistent ; le modèle choisit la catégorie générale, sans preuve de son mécanisme interne.

**Intérêt de la sélection :** Confusion infrastructure → other, confiance intermédiaire.

### E08 — humanitarian, protocole A

- Tweet : `911846702558138368` ; image : `911846702558138368_0` ; événement : `mexico_earthquake`.
- Vraie classe : `not_humanitarian` ; prédite : `rescue_volunteering_or_donation_effort` ; confiance : **0.9928**.
- Comparaison : A → `rescue_volunteering_or_donation_effort` (p=0.9928), B → `rescue_volunteering_or_donation_effort` (p=0.9879).
- Image locale : `data/CrisisMMD_v2.0/data_image/mexico_earthquake/24_9_2017/911846702558138368_0.jpg` ; fichier présent : True.

Texte original (aucune normalisation) :

<pre>Volunteers filling semi trailer for Mexico earthquake relief https://t.co/rHt9ZzLeyH https://t.co/xmLr1CYie8</pre>

**Catégorie :** Erreur potentielle d’annotation ou étiquette discutable.

**Observation :** Le texte décrit des volontaires remplissant une semi-remorque pour les secours du séisme au Mexique ; le label fourni est not_humanitarian.

**Hypothèse, non démontrée :** Le sens textuel paraît compatible avec la classe prédite rescue ; une relecture de l’annotation et du contexte multimodal serait nécessaire. Ce n’est pas une preuve d’erreur d’annotation.

**Intérêt de la sélection :** Label discutable et erreur très confiante, conservée comme erreur officielle.

### E09 — humanitarian, protocole B

- Tweet : `910579624857522176` ; image : `910579624857522176_0` ; événement : `hurricane_maria`.
- Vraie classe : `infrastructure_and_utility_damage` ; prédite : `other_relevant_information` ; confiance : **0.5918**.
- Comparaison : A → `infrastructure_and_utility_damage` (p=0.8174), B → `other_relevant_information` (p=0.5918).
- Image locale : `data/CrisisMMD_v2.0/data_image/hurricane_maria/20_9_2017/910579624857522176_0.jpg` ; fichier présent : True.

Texte original (aucune normalisation) :

<pre>Houses damaged, roofs blown off as Hurricane Maria batters USVI https://t.co/bP9jU23bwS https://t.co/ZRUOhXm5CN</pre>

**Catégorie :** Confusion entre deux classes proches; Vocabulaire spécifique à une catastrophe ou à un lieu.

**Observation :** Le texte décrit des maisons endommagées et des toits arrachés aux USVI par Maria.

**Hypothèse, non démontrée :** La description des dégâts est explicite ; le contexte local peut accompagner le décalage, mais on ne peut pas l’isoler comme cause. A est correct et B prédit other.

**Intérêt de la sélection :** Dégradation A correct/B faux dans une classe touchée sur Maria.

### E10 — humanitarian, protocole B

- Tweet : `930293724990410752` ; image : `930293724990410752_0` ; événement : `iraq_iran_earthquake`.
- Vraie classe : `affected_individuals` ; prédite : `rescue_volunteering_or_donation_effort` ; confiance : **0.5544**.
- Comparaison : A → `affected_individuals` (p=0.9329), B → `rescue_volunteering_or_donation_effort` (p=0.5544).
- Image locale : `data/CrisisMMD_v2.0/data_image/iraq_iran_earthquake/14_11_2017/930293724990410752_0.jpg` ; fichier présent : True.

Texte original (aucune normalisation) :

<pre>Iran-Iraq earthquake: 450 dead, 7,000 injured as Iranian rescue missions wind down https://t.co/IfAwYPwWeF https://t.co/EY1wzX6pa6</pre>

**Catégorie :** Confusion entre deux classes proches.

**Observation :** Le texte associe 450 morts, 7 000 blessés et la fin des missions de secours en Iran.

**Hypothèse, non démontrée :** Les indices de victimes et de secours rendent deux catégories sémantiquement plausibles ; le label officiel affected_individuals reste la référence.

**Intérêt de la sélection :** Rappel nul de la classe affectée en Irak-Iran sous B, exemple de confiance modérée.

## 4. Texte réutilisable pour les livrables

### Milestone — constat principal

Sur le test officiel, la baseline TF-IDF réalise 269 erreurs en informative et 248 en humanitarian. Elle retrouve difficilement les messages not_informative (rappel 60,52 %) et la classe rare affected_individuals (rappel 33,33 %, F1 35,29 %, n=9). Les exemples illustrent des textes peu explicites, des frontières entre catégories proches et des désaccords d’annotation possibles, sans permettre une attribution causale certaine.

### Rapport final — généralisation

La comparaison appariée sur le test officiel montre des pertes de F1 macro importantes en informative pour srilanka_floods (29,53 points) et hurricane_maria (21,61 points), et en humanitarian pour hurricane_maria (18,91 points). Ces pertes ne sont pas systématiques : informative/California gagne 4,83 points hors-domaine. L’inspection par classe distingue la perte de rappel de la contamination par faux positifs ; les faibles supports de certaines classes et les intervalles de confiance empêchent de généraliser ces constats sans réserve.

## 5. Vérifications et limites

- 30 fichiers de prédictions et leurs JSON validés ; vrais labels identiques aux splits et probabilités de la classe prédite vérifiées.
- 14 comparaisons A/B sur le test officiel, jamais sur all_target ; prédictions in-domain identiques aux sorties A originales.
- Tous les cas erronés A et B conservés dans l’inventaire complet ; les exemples ne servent pas à estimer la fréquence des catégories qualitatives.
- Les classes absentes restent dans les métriques de la tâche avec la convention du module commun ; elles sont signalées et ne sont pas traitées comme des pertes de rappel sur des exemples observés.
- Les noms de lieux et indices lexicaux suggèrent des hypothèses de décalage ; aucun poids de modèle ni ablation ne démontre ici un mécanisme causal.
- Les étiquettes discutables sont seulement signalées pour une relecture future ; aucun label n’est modifié.
- Aucun entraînement, nouvelle prédiction, réglage sur le test ou modification d’un résultat existant.
- Images : présence et décodage contrôlés pour les dix exemples, sans assertion inventée sur leur contenu.
- Empreintes des entrées et résultats préexistants vérifiées en fin d’exécution ; contrôles dans results/tables/tfidf_error_analysis_checks.json.

## 6. Fichiers produits

- notebooks/07_error_analysis_tfidf.ipynb : analyse exécutée, textes et images réelles affichés.
- results/tables/tfidf_error_analysis.csv : dix exemples et annotations qualitatives.
- results/tables/tfidf_error_analysis_all_errors.csv : inventaire exhaustif A + B hors-domaine sur test officiel.
- results/tables/tfidf_error_analysis_A_classes.csv et tfidf_error_analysis_B_classes.csv : scores et variations par classe.
- results/tables/tfidf_error_analysis_B_events.csv : comparaison par événement, IC95 et transitions.
- results/tables/tfidf_error_analysis_confusions.csv : toutes les confusions dirigées A et B in/out.
- results/tables/tfidf_error_analysis_transitions.csv : les 2 489 comparaisons appariées, succès compris.
- results/tables/tfidf_error_analysis_checks.json : validation et empreintes.
- reports/tfidf_error_analysis.md : présent rapport.

# TRAIL LAB

### Présentation

`trail-lab` est une collection de notebooks Jupyter open source dédiés à l'analyse de la performance en trail running. À partir de fichiers FIT bruts exportés depuis des montres GPS, les notebooks couvrent un ensemble de méthodes analytiques ancrées dans la littérature en physiologie de l'effort et en data science.

### Public visé
Ce travail que j'ai d'abord réalisé pour moi, vise sans doute des personnes un peu "geek" qui aiment jouer avec des données. 

Il faut savoir manipuler un langage de programmation et si c'est du Python, c'est encore mieux. Il y a un peu de maths, normal, mais rien de très effrayant.


### Warning

Avertissement : je suis data scientist, pas spécialiste de physiologie de l'exercice. Ce que tu lis ici, c'est le carnet de bord d'un trailer curieux qui aime comprendre ses données — pas un conseil médical ou d'entraînement. 

Les analyses sont fournies à titre informatif uniquement ; je décline toute responsabilité quant à leur usage. Les sources sont là pour que tu puisses vérifier par toi-même.

---
### Notebooks

| Notebook | Description |
|---|---|
| `00_lire_donnees.ipynb` | Chargement et prétraitement des fichiers FIT + premiers graphiques|
| `02_terrain.ipynb` | Variabilité d'allure, GAP, splits positifs/négatifs |


> La liste des notebooks s'enrichira au fil du projet.


### Prérequis

```
python >= 3.9
fitparse
pandas
numpy
matplotlib
scipy
folium
```

### Utilisation

Cloner le dépôt et ouvrir un notebook dans Jupyter :

```bash
git clone https://github.com/GregS1t/trail-lab.git
cd trail-lab
jupyter lab
```

Placer les fichiers `.fit` dans le dossier `data/` et exécuter `00_dataset_builder.ipynb` en premier.

### Auteur

Grégory Sainton.

### Licence

- Licence         : CC BY-NC-SA 4.0
-                   https://creativecommons.org/licenses/by-nc-sa/4.0/

>    Vous êtes libre de partager et d'adapter ce travail, à condition de :
>     · citer l'auteur (BY)
>     · ne pas en faire un usage commercial (NC)
>     · redistribuer sous la même licence (SA)
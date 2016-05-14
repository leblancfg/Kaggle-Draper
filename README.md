# Kaggle-Draper
*Private repo*

C'est ici qu'iront les fichiers --  mais pas les images, elles sont trop grosses. Localement, j'ai les dossiers "test" et "train" au meme niveau que "Kaggle-Draper". Ce sera plus facile si on utilise tous cette meme convention-la.


Pour l'instant, j'ai commence a coder un petit script qui fera un "AKAZE stitching" pour chacune des permutations d'images par image set, i.e. 1&2, 1&3, 1&4, 1&5, 2&3, etc. En meme temps, ca creera un fichier .csv qui contient le *train target* en valeur binaire:
- 1 si l'image de gauche est plus vieille,
- 0 si l'image de droite est plus vieille.

On aura donc une liste de 20 probabilites chronologiques par *test image set*. Ce sera ensuite de l'algebre simple de cruncher ces valeurs-la pour arriver a une suite chronologique ordonnee, du genre (3,4,5,2,1). Essentiellement un probleme du genre "Alice est plus vieille que Jack, Antoine est plus jeune qu'Alice", etc. Voir sample_submission.csv pour voir quel format Kaggle accepte les solutions.

More, *way* more to come! Reste a trouver comment on va pouvoir splitter nos efforts a trois!
-fl

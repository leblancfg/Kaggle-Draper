# Kaggle-Draper

## Exposé

Voici notre solution pour le [Draper Satellite Image Chronology Challenge](https://www.kaggle.com/c/draper-satellite-image-chronology), sur Kaggle. En grandes lignes, nous utilisons les techniques suivantes pour résoudre le problème:
* Assemblage de photos par [BRISK](http://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html?highlight=brisk) et [Bruteforce-Hamming](http://docs.opencv.org/3.0-beta/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html?highlight=hamming).
* Déterminer l'homographie par [RANSAC](http://docs.opencv.org/3.0-beta/doc/tutorials/calib3d/table_of_content_calib3d/table_of_content_calib3d.html?highlight=ransac)
* Trouver les motifs dans les images susceptibles de déterminer la chronologie des images, en utilisant un [réseau neuronal convolutif](https://fr.wikipedia.org/wiki/R%C3%A9seau_neuronal_convolutif) avec l'architecture [VGG-16](https://github.com/albertomontesg/keras-model-zoo), en appelant [Tensorflow](https://www.tensorflow.org/) par [Keras](https://keras.io/).

Avec les outils suivants:
* Python 2.7.12 (devrait fonctionner avec 2.7.x)
* OpenCV (3.1.0 + contributor modules)
* Tensorflow 0.9
* Keras (master branch)


## Notes d'équipe
Localement, j'ai les dossiers "test" et "train" au meme niveau que "Kaggle-Draper". Ce sera plus facile si on utilise tous cette meme convention-la.

Pour l'instant, le script commence avec une variation sur [AKAZE stitching](https://www.kaggle.com/nigelcarpenter/draper-satellite-image-chronology/akaze-stitching "Kaggle script") pour chacune des permutations d'images par image set, i.e. 1&2, 1&3, 1&4, 1&5, 2&3, etc. En meme temps, ca creera un fichier .csv qui contient le *train target* en valeur binaire:
- 1 si l'image de gauche est plus vieille,
- 0 si l'image de droite est plus vieille.

On aura donc une liste de 20 probabilites chronologiques par *test image set*. Ce sera ensuite de l'algebre simple de cruncher ces valeurs-la pour arriver a une suite chronologique ordonnee, du genre (3,4,5,2,1). Essentiellement un probleme du genre "Alice est plus vieille que Jack, Antoine est plus jeune qu'Alice", etc. Voir sample_submission.csv pour voir quel format Kaggle accepte les solutions.

-fl

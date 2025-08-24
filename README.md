I- Contexte

Ce projet a pour objectif de développer un système intelligent capable de détecter 
deux affections cutanées courantes : l’acné et l’eczéma, à partir d’images. 
L’approche repose sur l’utilisation de techniques de deep learning et de modèles 
de vision par ordinateur pour automatiser le diagnostic, offrant ainsi un outil 
d’aide à la décision médicale accessible. 

II- Méthode de résolution 

La présente étude repose sur une approche d’apprentissage profond visant à 
développer un modèle de détection automatique de l’acné et de l’eczéma à partir 
d’images en niveaux de gris. L’objectif principal est de concevoir, entraîner et 
évaluer un modèle de classification performant, tout en intégrant une interface 
utilisateur permettant une interaction intuitive avec le système de diagnostic. 
1. Source des données 
Les images utilisées ont été téléchargées depuis la plateforme Kaggle, qui 
propose des bases d’images médicales annotées. Ces images, initialement placées 
dans un fichier d’extension zip, ont été décompressées pour leur utilisation.  
Voici la composition du dossier téléchargé : 
• 1294 images représentant des lésions d’acné ;  
• 1280 images représentant des lésions d’eczéma. 
Les images ont été par la suite organisées dans un dossier principal nommé Base, 
contenant deux sous-dossiers :    
• acné (pour les images d’acné) ;  
• eczéma (pour les images d’eczéma). 
Cette organisation est compatible avec la fonction ImageFolder de PyTorch.
2. Traitement et augmentation des données 
Pour préparer les images à l’apprentissage, les transformations suivantes ont 
été appliquées : 
• il a d’abord été procédé au redimensionnement des images à une taille 
uniforme de 224x224 pixels, afin de correspondre à l’entrée attendue par 
les modèles pré-entraînés ; 
• ensuite, s’en est suivie l’augmentation des données avec un retournement 
horizontal aléatoire (flip) pour améliorer la généralisation du modèle ; 
• enfin, les images ont été converties en tenseurs PyTorch et normalisation 
des canaux RGB avec les moyennes et écarts-types standards des jeux 
ImageNet, utilisés pour le pré-entraînement des réseaux. 
 
Ces étapes garantissent une meilleure robustesse du modèle et évitent le 
surapprentissage. 
3. Architecture du modèle 
Le modèle repose sur la combinaison de deux réseaux convolutifs profonds 
préentraînés sur ImageNet : 
• DenseNet121, dont la structure dense favorise la réutilisation des 
caractéristiques extraites et la réduction du nombre de paramètres. 
• EfficientNet-B0, reconnu pour son efficacité computationnelle et ses 
bonnes performances en classification d’images. 
Les couches de classification finales de ces deux réseaux ont été supprimées afin 
d’extraire uniquement les représentations intermédiaires (1024 pour 
DenseNet121, 1280 pour EfficientNet-B0). Ces représentations ont ensuite été 
concaténées et passées à travers un classifieur composé : 
• d’une couche entièrement connectée de 512 neurones avec activation 
ReLU,• d’un dropout (p = 0.5) pour réduire le surapprentissage, 
• d’une couche de sortie ajustée au nombre de classes. 
4. Paramètres et entraînement du modèle  
L’entraînement a été réalisé à l’aide de l’optimiseur Adam, réputé pour sa 
stabilité et son efficacité dans l’optimisation de réseaux profonds. Les 
hyperparamètres utilisés sont : 
• Taille de batch : 32, 
• Nombre d’époques : 10, 
• Taux d’apprentissage : 0,001. 
La fonction de coût utilisée est l'entropie croisée (cross-entropy loss), 
particulièrement adaptée aux tâches de classification multi-classes. 
L’apprentissage a été effectué sur GPU lorsque celui-ci était disponible, 
permettant de réduire significativement les temps de calcul. 
Durant chaque époque, le modèle a été entraîné en mode supervision complète. 
Pour chaque lot d’images, une passe avant (forward pass) a permis d’obtenir les 
prédictions, suivie du calcul de la perte et de la rétropropagation des gradients. 
Le modèle a ainsi ajusté ses poids de manière itérative pour minimiser l’erreur. 
5. Evaluation et sauvegarde du modèle 
À l’issue de la phase d'entraînement, le modèle a été évalué sur l’ensemble de 
validation, sans recalcul des gradients (mode eval). Les prédictions générées ont 
été comparées aux étiquettes réelles, et les performances ont été quantifiées via 
le rapport de classification (précision, rappel, F1-score) produit par la 
bibliothèque scikit-learn. 
Enfin, le modèle entraîné a été sauvegardé sous le format .pth à l’aide de la 
fonction torch.save, en vue d’une réutilisation ultérieure sans réentraînement. 

III- Présentation des résultats du modèle obtenu 

Les résultats obtenus témoignent d'une excellente capacité du modèle à 
distinguer les deux classes considérées, à savoir l’acné et l’eczéma. En effet, le 
modèle atteint une précision de 0.96 pour l’acné et de 0.98 pour l’eczéma, 
indiquant que la majorité des prédictions faites pour chaque classe sont 
correctes. Le rappel, quant à lui, s’élève à 0.98 pour l’acné et à 0.96 pour 
l’eczéma, ce qui reflète la capacité du modèle à identifier les cas pertinents dans 
les données. Le score F1, qui harmonise précision et rappel, est de 0.97 pour les 
deux classes, soulignant un équilibre remarquable entre les deux critères. 
L’exactitude globale du modèle est de 97 %, ce qui signifie que 515 images sur 
530 ont été correctement classées. Les moyennes macro et pondérée (macro avg 
et weighted avg) affichent également une valeur de 0.97, traduisant la stabilité et 
la robustesse du modèle, même en présence d’un léger déséquilibre de classes. 

IV- Interface utilisateur 

Pour faciliter l’utilisation du modèle par des utilisateurs non techniques, une 
interface interactive a été développée avec Streamlit. 
Les principales fonctionnalités qui y sont observables sont les suivantes : 
• Téléchargement d’image d’une lésion cutanée via un navigateur web ; 
• Visualisation des étapes de prétraitement appliquées à l’image 
(redimensionnement, retournement, normalisation) ;  
• Prédiction automatique avec affichage des probabilités pour chaque 
classe ;  
• Indication du diagnostic probable avec le niveau de confiance ;  
• Affichage de conseils pratiques selon la prédiction (nettoyage de la peau, 
consultation médicale, etc.) ;  
• Gestion du cas où les probabilités d’appartenance aux deux classes (acné 
et eczéma) sont toutes inférieures  à 95% avec affichage d’un diagnostic 
par défaut « Néant » pour plus de prudence.  
Cette interface permet un usage intuitif et informatif, avec un retour visuel utile 
à l’utilisateur final.

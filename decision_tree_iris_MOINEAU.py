# --- BLOC 1 : Inclusion des librairies ---
# Ce bloc importe les librairies nécessaires
# Matplotlib est utilisé pour créer des visualisations et des graphiques en 2D
# Scikit-learn est une bibliothèque libre Python destinée à l'apprentissage automatique
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix


# --- BLOC 2 : Importation des bases de données ---
#  L'ensemble de données Iris est chargé 
from sklearn import datasets
from sklearn.datasets import load_iris
iris = datasets.load_iris()


# --- BLOC 3 : Visualisation des observations et des variables ---
# Des informations sur les observations et les variables sont affichées.
print(iris.data)
iris.data
iris.target
iris.frame
iris.feature_names
iris.target_names
X, y = iris.data, iris.target
X.shape


# --- BLOC 4 : Séparation des données en entrainement et tests ---
# Sélection des données : 80% pour l'apprentissage, 20% pour la validation
# Les données sont séparées en ensembles d'entraînement et de test pour préparer l'apprentissage du modèle.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)


# --- BLOC 5 : Réalisation de l’entrainement par la méthode des arbres de décision ---
# Un arbre de décision est créé et entraîné à l'aide des données d'entraînement.
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)


# --- BLOC 6 : Visualisation de l’arbre de décisions ---
# On utilise la librairie matplotlib
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True)
r = export_text(clf, feature_names=iris['feature_names'])

print ("-----------aaaaaa---")
print (iris['feature_names'])
print ("-----------aaaaaa---")
print ("-----------aaaaaa---")
print ("-----------aaaaaa---")
print(r)
fig.savefig('decisiontree_train_80.jpg')
plt.show()


# --- BLOC 7 : Réalisation la prédiction et calcul du score  --- 
# Le modèle est utilisé pour prédire les étiquettes des données de test et le score est calculé pour évaluer la qualité du modèle
predictions = clf.predict(X_test)

# On peut de cette façon calculer le score en test :
print("score : ", clf.score(X_test, y_test))
print("matrice de confusion : ", confusion_matrix(y_test, predictions))
# On constate que le modèle est de très bonne qualité car il a un score de 1.0. 


# --- BLOC 8 : Changement des valeurs de paramètres max_depth et min_samples_leaf ---
# Le paramètre max_depth est un seuil sur la profondeur maximale de l’arbre. 
# Le paramètre min_samples_leaf donne le nombre minimal d’échantillons dans un nœud feuille. 
# Ils permettent de mettre des contraintes sur la construction de l’arbre et donc de contrôler indirectement le phénomène de sur-apprentissage.
clf = tree.DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 10)
clf.fit(X_train, y_train)

tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True)
r = export_text(clf, feature_names=iris['feature_names'])
print(r)
fig.savefig('decisiontree2_train_80.jpg')
plt.show()

# Interrogation sur les paramètres max_depth et min_samples_leaf
predictions = clf.predict(X_test)

# On peut de cette façon calculer le score en test :
print("score : ", clf.score(X_test, y_test))
print("matrice de confusion : ", confusion_matrix(y_test, predictions))
# Le modèle est de moins bonne qualité puisque le score est maintenant de 0.967.


# --- BLOC 9 : Réaliser la cross validation via un échantionnage 5 fold ---
# On décide de refaire le traitement avec 95% et 5% 
# En général on utilise une division apprentissage/test de type 80/20 ou 70/30 
# Mais comme ici le problème d’apprentissage est particulièrement simple nous prenons seulement 5% de la base comme échantillon d’apprentissage 
# (sinon, on risque de ne rien voir quand on modifie les paramètres).
 
# Inclusion des librairies
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

# Séparation des données en entrainement et tests 
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target,
            train_size=0.05, random_state=0)

best_score = float('-inf')
best_model = None
K = 5

# On fait varier le paramètre mdepth entre 1 et 7
for mdepth in [1, 2, 3, 4, 5, 6, 7]:
    clf = tree.DecisionTreeClassifier(max_depth=mdepth)
    clf = clf.fit(X_train, y_train)
    # Cross validation
    scores = cross_val_score(clf, X_test, y_test, cv=K)
    mean_score = scores.mean()
    print("Max Depth:", mdepth)
    print("Cross-validated scores:", scores)
    print("Mean Cross-validated score:", mean_score)
    # Pour récupérer le meilleur score et meilleur model 
    if mean_score > best_score:
        best_score = mean_score
        best_model = clf
    predictions = clf.predict(X_test)
    print(confusion_matrix(y_test,predictions))

# On fait varier le paramètre msplit entre 2 et 10
for msplit in [2, 3, 5, 10, 15, 20]:
    clf = tree.DecisionTreeClassifier(min_samples_split=msplit)
    clf = clf.fit(X_train, y_train)
    # Cross validation
    scores = cross_val_score(clf, X_test, y_test, cv=K)
    mean_score = scores.mean()
    print("Min Samples Split:", msplit)
    print("Cross-validated scores:", scores)
    print("Mean Cross-validated score:", mean_score)
    # Pour récupérer le meilleur score et meilleur model 
    if mean_score > best_score:
        best_score = mean_score
        best_model = clf
    predictions = clf.predict(X_test)
    print(confusion_matrix(y_test, predictions))


# --- BLOC 10 : Affichage du modèle qui a le meilleur score ---

# Affichage du meilleur score et meilleur model 
print("\n\nMeilleur score :", best_score)
print("\nModèle model :", best_model)

# Affichage du meilleur arbre de décision 
tree.plot_tree(best_model,
               feature_names = fn, 
               class_names=cn,
               filled = True)
r = export_text(clf, feature_names=iris['feature_names'])
print(r)
fig.savefig('best_decisiontree_train_0.95.jpg')
plt.show()

# On obtient donc ça comme résultats : 
# Meilleur score : 0.958128078817734
# Modèle : DecisionTreeClassifier(max_depth=3)

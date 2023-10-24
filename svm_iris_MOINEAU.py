"""
==================================================
Plot different SVM classifiers in the iris dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the iris
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.

.. NOTE:: while plotting the decision function of classifiers for toy 2D
   datasets can help get an intuitive understanding of their respective
   expressive power, be aware that those intuitions don't always generalize to
   more realistic high-dimensional problems.

"""

print(__doc__)


# --- BLOC 1 : Inclusion des librairies ---
# Ce bloc importe les librairies nécessaires
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
# Ajout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# --- BLOC 2 : Importation des bases de données ---
#  L'ensemble de données Iris est chargé 
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- BLOC 3 : SVM Instances and Model Fitting ---
# Plusieurs instances de modèles SVM sont créées avec différents noyaux et paramètres (linéaire, linéaire, RBF, polynomial, sigmoid).
# C : paramètre de régularisation pour le SVM.
# Les modèles sont ajustés  aux données d'entraînement (X et y).

# We do not scale our data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C), # SVM linear
          svm.LinearSVC(C=C, max_iter=10000), # SVM linear
          svm.SVC(kernel='rbf', gamma=0.7, C=C), # SVM rbf
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C), # SVM polynome
          # Ajout du sigmoid
          svm.SVC(kernel='sigmoid', gamma='auto', C=C) # SVM sigmoid
          )
models = (clf.fit(X_train, y_train) for clf in models)
models = list(models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel',
          # Ajout du sigmoid et RBF
          'SVC with sigmoid kernel',
          )

# --- BLOC 4 : Set-up Grid for Plotting --- 
# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(3, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Définir la plage des axes X et Y pour la grille
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# --- BLOC 5 : Plotting the Models --- 
for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(f'{title} - Mean Score: {np.mean(cross_val_score(clf, X_test, y_test, cv=5)):.2f}')

plt.show()


# --- BLOC 6 : Évaluation des Performances --- 
# Ajout
best_score = float('-inf')
best_model = None

for clf, title in zip(models, titles):
    print(clf, title)
    # Perform cross-validation
    predictions = clf.predict(X_test)
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    # Calculate accuracy
    mean_score = np.mean(scores)
    
    # Display the results
    print("Model:", title)
    print("Cross-validated mean score:", mean_score)
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:\n", cm)
    print("="*40)
    
    # Update the best model
    if mean_score > best_score:
        best_score = mean_score
        best_model = clf

# Display the best model
print("\n\nBest Score : ", best_score)
print("\nBest Model : ", best_model)
print("\n\n")

# On obtient donc ça comme résultats :
# Best Score :  0.9
# Best Model :  SVC(gamma='auto', kernel='poly')
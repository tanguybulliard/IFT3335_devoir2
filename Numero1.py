# -----------------------------------------------------------------------------
# IFT3335 - TP2 - Classification pour la désambiguisation de sens - Numero 1
# Tanguy Bulliard - 20126144
# Joey Van Melle - 20145502

# ---------------------------- TABLE OF CONTENT -------------------------------
# 1. Preparatifs
#   1.1. Preprocess
#   1.2. Classification
#       1.2.1 SVM
#       1.2.2 Reseau Neuronal
#       1.2.3 Arbre De Decision
#       1.2.4 Naive Bayes
#   1.3. Selection et ponderation des features
# -----------------------------------------------------------------------------


# 1.1. Preprocess -------------------------------------------------------------

from sklearn.datasets import load_iris

dataset = load_iris()
X = dataset.data  # La liste des données de chaque iris contenues dans des tableaux
y = dataset.target  # La liste des classifications valides des iris

# Creation des listes d'entrainements et de tests :
# X_train est la liste des données d'entrainement
# X_test est la liste des données de test contenues dans des tableaux
# y_train est la liste des classes pour l'entrainement
# y_test est la liste des classes pour le test.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Transformation des données en nombre entier pour éviter que certaine donnée
# apparaissant dans le test n'existe pas dans l'entrainement.

for tab in X_train:
    for index in range(len(tab)):
        if tab[index] - 0.5 < int(tab[index]):
            tab[index] = int(tab[index])
        else:
            tab[index] = int(tab[index]) + 1

for tab in X_test:
    for index in range(len(tab)):
        if tab[index] - 0.5 < int(tab[index]):
            tab[index] = int(tab[index])
        else:
            tab[index] = int(tab[index]) + 1

# 1.2. Classification ---------------------------------------------------------

# SVM

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

model = svm.SVC()
model.fit(X_train, y_train)  # Entrainement du modele SVM
modelPred = model.predict(X_test)  # Prediction du modele SVM

# print(classification_report(y_test, modelPred))
# print(confusion_matrix(y_test, modelPred))

# Reseau Neuronal

from sklearn.neural_network import MLPClassifier

network = MLPClassifier(hidden_layer_sizes=(7, 7, 7), max_iter=1000)
network.fit(X_train, y_train)  # Entrainement du reseau neuronal
networkPred = network.predict(X_test)  # Prediction du reseau neuronal

# print(classification_report(y_test, networkPred))
# print(confusion_matrix(y_test, networkPred))

# Arbre De Decision

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)  # Entrainement de l'arbre de decision
treePred = tree.predict(X_test)  # Prediction de l'arbre de decision

# print(classification_report(y_test, treePred))
# print(confusion_matrix(y_test, treePred))

# Naive Bayes

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train, y_train)  # Entrainement du Naive Bayes
nbPred = nb.predict(X_test)  # Prediction du Naive Bayes

# print(classification_report(y_test, nbPred))
# print(confusion_matrix(y_test, nbPred))

# 1.3. Selection et ponderation des features ----------------------------------

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_extraction.text import CountVectorizer

dataset = fetch_20newsgroups(subset='train')
print(dataset)
#cv = CountVectorizer(dataset)
#print(cv.vocabulary_)

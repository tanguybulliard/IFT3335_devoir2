# -----------------------------------------------------------------------------
# IFT3335 - TP2 - Classification pour la désambiguisation de sens - Numero 2
# Tanguy Bulliard - 20126144
# Joey Van Melle - 20145502

# ---------------------------- TABLE OF CONTENT -------------------------------
# 2. La tache de desambiguisation de sens de mots
#   2.1. Preparation des données
#   2.2. Naive Bayes
#   2.3. Arbre de decision
#   2.4. MultiLayerPerceptron
#   2.5. Naive Bayes avec négligence des mots outils
#   2.6. Arbre de decision avec négligence des mots outils
#   2.7. MultiLayerPerceptron avec négligence des mots outils
#   2.8. Naive Bayes avec troncature des mots
#   2.9. Arbre de decision avec troncature des mots
#   2.10. MultiLayerPerceptron troncature des mots
#   2.11 Tests
# -----------------------------------------------------------------------------

import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize



# 2. La tache de desambiguisation de sens de mots -----------------------------

# Tache : trouver le sens du mot interest dans une phrase.

# 2.1. Preparation des données ------------------------------------------------

# Création d'une liste contenant chaque ligne du texte
with open('text.txt') as txt:
    lineList = txt.read().split('$$')

# Retrait des lignes ne contenant pas le mot "interest"
newList = list()
for line in lineList:
    if 'interest' in line:
        newList.append(line)

lineList = newList

# Création d'une liste (wordList) contenant une liste des mots de chaque ligne
wordList = list()
for line in lineList:
    separatedLine = line.split(" ")
    wordListLine = list()
    for word in separatedLine:
        if '/' in word:
            index = word.find('/')
            w = word[0: index]
            if (word[index + 1:len(word)]).isalpha():
                wordListLine.append(w)
    wordList.append(wordListLine)


# fonction pour créer une liste des bags contenant les n mots avant et après le mot "interest".


def create_bags(n):
    bag_list = list()
    index = 0
    for line in wordList:
        for word in line:
            if "interest" in word:
                index = line.index(word)
        bag = dict()
        j = 0
        for i in range(index - n, index + 1 + n):
            if 0 <= i < len(line) and i != index:
                bag.update({j: line[i]})
                j += 1
        bag_list.append(bag)
    return bag_list


# Création d'une liste (catList) contenant une liste de la catégorie grammaticale  des mots de chaque ligne
catList = list()
for line in lineList:
    separatedLine = line.split(" ")
    catListLine = list()
    for word in separatedLine:
        if '/' in word:
            index = word.find('/')
            w = word[index + 1: len(word)]
            if '\n' in w:
                w = w[0:len(w)-1]
            catListLine.append(w)
    catList.append(catListLine)

# Création de la liste (indexList) contenant l'index de la position du mot "interest"
indexList = list()
for line in lineList:
    separatedLine = line.split(" ")
    index = 0
    for word in separatedLine:
        if "interest" in word:
            indexList.append(index)
            break
        else:
            index += 1

# fonction pour créer une liste des listes contenant les catégories grammaticales des n mots avant
# et après le mot "interest".


def create_gram_lists(n):
    gram_list = list()
    for line, index in zip(catList, indexList):
        before = list()
        for i in range(index - n, index):
            if 0 <= i < len(line) and i != index:
                before.append(line[i])
        after = list()
        for i in range(index + 1, index + 1 + n):
            if 0 <= i < len(line) and i != index:
                after.append(line[i])

        while len(before) < n:
            before.insert(0, 0)
        while len(after) < n:
            after.append(0)
        gram_list.append([before, after])
    return gram_list #une liste contenant des tableaux [0]: catégorie des mots avant interest,
                    # [1] catégorie des mots après


# Création d'une liste indiquant le sens de "interest" dans la phrase
y = list()
for line in lineList:
    index = line.find("_") + 1
    y.append(line[index])

# Numérisation des mots


def numerise_word(n):
    str_list = list()
    bags = create_bags(n)
    for bag in bags:
        str = " ".join(bag.values())
        str_list.append(str)
    cv = CountVectorizer()
    return cv.fit_transform(str_list)


# Numérisation des mots en utilisant des stopwords


def numerise_word_with_stopwords(n):
    str_list = list()
    bags = create_bags(n)
    for bag in bags:
        str = " ".join(bag.values())
        str_list.append(str)
    cv = CountVectorizer(stop_words='english')
    return cv.fit_transform(str_list)

# Numérisation des mots en utilisant un algorithme de stemming

def numerise_word_with_stemming(n):
    ps = PorterStemmer()
    str_list = list()
    bags = create_bags(n)
    for bag in bags:
        stemmed_words = list()
        for word in bag.values():
            stemmed_words.append(ps.stem(word))
        str = " ".join(stemmed_words)
        str_list.append(str)
    cv = CountVectorizer()
    return cv.fit_transform(str_list)

# Numérisation des catégories grammaticales des mots


def numerise_cat(n):
    # Creation d'un dictionnaire contenant toute les categories grammaticales possibles
    catDict = dict()
    gram_list = create_gram_lists(n)
    for line in gram_list:
        for cat in line[0]:
            catDict.update({cat: 0})
        for cat in line[1]:
            catDict.update({cat: 0})

    # Association d'un nombre a chaque categorie
    i = 0
    for key in catDict.keys():
        catDict.update({key: i})
        i += 1

    # Numerisation des attributs
    catX = list()
    for line in gram_list:
        array = list()
        for cat in line[0]:
            array.append(catDict.get(cat))
        for cat in line[1]:
            array.append(catDict.get(cat))
        catX.append(array)
    return catX


# 2.2 Naive Bayes -------------------------------------------------------------
# 2.2.1 Naive Bayes avec les sacs de mots

def naive_bayes(n):
    start_time = time.time()
    X = numerise_word(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    nb = MultinomialNB()
    nb.fit(X_train, y_train)  # Entrainement du Naive Bayes
    nbPred = nb.predict(X_test)  # Prediction du Naive Bayes

    print('Result for Naive Bayes of size ' + str(n) + ' : ')
    print(classification_report(y_test, nbPred))
    print(confusion_matrix(y_test, nbPred))
    print("--- %s seconds ---" % (time.time() - start_time))

# 2.2.2 Naive Bayes avec les catégories grammaticales


def naive_bayes_gram(n):
    start_time = time.time()
    X = numerise_cat(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    nb = MultinomialNB()
    nb.fit(X_train, y_train)  # Entrainement du Naive Bayes
    nbPred = nb.predict(X_test)  # Prediction du Naive Bayes

    print('Result for Naive Bayes Gram of size ' + str(n) + ' : ')
    print(classification_report(y_test, nbPred))
    print(confusion_matrix(y_test, nbPred))
    print("--- %s seconds ---" % (time.time() - start_time))

# 2.3. Arbre de decision -------------------------------------------------------------
# 2.3.1 Arbre de decision avec les sacs de mots


def tree(n):
    start_time = time.time()
    X = numerise_word(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)  # Entrainement de l'arbre de decision
    treePred = tree.predict(X_test)  # Prediction de l'arbre de decision

    print('Result for Decision Tree of size ' + str(n) + ' : ')
    print(classification_report(y_test, treePred))
    print(confusion_matrix(y_test, treePred))
    print("--- %s seconds ---" % (time.time() - start_time))

# 2.3.2 Arbre de decision avec les catégories grammaticales


def tree_gram(n):
    start_time = time.time()
    X = numerise_cat(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)  # Entrainement de l'arbre de decision
    treePred = tree.predict(X_test)  # Prediction de l'arbre de decision

    print('Result for Decision Tree Gram of size ' + str(n) + ' : ')
    print(classification_report(y_test, treePred))
    print(confusion_matrix(y_test, treePred))
    print("--- %s seconds ---" % (time.time() - start_time))

# 2.4. Reseau Neuronnal -------------------------------------------------------------
# 2.4.1 Reseau Neuronnal avec les sacs de mots


def network(n, m):
    start_time = time.time()
    X = numerise_word(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    network = MLPClassifier(hidden_layer_sizes=(m, m, m), max_iter=1000)
    network.fit(X_train, y_train)  # Entrainement du reseau neuronal
    networkPred = network.predict(X_test)  # Prediction du reseau neuronal

    print('Result for Neural Network of size ' + str(n) + ', with ' + str(m) + ' layers : ')
    print(classification_report(y_test, networkPred))
    print(confusion_matrix(y_test, networkPred))
    print("--- %s seconds ---" % (time.time() - start_time))

# 2.4.2 Reseau Neuronnal avec les catégories grammaticales


def network_gram(n, m):
    start_time = time.time()
    X = numerise_cat(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    network = MLPClassifier(hidden_layer_sizes=(m, m, m), max_iter=1000)
    network.fit(X_train, y_train)  # Entrainement du reseau neuronal
    networkPred = network.predict(X_test)  # Prediction du reseau neuronal

    print('Result for Neural Network Gram of size ' + str(n) + ', with ' + str(m) + ' layers : ')
    print(classification_report(y_test, networkPred))
    print(confusion_matrix(y_test, networkPred))
    print("--- %s seconds ---" % (time.time() - start_time))

# 2.5. Naive Bayes avec négligence des mots outils ----------------------------------


def naive_bayes_stopword(n):
    start_time = time.time()
    X = numerise_word_with_stopwords(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    nb = MultinomialNB()
    nb.fit(X_train, y_train)  # Entrainement du Naive Bayes
    nbPred = nb.predict(X_test)  # Prediction du Naive Bayes

    print('Result for Naive Bayes With Stopword of size ' + str(n) + ' : ')
    print(classification_report(y_test, nbPred))
    print(confusion_matrix(y_test, nbPred))
    print("--- %s seconds ---" % (time.time() - start_time))

# 2.6. Arbre de decision avec négligence des mots outils ----------------------------


def tree_stopword(n):
    start_time = time.time()
    X = numerise_word_with_stopwords(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)  # Entrainement de l'arbre de decision
    treePred = tree.predict(X_test)  # Prediction de l'arbre de decision

    print('Result for Decision Tree With Stopword of size ' + str(n) + ' : ')
    print(classification_report(y_test, treePred))
    print(confusion_matrix(y_test, treePred))
    print("--- %s seconds ---" % (time.time() - start_time))

# 2.7. MultiLayerPerceptron avec négligence des mots outils -------------------------


def network_stopword(n, m):
    start_time = time.time()
    X = numerise_word_with_stopwords(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    network = MLPClassifier(hidden_layer_sizes=(m, m, m), max_iter=1000)
    network.fit(X_train, y_train)  # Entrainement du reseau neuronal
    networkPred = network.predict(X_test)  # Prediction du reseau neuronal

    print('Result for Neural Network With Stopword of size ' + str(n) + ', with ' + str(m) + ' layers : ')
    print(classification_report(y_test, networkPred))
    print(confusion_matrix(y_test, networkPred))
    print("--- %s seconds ---" % (time.time() - start_time))

# 2.8. Naive Bayes avec troncature des mots -----------------------------------------


def naive_bayes_stemming(n):
    start_time = time.time()
    X = numerise_word_with_stemming(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    nb = MultinomialNB()
    nb.fit(X_train, y_train)  # Entrainement du Naive Bayes
    nbPred = nb.predict(X_test)  # Prediction du Naive Bayes

    print('Result for Naive Bayes With Stemming of size ' + str(n) + ' : ')
    print(classification_report(y_test, nbPred))
    print(confusion_matrix(y_test, nbPred))
    print("--- %s seconds ---" % (time.time() - start_time))

# 2.9. Arbre de decision avec troncature des mots -----------------------------------


def tree_stemming(n):
    start_time = time.time()
    X = numerise_word_with_stemming(2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)  # Entrainement de l'arbre de decision
    treePred = tree.predict(X_test)  # Prediction de l'arbre de decision

    print('Result for Decision Tree With Stemming of size ' + str(n) + ' : ')
    print(classification_report(y_test, treePred))
    print(confusion_matrix(y_test, treePred))
    print("--- %s seconds ---" % (time.time() - start_time))

# 2.10. MultiLayerPerceptron troncature des mots ------------------------------------


def network_stemming(n, m):
    start_time = time.time()
    X = numerise_word_with_stemming(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    network = MLPClassifier(hidden_layer_sizes=(m, m, m), max_iter=1000)
    network.fit(X_train, y_train)  # Entrainement du reseau neuronal
    networkPred = network.predict(X_test)  # Prediction du reseau neuronal

    print('Result for Neural Network With Stemming of size ' + str(n) + ', with ' + str(m) + ' layers : ')
    print(classification_report(y_test, networkPred))
    print(confusion_matrix(y_test, networkPred))
    print("--- %s seconds ---" % (time.time() - start_time))


# 2.11. Tests -----------------------------------------------------------------------
for n in range(1, 5):
    naive_bayes(n)
    naive_bayes_gram(n)
    naive_bayes_stopword(n)
    naive_bayes_stemming(n)

    tree(n)
    tree_gram(n)
    tree_stopword(n)
    tree_stemming(n)

    for m in range(7, 11):
        network(n, m)
        network_gram(n, m)
        network_stopword(n, m)
        network_stemming(n, m)

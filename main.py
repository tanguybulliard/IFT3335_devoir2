"""
https://medium.com/@rangavamsi5/na%C3%AFve-bayes-algorithm-implementation-from-scratch-in-python-7b2cc39268b9
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def replaceName(a,stop_list,y,x):
    my_listX = []
    words=[]
    i=-1

    for k in a.split("\n"):
        b = re.sub(r"[^a-z-A-Z-0-9.%_']+", ' ', k)#on filtre les caractères pour ne garder que les valeurs numeriques, alphabetique ainsi que _.%
        bb=re.sub(r"[^a-z-0-9.%_']+", ' ', k)

        words=b.split()

        if (len(words) > 1):
            i += 1
            my_listX.append(bb)

            for kk in stop_list:
                if kk in words:
                    try:
                        while True:
                            index = words.index(kk)
                            words.remove(kk)  # on enlève les stops words
                            words.pop(index)  # on enlève son groupe avec
                    except ValueError:
                        pass

                else:
                    pass
            root_verb=['VBZ','VBP','VBG','VBD','VBN']
            root_noun=['NNS']
            ps = PorterStemmer()
            stemmer = SnowballStemmer("english")



            for k in root_verb:
                if k in words:
                    index = words.index(k)
                    words.insert(index,'VB')#on change le groupe en VB
                    words.pop(index+1)
                    w=words[index-1]
                    rootWord = ps.stem(w)
                    words.insert(index-1,rootWord)
                    words.pop(index)
                else:
                    pass

            for k in root_noun:
                if k in words:
                    index = words.index(k)
                    words.insert(index,'NN')#on change le groupe en VB
                    words.pop(index+1)
                    w=words[index-1]
                    rootWord = stemmer.stem(w)
                    words.insert(index-1,rootWord)
                    words.pop(index)
                else:
                    pass


            toBeRemoved = ["''"]
            removed = toBeRemoved[0]
            try:
                while True:
                    words.remove(removed)
            except ValueError:
                pass


            interest=['interest_1', 'interests_1','interest_2', 'interests_2','interest_3', 'interests_3','interest_4', 'interests_4','interest_5', 'interests_5','interest_6', 'interests_6']
            for k in interest:
                if(k in words):
                    index = words.index(k)
                    lenght = len(k)-1
                    interest=k[lenght]

                    y[i][1] = interest

                else:
                    pass
            # _________________TRAINING DATA_____________



            
    
        else:
            pass

    #________________DATA PREPROCESS____________________
    train_x,test_x=train_test_split(my_listX,test_size=0.7,shuffle=False)
    print(train_x[0])
    split = len(train_x)
    train_y, test_y = y[:split, :], y[split:, :]

    vectorizer=CountVectorizer()
    #vectorizer.fit_transform(train_x)
    train_x_vectors=vectorizer.fit_transform(train_x)




    #___________BAG OF WORDS___________
    vectorizer = TfidfVectorizer()
    train_x_vectors = vectorizer.fit_transform(train_x)
    test_x_vectors = vectorizer.transform(test_x)

    #_________DATA TRANSFORMATION______

    train_y=pd.Series(train_y)
    test_y=pd.Series(test_y)
    train_x_vectors = pd.DataFrame(train_x_vectors.todense())
    """__________DECISION TREE__________

    			ça bug ici

    clf_dec = DecisionTreeClassifier()
    print(type(train_x_vectors))
    print(type(train_y))
    clf_dec.fit(train_x_vectors, train_y)



    print(clf_dec.score(test_x_vectors, test_y))

    """







if __name__ == '__main__':

    a = open('corpus.txt').read()
    stop_words=open('stop_words.txt').read()# j'ai ajouté 'n''although'' n't'''s' à la liste téléchargé depuis studium
    stop_list=[]
    for k in stop_words.split("\n"):
        stop_list.append(k)

    y = np.zeros((2368, 2))
    for x in range(2368):
        y[x][0]=x

    x =np.zeros((2368,9))
    for k in range(2368):
        x[k][0]=k

    replaceName(a,stop_list,y,x)








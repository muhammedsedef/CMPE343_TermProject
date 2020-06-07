
from django.shortcuts import render
from django.views.generic import TemplateView
import pandas as pd
import numpy as np
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import textblob
from textblob import Word
nltk.download("wordnet")
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import cross_val_score

def executeMyModel():
    import pandas as pd
    df = pd.read_csv("yelp.csv")
    df = df[["stars", "text"]]
    preprocessing = df["text"]
    preprocessing = preprocessing.str.replace("[^\w\s]"," ")
    preprocessing = preprocessing.str.replace("\d","")
    preprocessing = preprocessing.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords.words("english")))
    preprocessing = preprocessing.apply(lambda x: " ".join(WordNetLemmatizer().lemmatize(i,"v") 
                                                       for i in x.split()))
    df["text"] = preprocessing
    yelp = df[["text","stars"]]
    df = yelp.copy()
    X = df["text"]
    y = df["stars"]
    X_upsampled1, y_upsampled1 = resample(X[y == 1],
                                    y[y == 1],
                                    replace=True,
                                    n_samples=X[y == 4].shape[0],
                                    random_state=123)

    X_upsampled2, y_upsampled2 = resample(X[y == 2],
                                     y[y == 2],
                                     replace=True,
                                     n_samples=X[y == 4].shape[0],
                                     random_state=123)

    X_upsampled3, y_upsampled3 = resample(X[y == 3],
                                     y[y == 3],
                                     replace=True,
                                     n_samples=X[y == 4].shape[0],
                                     random_state=123)

    X_upsampled5, y_upsampled5 = resample(X[y == 5],
                                     y[y == 5],
                                     replace=True,
                                     n_samples=X[y == 4].shape[0],
                                     random_state=123)
    X_bal = np.hstack((X[y == 4], X_upsampled1, X_upsampled2, X_upsampled3, X_upsampled5))
    y_bal = np.hstack((y[y == 4], y_upsampled1, y_upsampled2, y_upsampled3, y_upsampled5))
    X_train, X_test, y_train, y_test = \
        train_test_split(X_bal, y_bal, 
                     test_size=0.20,
                     stratify=y_bal,
                     random_state=1)
    yelp = pd.DataFrame(X_bal,columns=["text"])  
    yelp2 = pd.DataFrame(y_bal,columns=["stars"])
    yelp = yelp.join(yelp2)
    df = yelp.copy()
    df.groupby("stars")
    X = df["text"]
    y = df["stars"]
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scores = cross_val_score(RandomForestClassifier(n_estimators=40), X, y, cv=10)
    mean = np.mean(scores)
    return mean



class AnasayfaSayfaGorunumu(TemplateView):

    def get(self, request, **kwargs):
        mydf = executeMyModel()
        date_dict = {"access_records": mydf}
        return render(request,'anasayfa.html',context=date_dict)

class HakkimdaSayfaGorunumu(TemplateView):
    template_name="hakkımda.html"

class IletisimSayfaGorunumu(TemplateView):
    template_name="iletişim.html"
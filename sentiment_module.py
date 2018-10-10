import nltk


from nltk.classify.scikitlearn import SklearnClassifier
import random
#nltk.download('punkt')
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import io
import pickle

class VoteClassifier(ClassifierI):
  def __init__(self,*classifiers):
    self.classifiers=classifiers

  def classify(self,features):
    votes=[]
    for c in self.classifiers:
      v=c.classify(features)
      votes.append(v)
    return mode(votes)

  def confidence(self,features):
    votes=[]
    for c in self.classifiers:
      v=c.classify(features)
      votes.append(v)
    choice_votes=votes.count(mode(votes))
    conf=choice_votes/len(votes)
    return conf


saved_features=open('/home/ved/Desktop/word_features.pickle','rb')
word_features=pickle.load(saved_features)
saved_features.close()

def find_features(document):
  words=word_tokenize(document.lower())
  features={}
  for w in word_features:
    features[w]=(w in words)
  return features



saved_featuresets=open('/home/ved/Desktop/featuresets.pickle','rb')
featuresets=pickle.load(saved_featuresets)
saved_featuresets.close()

random.shuffle(featuresets)



saved_classifier=open('/home/ved/Desktop/classifier.pickle','rb')
classifier=pickle.load(saved_classifier)
saved_classifier.close()


saved_MNB=open('/home/ved/Desktop/MNB_classifier.pickle','rb')
MNB_classifier=pickle.load(saved_MNB)
saved_MNB.close()



saved_Bernoulli=open('/home/ved/Desktop/BernoulliNB_classifier.pickle','rb')
BernoulliNB_classifier=pickle.load(saved_Bernoulli)
saved_Bernoulli.close()


saved_Logistic=open('/home/ved/Desktop/LogisticRegression_classifier.pickle','rb')
LogisticRegression_classifier=pickle.load(saved_Logistic)
saved_Logistic.close()


saved_SGD=open('/home/ved/Desktop/SGDClassifier_classifier.pickle','rb')
SGDClassifier_classifier=pickle.load(saved_SGD)
saved_SGD.close()



saved_LinearSVC=open('/home/ved/Desktop/LinearSVC_classifier.pickle','rb')
LinearSVC_classifier=pickle.load(saved_LinearSVC)
saved_LinearSVC.close()


saved_NuSVC=open('/home/ved/Desktop/NuSVC_classifier.pickle','rb')
NuSVC_classifier=pickle.load(saved_NuSVC)
saved_NuSVC.close()


voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


def sentiment(features):
  featureset=find_features(features)
  classification=voted_classifier.classify(featureset)
  conf=voted_classifier.confidence(featureset)
  return classification, conf

   
  
    

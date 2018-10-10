import nltk
#nltk.download('averaged_perceptron_tagger')
import pickle

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

fp=io.open('/home/ved/Desktop/positive.txt',encoding='latin-1')
positive_sentiments=fp.read()
fp.close()
fp=io.open('/home/ved/Desktop/negative.txt',encoding='latin-1')
negative_sentiments=fp.read()
fp.close()

all_words=[]
documents=[]
allowed_word_types=["J"]
for l in positive_sentiments.split('\n'):
  documents.append((l,"pos"))
  words=word_tokenize(l)
  pos=nltk.pos_tag(words)
  for p in pos:
    if p[1][0] in allowed_word_types:
     all_words.append(p[0].lower())

for l in negative_sentiments.split('\n'):
  documents.append((l,"neg"))
  words=word_tokenize(l)
  pos=nltk.pos_tag(words)
  for p in pos:
    if p[1][0] in allowed_word_types:
     all_words.append(p[0].lower())



all_words=nltk.FreqDist(all_words)
word_features=list(all_words.keys())[:5000]

save_features=open('/home/ved/Desktop/word_features.pickle','wb')
pickle.dump(word_features,save_features)
save_features.close()

def find_features(document):
  words=word_tokenize(document.lower())
  features={}
  for w in word_features:
    features[w]=(w in words)
  return features

featuresets=[(find_features(rev),category) for (rev,category) in documents]
random.shuffle(featuresets)

save_featuresets=open('/home/ved/Desktop/featuresets.pickle','wb')
pickle.dump(featuresets,save_featuresets)
save_featuresets.close()

training_set=featuresets[:10000]
testing_set=featuresets[10000:]

classifier=nltk.NaiveBayesClassifier.train(training_set)
print('Original classifier accuracy is :', (nltk.classify.accuracy(classifier,testing_set))*100)

save_classifier=open('/home/ved/Desktop/classifier.pickle','wb')
pickle.dump(classifier,save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_MNB=open('/home/ved/Desktop/MNB_classifier.pickle','wb')
pickle.dump(MNB_classifier,save_MNB)
save_MNB.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_Bernoulli=open('/home/ved/Desktop/BernoulliNB_classifier.pickle','wb')
pickle.dump(BernoulliNB_classifier,save_Bernoulli)
save_Bernoulli.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_Logistic=open('/home/ved/Desktop/LogisticRegression_classifier.pickle','wb')
pickle.dump(LogisticRegression_classifier,save_Logistic)
save_Logistic.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

save_SGD=open('/home/ved/Desktop/SGDClassifier_classifier.pickle','wb')
pickle.dump(SGDClassifier_classifier,save_SGD)
save_SGD.close()
# Due to low accuracy it is ignored
#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(training_set)
#print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_LinearSVC=open('/home/ved/Desktop/LinearSVC_classifier.pickle','wb')
pickle.dump(LinearSVC_classifier,save_LinearSVC)
save_LinearSVC.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

save_NuSVC=open('/home/ved/Desktop/NuSVC_classifier.pickle','wb')
pickle.dump(NuSVC_classifier,save_NuSVC)
save_NuSVC.close()


voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)



  
    

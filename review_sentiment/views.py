from django.http import  HttpResponse
from django.shortcuts import render
import joblib
import spacy
from spacy import displacy


def home(request):
    return render(request,'home.html')

def result(request):

    classifier = joblib.load('final_sentiment_model.pkl')
    cv = joblib.load('vect-transform.pkl')
    s = request.GET['email_subject']

    # print(s)
    # s = [s]

    from spacy.lang.en.stop_words import STOP_WORDS
    stopwords = list(STOP_WORDS)

    import en_core_web_sm
    nlp = en_core_web_sm.load()
    doc = nlp(s)

    import string
    punct = string.punctuation

    tokens = []
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    
    cleaned_tokens = ""
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens+=" "
            cleaned_tokens+=token  

    vect = cv.transform([cleaned_tokens]).toarray()
    my_prediction = classifier.predict(vect)
    # ans = cls.predict(s)

    res = ""
    if(my_prediction==0):
        res="OOPS!!!! You Got A BAD REVIEW"
    else:
        res="Congrats It's A GOOD REVIEW"

    return render(request,'result.html',{'res':res})
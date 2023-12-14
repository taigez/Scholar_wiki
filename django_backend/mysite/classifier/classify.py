from cProfile import label
import re
import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
import requests
import urllib
from requests_html import HTML
from requests_html import HTMLSession
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from .models import Awd_data, Edu_data, Int_data, RawA, RawE, RawI
from .models import Sentences_awd, Sentences_edu, Sentences_int, Sentences_temp_int, Sentences_temp_awd, Sentences_temp_edu, Sentences_irr_awd, Sentences_irr_edu, Sentences_irr_int

import spacy

training_round = 200


with open('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/version.txt') as f:
    output = [line.strip() for line in f.readlines()]

awards_ver = output[2]
edu_ver = output[5]
interest_ver = output[8]

amodel_add = output[0] + awards_ver + '.h5'
adata_add = output[1] + awards_ver + '.csv'
emodel_add = output[3] + edu_ver + '.h5'
edata_add = output[4] + edu_ver + '.csv'
imodel_add = output[6] + interest_ver + '.h5'
idata_add = output[7] + interest_ver + '.csv'

awards_model = keras.models.load_model(amodel_add, custom_objects={'KerasLayer':hub.KerasLayer})
edu_model =  keras.models.load_model(emodel_add, custom_objects={'KerasLayer':hub.KerasLayer})
interest_model = keras.models.load_model(imodel_add, custom_objects={'KerasLayer':hub.KerasLayer})

def get_source(url):
    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)

def scrape_google(query):

    query = urllib.parse.quote_plus(query)
    response = get_source("https://www.google.com/search?q=" + query)

    links = list(response.html.absolute_links)
    google_domains = ('https://www.google.', 
                      'https://google.', 
                      'https://webcache.googleusercontent.', 
                      'http://webcache.googleusercontent.', 
                      'https://policies.google.',
                      'https://support.google.',
                      'https://maps.google.')

    for url in links[:]:
        # remove google domains
        if url.startswith(google_domains):
            links.remove(url)

        # remove non edu links
        elif 'linkedin' in url:
            links.remove(url)
            
    return links

def read_from_link(url):
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
    req = Request(url, headers=hdr)
    try:
        html = urlopen(req).read()
        soup = BeautifulSoup(html, features="html.parser")
        print(soup.title.get_text())
        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = soup.body.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # drop blank lines
        full_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # print(full_text)
        return full_text
    except:
        pass

def summarize_text(links):
    for link in links:
        print(link)
        full_text = read_from_link(link)
        sens = split_sen(full_text)
        for sen in sens:
            if len(sen) > 500:
                continue
            sen = sen.replace('\xa0', ' ')
            sen = sen.replace('\u00a0', ' ')
            ascore = awards_model.predict([sen])[0][0]
            escore = edu_model.predict([sen])[0][0]
            iscore = interest_model.predict([sen])[0][0]
            if  escore > 0.5 and escore == max(ascore, escore, iscore):
                new_edu = RawE(body=sen)
                new_edu.save()
            
            elif  iscore > 0.5 and iscore == max(ascore, escore, iscore):
                new_int = RawI(body=sen)
                new_int.save()
            
            elif  ascore > 0.5 and ascore == max(ascore, escore, iscore):
                new_awd = RawA(body=sen)
                new_awd.save()


def split_sen(text):
    sentences = []
    nlp = spacy.load("en_core_web_lg")
    if text != None:
        doc = nlp(text)
        for sent in doc.sents:
            sentences.append(sent.text)
    return sentences

def process_paragraph(text):
    final_dict = {}
    background = []
    interest = []
    awards = []

    sens = split_sen(text)
    for sen in sens:
        sen = sen.replace('\xa0', ' ')
        sen = sen.replace('\u00a0', ' ')
        if edu_model.predict([sen])[0][0] > 0.5:
            background.append(sen)
        
        if interest_model.predict([sen])[0][0] > 0.5:
            interest.append(sen)
        
        if awards_model.predict([sen])[0][0] > 0.5:
            awards.append(sen)

    final_dict["background"] = background
    final_dict["interest"] = interest
    final_dict["awards"] = awards

    return final_dict, len(sens)

def train_awd():
    global awards_ver, output

    # store related 
    for sen in Sentences_awd.objects.all():
        new_awd_data = Awd_data(weight=0, label=1, text=sen.body)
        new_awd_data.save()
    Sentences_awd.objects.all().delete()

    #store unrelated
    for sen in Sentences_irr_awd.objects.all():
        new_awd_data = Awd_data(weight=1, label=0, text=sen.body)
        new_awd_data.save()
    Sentences_irr_awd.objects.all().delete()

    related_l = []
    related_t = []
    unrelated_wl = []
    unrelated_wt = []
    unrelated_l = []
    unrelated_t = []
    for item in Awd_data.objects.raw('SELECT * FROM "classifier_awd_data"'):
        if item.label == 1:
            related_l.append(1)
            related_t.append(item.text)
        elif item.weight == 1:
            unrelated_wl.append(0)
            unrelated_wt.append(item.text)
        else:
            unrelated_l.append(0)
            unrelated_t.append(item.text)
    
    related = {'Awards': related_l, 'Text': related_t}
    unrelated_w = {'Awards': unrelated_wl, 'Text': unrelated_wt}
    unrelated = {'Awards': unrelated_l, 'Text': unrelated_t}
    df_related = pd.DataFrame(data=related)
    df_unrelated = pd.DataFrame(data=unrelated)
    df_unrelated_w = pd.DataFrame(data=unrelated_w)

    if df_unrelated_w.shape[0] < df_related.shape[0]:
        df_down = df_unrelated.sample(df_related.shape[0] - df_unrelated_w.shape[0])
        df_unrelated_w = pd.concat([df_down, df_unrelated_w])

    df_balanced = pd.concat([df_unrelated_w, df_related])

    X_train, X_test, y_train, y_test = train_test_split(df_balanced['Text'], df_balanced['Awards'], stratify=df_balanced['Awards'])
    
    awards_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/backups/amodel_' + awards_ver + '.h5')
    awards_model.fit(X_train, y_train, epochs=training_round)
    os.remove('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/amodel_' + awards_ver + '.h5')
    awards_ver = (str)((int)(awards_ver) + 1)
    awards_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/amodel_' + awards_ver + '.h5')
    
    output[2] = awards_ver
    with open('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/version.txt', "w") as f:
        for item in output:
            f.write(str(item) + '\n')


def train_edu():
    global edu_ver, output

    # store related 
    for sen in Sentences_edu.objects.all():
        new_edu_data = Edu_data(weight=0, label=1, text=sen.body)
        new_edu_data.save()
    Sentences_edu.objects.all().delete()

    #store unrelated
    for sen in Sentences_irr_edu.objects.all():
        new_edu_data = Edu_data(weight=1, label=0, text=sen.body)
        new_edu_data.save()
    Sentences_irr_edu.objects.all().delete()

    related_l = []
    related_t = []
    unrelated_wl = []
    unrelated_wt = []
    unrelated_l = []
    unrelated_t = []
    for item in Edu_data.objects.raw('SELECT * FROM "classifier_edu_data"'):
        if item.label == 1:
            related_l.append(1)
            related_t.append(item.text)
        elif item.weight == 1:
            unrelated_wl.append(0)
            unrelated_wt.append(item.text)
        else:
            unrelated_l.append(0)
            unrelated_t.append(item.text)
    
    related = {'Ed': related_l, 'Text': related_t}
    unrelated_w = {'Ed': unrelated_wl, 'Text': unrelated_wt}
    unrelated = {'Ed': unrelated_l, 'Text': unrelated_t}
    df_related = pd.DataFrame(data=related)
    df_unrelated = pd.DataFrame(data=unrelated)
    df_unrelated_w = pd.DataFrame(data=unrelated_w)

    if df_unrelated_w.shape[0] < df_related.shape[0]:
        df_down = df_unrelated.sample(df_related.shape[0] - df_unrelated_w.shape[0])
        df_unrelated_w = pd.concat([df_down, df_unrelated_w])

    df_balanced = pd.concat([df_unrelated_w, df_related])

    X_train, X_test, y_train, y_test = train_test_split(df_balanced['Text'], df_balanced['Ed'], stratify=df_balanced['Ed'])
    
    edu_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/backups/emodel_' + edu_ver + '.h5')
    edu_model.fit(X_train, y_train, epochs=training_round)
    os.remove('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/emodel_' + edu_ver + '.h5')
    edu_ver = (str)((int)(edu_ver) + 1)
    edu_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/emodel_' + edu_ver + '.h5')
    
    output[5] = edu_ver
    with open('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/version.txt', "w") as f:
        for item in output:
            f.write(str(item) + '\n')

def train_int():
    global interest_ver, output

    # store related 
    for sen in Sentences_int.objects.all():
        new_int_data = Int_data(weight=0, label=1, text=sen.body)
        new_int_data.save()
    Sentences_int.objects.all().delete()

    #store unrelated
    for sen in Sentences_irr_int.objects.all():
        new_int_data = Int_data(weight=1, label=0, text=sen.body)
        new_int_data.save()
    Sentences_irr_int.objects.all().delete()

    related_l = []
    related_t = []
    unrelated_wl = []
    unrelated_wt = []
    unrelated_l = []
    unrelated_t = []
    for item in Int_data.objects.raw('SELECT * FROM "classifier_int_data"'):
        if item.label == 1:
            related_l.append(1)
            related_t.append(item.text)
        elif item.weight == 1:
            unrelated_wl.append(0)
            unrelated_wt.append(item.text)
        else:
            unrelated_l.append(0)
            unrelated_t.append(item.text)
    
    related = {'Interest': related_l, 'Text': related_t}
    unrelated_w = {'Interest': unrelated_wl, 'Text': unrelated_wt}
    unrelated = {'Interest': unrelated_l, 'Text': unrelated_t}
    df_related = pd.DataFrame(data=related)
    df_unrelated = pd.DataFrame(data=unrelated)
    df_unrelated_w = pd.DataFrame(data=unrelated_w)

    if df_unrelated_w.shape[0] < df_related.shape[0]:
        df_down = df_unrelated.sample(df_related.shape[0] - df_unrelated_w.shape[0])
        df_unrelated_w = pd.concat([df_down, df_unrelated_w])

    df_balanced = pd.concat([df_unrelated_w, df_related])

    X_train, X_test, y_train, y_test = train_test_split(df_balanced['Text'], df_balanced['Interest'], stratify=df_balanced['Interest'])
    
    interest_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/backups/imodel_' + interest_ver + '.h5')
    interest_model.fit(X_train, y_train, epochs=training_round)
    os.remove('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/imodel_' + interest_ver + '.h5')
    interest_ver = (str)((int)(interest_ver) + 1)
    interest_model.save('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/imodel_' + interest_ver + '.h5')
    
    output[8] = interest_ver
    with open('C:/Users/taige/Desktop/Research/summer2022/week8/django/mysite/classifier/resource_data/version.txt', "w") as f:
        for item in output:
            f.write(str(item) + '\n')

def get_csv():
    related_l = []
    related_t = []
    unrelated_wl = []
    unrelated_wt = []
    unrelated_l = []
    unrelated_t = []
    for item in Awd_data.objects.raw('SELECT * FROM "classifier_awd_data"'):
        if item.label == 1:
            related_l.append(1)
            related_t.append(item.text)
        elif item.weight == 1:
            unrelated_wl.append(0)
            unrelated_wt.append(item.text)
        else:
            unrelated_l.append(0)
            unrelated_t.append(item.text)
    
    related = {'Interest': related_l, 'Text': related_t}
    unrelated_w = {'Interest': unrelated_wl, 'Text': unrelated_wt}
    unrelated = {'Interest': unrelated_l, 'Text': unrelated_t}
    df_related = pd.DataFrame(data=related)
    df_unrelated = pd.DataFrame(data=unrelated)
    df_unrelated_w = pd.DataFrame(data=unrelated_w)


    df= pd.concat([df_unrelated_w, df_related, df_unrelated])
    df.to_csv('awd_data.csv')

    return 


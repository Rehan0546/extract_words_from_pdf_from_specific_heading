
folder = ''
csv_file_path = 'Tito_1_Confidential.csv'

import PyPDF2
import nltk
import pandas as pd
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import wordnet, stopwords
import numpy as np
import re
import heapq
import glob
import os
import json


def all_words_Sents_extraction(sentences,words):
  all_words_Sents = []
  for word in words:
    word_sent_list = []
    for sent in sentences:
      if word.lower() in sent.lower():
        index = sent.lower().index(word.lower())
        before = sent[:index].replace('\n',' ')
        try:
          after = sent[index+len(word):].replace('\n',' ')
        except:
          after = ''
        all_words_Sents.append([str(before),str(word),str(after)])
    
    # if len(word_sent_list):
    #   all_words_Sents.append(word_sent_list)
  return all_words_Sents

def pdf_getting_text(pdf_file):
  # reading pdf
  pdfReader = PyPDF2.PdfFileReader(pdf_file)
  text = ''
  # iterate over pdf to collect text
  for i in range(pdfReader.numPages):
    pageObj = pdfReader.getPage(i)
    text = text + ' '+ pageObj.extractText()
  return text

def BOW(all_words, lower_text): #Bag of Words 
  # finding and counting words in given pdf and saving into dictionary 
  words_counts = dict()
  for word in all_words:
    if word.lower() in lower_text:
      words_counts[word] = lower_text.count(word.lower())
  # words_counts = dict(sorted(words_counts.items(), key=lambda item: item[1], reverse = True))
  return words_counts

def find_synonyms(word):
  synonyms = []
  for syn in wordnet.synsets(word):
      for i in syn.lemmas():
          synonyms.append(i.name())
  return synonyms

def descriptive_statistics(article_text,
                           k = 7,
                           stopwords = nltk.corpus.stopwords.words('english')):
  # Removing Square Brackets and Extra Spaces
  article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
  article_text = re.sub(r'\s+', ' ', article_text)
  # Removing special characters and digits
  formatted_article_text  = re.sub('[^a-zA-Z]', ' ', article_text )
  formatted_article_text  = re.sub(r'\s+', ' ', formatted_article_text )

  sent_article_text = nltk.sent_tokenize(article_text)

  word_frequencies = {}
  for word in nltk.word_tokenize(formatted_article_text ):
      if word not in stopwords:
          if word not in word_frequencies.keys():
              word_frequencies[word] = 1
          else:
              word_frequencies[word] += 1
  
  sentence_scores = {}
  for sent in sent_article_text:
      for word in nltk.word_tokenize(sent.lower()):
          if word in word_frequencies.keys():
              if len(sent.split(' ')) < 30:
                  if sent not in sentence_scores.keys():
                      sentence_scores[sent] = word_frequencies[word]
                  else:
                      sentence_scores[sent] += word_frequencies[word]
  summary_sentences = heapq.nlargest(k, sentence_scores, key=sentence_scores.get)

  summary = ' '.join(summary_sentences)
  return summary

def information_get(pdf_file,csv_file,
        col = 'Search Terms', 
        STOPWORDS = set(stopwords.words('english'))):
  text =  pdf_getting_text(pdf_file)
  lower_text = text.lower()
  # some Preprocessing and removing stopwords on csv words column
  words = csv_file[col]
  words = " ".join(words).lower()
  words = " ".join([word for word in str(words).split() if word not in STOPWORDS])
  words = words.split()

  # finding synonyms of given words
  all_words = []
  for w in words:
    synonyms = find_synonyms(w)
    all_words = all_words + synonyms

  # joining synonyms and orignal words
  all_words = list(csv_file['Search Terms']) + all_words 
  # Finding BAG of Words of all colected words.
  words_counts = BOW(list(set(all_words)), lower_text)

  # Finding all sentences has contains given words
  # split into [before, word, after]
  sentences = np.array(text.split('.'))
  words = list(set(np.array(csv_file['Search Terms'])))
  all_words_Sents = all_words_Sents_extraction(sentences,words)

  summary = descriptive_statistics(text.replace('\n',' '))

  return words_counts, all_words_Sents, summary

csv_file = pd.read_csv(csv_file_path)
files_paths = glob.glob(folder+'*.pdf')
for file_ in files_paths:
  pdf_file = open(file_, 'rb')

  file_name = file_[:-3]
  if not os.path.exists(file_name):
    os.makedirs(file_name)

  words_counts, all_words_Sents, summary = information_get(pdf_file,csv_file)

  with open(os.path.join(file_name,'words_counts.txt'), 'w') as file:
     file.write(json.dumps(words_counts)) 
  
  with open(os.path.join(file_name,'summary.txt'), 'w') as file:
     file.write(summary)
  
  pd.DataFrame(all_words_Sents, columns =['before', 'word', 'after']).to_csv(os.path.join(file_name,'all_words_Sents.csv'))


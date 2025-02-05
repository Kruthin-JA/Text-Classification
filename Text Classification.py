#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
dataset=pd.read_csv('hate_speech.csv')
dataset.head()


# In[36]:


dataset.shape


# In[37]:


dataset.label.value_counts()


# In[38]:


for index, tweet in enumerate(dataset["tweet"] [10:15]):
    print(index+1,"-",tweet)


# In[39]:


import re
# Clean text from noise
def clean_text(text):
    # Filter to allow only alphabets
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    # Remove Unicode characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Convert to lowercase to maintain consistency
    text = text.lower()
    return text


# In[40]:


dataset['clean_text'] = dataset.tweet.apply(lambda x: clean_text(x))


# In[41]:


dataset.head(10)


# In[42]:


import nltk
nltk.download('stopwords')


# In[43]:


stop = stopwords.words('english')


# In[44]:


# Generate word frequency
def gen_freq(text):
    # Will store the list of words
    word_list = []
    # Loop over all the tweets and extract words into word_list
    for tw_words in text.split():
        word_list.extend(tw_words)
    
    # Create word frequencies using word_list
    word_freq = pd.Series(word_list).value_counts()
    # Drop the stopwords during the frequency calculation
    word_freq = word_freq.drop(stop, errors='ignore')
    
    return word_freq


# In[45]:


def any_neg(words):
    for word in words:
        if word in ['n', 'no', 'non', 'not'] or re.search(r"\wn't", word):
            return 1
        else:
            return 0


# In[46]:


def any_rare(words, rare_100):
    for word in words:
        if word in rare_100:
            return 1
        else:
            return 0


# In[47]:


def is_question(words):
    for word in words:
        if word in ['when', 'what', 'how', 'why', 'who', 'where']:
            return 1
        else:
            return 0


# In[ ]:


word_freq = gen_freq(dataset.clean_text.str)
# 100 most rare words in the dataset
rare_100 = word_freq[-100] # last 100 rows/words

# Number of words in a tweet
dataset['word_count'] = dataset.clean_text.str.split().apply(lambda x: len(x))

# Negation present or not
dataset['any_neg'] = dataset.clean_text.str.split().apply(lambda x: any_neg(x))

# Prompt present or not
dataset['is_question'] = dataset.clean_text.str.split().apply(lambda x: is_question(x))

# Any of the most 100 rare words present or not
dataset['any_rare'] = dataset.clean_text.str.split().apply(lambda x: any_rare(x, rare_100))

# Character count of the tweet
dataset['char_count'] = dataset.clean_text.apply(lambda x: len(x))


# In[ ]:






# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division, print_function
#for word encoding
import codecs
#regex
import glob
#concurrency
import multiprocessing
#dealing with OS, reading a file
import os
#prettyprinting, human readable
import pprint
#regular expression
import re
#Natural language processing
import nltk
#word 2 vec
import gensim.models.word2vec as w2v
#dimensionality reduction
import sklearn.manifold
#math
import numpy as np
#plotting
import matplotlib.pyplot as plt
#parse pandas as pd
import pandas as pd
#visualization
import seaborn as sns

import gensim

# In[ ]:

# %pylab inline

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#step 1 process our data
# #clean data
# nltk.download('punkt') #pretrained tokenizer
# nltk.download('stopwords') #words like a, an, the


# In[ ]:


#get the book names, matching txt file
book_filenames = sorted(glob.glob('got1.txt'))
print('Found books:')
book_filenames


# In[ ]:


corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename,"r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()


# In[ ]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[ ]:


raw_sentences = tokenizer.tokenize(corpus_raw)


# In[ ]:


def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return words


# In[ ]:


#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[ ]:


print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))


# In[ ]:


token_count = sum(len(sentence) for sentence in sentences)
print(token_count)


# In[ ]:


#Train word to vector

# 3 main tasks that vectors help with# Distance, similarity and ranking

#Dimensionality of the resulting word vectors
num_features = 300

#min word count threshold
min_word_count = 3

#Number of threads to run in parallel
num_workers = multiprocessing.cpu_count()

#context window length
context_size = 7

#downsample setting for frequent words
downsampling = 1e-3

#seed for the RNG, to make the results reproducible
seed  = 1


# In[ ]:


thrones2vec = w2v.Word2Vec(sg = 1, seed = seed, workers = num_workers, 
                           size = num_features, min_count = min_word_count
                          ,window = context_size, sample = downsampling)


# In[ ]:


thrones2vec.build_vocab(sentences)


# In[ ]:


print(thrones2vec)


# In[ ]:


thrones2vec.train(sentences, epochs=thrones2vec.epochs,
                  total_examples = thrones2vec.corpus_count)


# In[ ]:


if not os.path.exists("trained"):
    os.makedirs("trained")


# In[ ]:


thrones2vec.save(os.path.join("trained", "thrones2vec.w2v"))


# In[ ]:


thrones2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v"))


# In[ ]:


#tsne compressing word

tsne = sklearn.manifold.TSNE(n_components = 2, random_state = 0)


# In[ ]:


all_word_vectors_matrix = thrones2vec.wv.vectors
print(all_word_vectors_matrix.shape)


# In[ ]:


#train t-SNE

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)


# In[ ]:


points = pd.DataFrame(
    [
        (word,coords[0], coords[1])
        for word, coords in [(word,all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index]) for word in thrones2vec.wv.vocab]
    ],
    columns = ['word', 'x','y']
)


# In[ ]:


print(points.head(10))


# In[ ]:


sns.set_context('poster')
data =points.plot.scatter('x','y',s=10, figsize=(10,20))
plt.show(data)

def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

data_2 = plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))

plt.show(data_2)


def nearest_similarity_cosmul(start1, end1, end2):
    similarities = thrones2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

print(nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun"))
print(nearest_similarity_cosmul("Jaime", "sword", "wine"))
print(nearest_similarity_cosmul("Arya", "Nymeria", "dragons"))






















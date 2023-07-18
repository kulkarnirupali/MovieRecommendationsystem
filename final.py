# This is movie recommendation system project using tools such as machine learning ,pandas, numpy,cosine similarity etc for any online platform.


import numpy as np
import pandas as pd

# Getting dataset from csv to dataframe using for further processing
movies = pd.read_csv("movies.csv.csv")

print(movies.head())

# to check null values in the dataset value
print(movies.isnull().sum())
print("Dropping all the null values to make dataset more clean")
print(movies.dropna(inplace=True))


# Use of iloc function to retrive data from any row or column i.e. specify the columns or row number in the square  bracket [].
print(movies.iloc[0].genre)

# use of abstract syntax tree to see tree of python code represented in the dataframe


# forming New dataset with required numbers of columns useful for the movie recommendation process
movies=movies[['id','title','genre','original_language','overview']]
print(movies.head())

# seperate every word from the 'overview' columns so that recommendation process will be more easy.
movies['overview']=movies['overview'].apply(lambda x:x.split())
print(movies.head())

# Removing bracket for better use in merging in the column 'overview'
movies['overview']=movies['overview'].apply(lambda x:''.join(x))
print(movies.head())

# merging all the columns into one column used in recommendation process.
movies['tags']=movies['overview']+movies['genre']+movies['original_language']
print(movies.head())

# Lowercase conversion of the column 'tags'
movies['tags']=movies['tags'].apply(lambda x:x.lower())
print(movies.head())

# Use of Countervectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()
print(vector)

# Use of natural language test kit i.e. NLKT
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies['tags']=movies['tags'].apply(stem)

from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(vector))
similarity = cosine_similarity(vector)
print(similarity[0])

print(sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6])

# final function which perform recommendation of movie

def recommendation(movie):
    movie_index = movies[movies['title']==movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(movies.iloc[i[0]].title)

print("Fallowing are the movies that system want to recommend you to watch further :")
print(recommendation('Devdas'))
print("Succefully completed.....")
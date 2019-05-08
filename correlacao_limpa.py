#O que mais você consegue explorar nesta base? Apresente pelo menos mais uma relação que conseguir encontrar utilizando uma técnica de visualização.

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


#atributos = ['color', 'director_name', 'num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',  'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name',  'movie_title', 'num_voted_users', 'cast_total_facebook_likes', 'actor_3_name', 'facenumber_in_poster', 'plot_keywords', 'movie_imdb_link', 'num_user_for_reviews', 'language', 'country', 'content_rating', 'budget','title_year', 'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio', 'movie_facebook_likes']

movie = pd.read_csv('movie_metadata.csv', delimiter=',')

movie=movie[['color', 'num_critic_for_reviews', 'duration',  'director_facebook_likes', 'actor_3_facebook_likes', 'actor_1_facebook_likes',  'gross', 'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster', 'num_user_for_reviews', 'language', 'country', 'content_rating','budget','title_year', 'actor_2_facebook_likes','imdb_score', 'aspect_ratio', 'movie_facebook_likes']]

chaves=movie.columns
for i in chaves:
	if is_string_dtype(movie[i]):
		movie[i].fillna('0', inplace=True)

	if is_numeric_dtype(movie[i]):
		movie[i].fillna(movie[i].mean(), inplace=True)	

print(movie)

dic={}
for i in chaves:
	if is_string_dtype(movie[i]):
		dic[i]=preprocessing.LabelEncoder()
		movie[i]=dic[i].fit_transform(movie[i])

print (movie)


corr = movie.corr()


fig, ax = plt.subplots()

plt.imshow(corr, cmap='hot', interpolation='none')  
plt.colorbar()  
plt.xticks(range(len(corr)), corr.columns, rotation = 'vertical')  
plt.yticks(range(len(corr)), corr.columns)
plt.subplots_adjust(left=0.10, bottom=0.20, right=0.80, top=0.99, wspace=0.20, hspace=0.20)

plt.show()

#print (movie.groupby('plot_keywords')['plot_keywords'].count())
#print (movie.isnull().sum())





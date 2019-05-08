#O que mais você consegue explorar nesta base? Apresente pelo menos mais uma relação que conseguir encontrar utilizando uma técnica de visualização.

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn import linear_model
from pandas.plotting import scatter_matrix

#atributos = ['color', 'director_name', 'num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',  'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name',  'movie_title', 'num_voted_users', 'cast_total_facebook_likes', 'actor_3_name', 'facenumber_in_poster', 'plot_keywords', 'movie_imdb_link', 'num_user_for_reviews', 'language', 'country', 'content_rating', 'budget','title_year', 'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio', 'movie_facebook_likes']



movie = pd.read_csv('movie_metadata.csv', delimiter=',')

#movie.drop(['color', 'director_name', 'actor_2_name', 'genres', 'actor_1_name', 'movie_title', 'actor_3_name', 'face_number_in_poster', 'plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating', 'budget', 'title_year', 'aspect_ratio'], axis=0, inplace = True)



movie=movie[['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross', 'num_voted_users', 'cast_total_facebook_likes', 'num_user_for_reviews', 'actor_2_facebook_likes', 'imdb_score', 'movie_facebook_likes']]

#atributos = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross', 'num_voted_users', 'cast_total_facebook_likes', 'num_user_for_reviews', 'actor_2_facebook_likes', 'imdb_score', 'movie_facebook_likes']

atributos = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']


chaves=movie.columns
for i in chaves:
	if is_string_dtype(movie[i]):
		movie[i].fillna('0', inplace=True)

	if is_numeric_dtype(movie[i]):
		movie[i].fillna(movie[i].mean(), inplace=True)	

#print(movie)

dic={}
for i in chaves:
	if is_string_dtype(movie[i]):
		dic[i]=preprocessing.LabelEncoder()
		movie[i]=dic[i].fit_transform(movie[i])

#print (movie)

#movie.sort_values(by=['gross'])
#fig,ax = plt.subplots()

x = movie.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
movie = pd.DataFrame(x_scaled, columns=atributos)
sm=scatter_matrix(movie, diagonal = 'kde')

[s.xaxis.label.set_rotation(0) for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]

#May need to offset label when rotating to prevent overlap of figure
[s.get_yaxis().set_label_coords(-0.2,0.2) for s in sm.reshape(-1)]

#Hide all ticks
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]
plt.show()










exit(0)



x = movie.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
movie = pd.DataFrame(x_scaled, columns=atributos)
x=movie['cast_total_facebook_likes']
x=[x]
print(x)
#x.reshape(x,(-1, 1))
y=movie['actor_1_facebook_likes']
y=[y]
modelo=linear_model.LinearRegression()
modelo.fit(x,y)
plt.scatter(x,y)
plt.plot(x, modelo.predict(x), linewidth=3, color='blue')
print(modelo.predict(x))
#movie.plot.scatter(x='cast_total_facebook_likes', y='actor_1_facebook_likes')
plt.show()


exit(0)

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





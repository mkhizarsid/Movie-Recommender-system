# %%
import pandas as pd
import numpy as np
credit= pd.read_csv('Data_set/tmdb_5000_credits.csv')
movie= pd.read_csv('Data_set/tmdb_5000_movies.csv')
credit.head(1)
movie.head(1)
# %%
movies=movie.merge(credit,on='title')
movies.head(1)
# %%
#id,title,overview,genres,keywords,cast,crew
movies=movies[['id','title','overview','genres','keywords','cast','crew']]
movies.head(1)
# %%
movies.isnull().sum()
# %%
movies.dropna(inplace=True)
# %%
movies.duplicated().sum()
# %%
movies.iloc[0].cast
# %%
import ast
ast.literal_eval
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies['genres']=movies['genres'].apply(convert)
movies.head(1)
# %%
movies.iloc[0].keywords
# %%
movies['keywords']=movies['keywords'].apply(convert)
# %%
movies.head(1)
movies.iloc[0].cast
# %%
def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L
movies['cast']=movies['cast'].apply(convert3)
# %%
movies.head()
# %%
movies.iloc[0].crew

# %%
def fetch_director(obj):
    L=[]    
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break  
    return L
movies['crew']=movies['crew'].apply(fetch_director)
# %%
movies.head(1)
# %%
movies['overview']=movies['overview'].apply(lambda x:x.split())
# %%
movies.head(1)
# %%
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i   in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
# %%
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
movies.head(1)
# %%
new_df=movies[['id','title','tags']]
new_df.head(1)
# %%
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
# %%
new_df.head(1)
# %%
new_df['tags']= new_df['tags'].apply(lambda x:x.lower())
new_df.head(1)
# %%
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')
# %%
vectors=cv.fit_transform(new_df['tags']).toarray()
# %%
print(vectors)
# %%
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

# %%
def stem (text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
# %%
new_df['tags']=new_df['tags'].apply(stem)
# %%
new_df.head(1)
# %%
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
# %%
def recommended(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)  
# %%
#recommended('The Dark Knight Rises')

# %%
# %%
import pickle

# Dataframe ko dictionary format mein save ker rahe hain
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))

# Similarity matrix ko save ker rahe hain
pickle.dump(similarity, open('similarity.pkl', 'wb'))
# %%

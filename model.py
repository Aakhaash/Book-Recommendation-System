import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv('./Datasets/Books.csv',low_memory=False)
ratings = pd.read_csv('./Datasets/Ratings.csv')
users = pd.read_csv('./Datasets/Users.csv')

# print(books.shape)
# print(ratings.shape)
# print(users.shape)

#Popularity Based Recommender System

ratings_with_name = ratings.merge(books,on='ISBN')

num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating' : 'num-ratings'},inplace=True)

avg_rating_df = ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating' : 'avg-rating'},inplace=True)

popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df = popular_df[popular_df['num-ratings']>=250].sort_values('avg-rating',ascending=False).head(50)

popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num-ratings','avg-rating']]

#Collaborative Filtering Based Recommender System

x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
considered_users = x[x].index

filtered_ratings = ratings_with_name[ratings_with_name['User-ID'].isin(considered_users)]

y = filtered_ratings.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index

final_ratings = filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt.fillna(0,inplace=True)

similarity_score = cosine_similarity(pt)

#Function to recommend books
def recommend(book_name):
    #Fetching index
    index = np.where(pt.index==book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x:x[1], reverse=True)[1:6]

    recommended_books = []
    for i in similar_books:
        recommended_books.append(pt.index[i[0]])
    
    return recommended_books
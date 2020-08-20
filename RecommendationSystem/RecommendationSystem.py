#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import operator
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()


# Import and read the train_file

# In[ ]:


r_cols_user = ['User_id','Movie_id','Rating']
user_rating_data = pd.read_csv('Filepath', skiprows = 1, sep=' ', header=None, names=r_cols_user)


# In[ ]:


user_rating_data


# Convert the dataframe into a utility matrix using pivot function

# In[ ]:


utility_matrix = user_rating_data.pivot(index='User_id', columns='Movie_id', values='Rating')


# In[ ]:


utility_matrix


# Replace all the NaN values with 0

# In[ ]:


utility_matrix = utility_matrix.fillna(0)


# Calculate the cosine similarity between users

# In[ ]:


df = cosine_similarity(utility_matrix, utility_matrix)


# Copy into dataframe with rows and columns as the user ids and cell values as the similarity measure

# In[ ]:


user_user_similarity = pd.DataFrame(data=df, index = utility_matrix.index, columns = utility_matrix.index,copy=True)


# In[ ]:


user_user_similarity


# Read the test data consisting of user-movie pairs

# In[ ]:


r_cols_test = ['User_id','Movie_id']
user_test_data = pd.read_csv('test.DAT', skiprows = 1, sep=' ', header=None, names=r_cols_test)


# In[ ]:


user_test_data


# Predict ratings using by implementing the collaborative filtering method based on user-user similarity

# In[ ]:


def get_ratings(user_test_data):
    pred = []
    for index, row in user_test_data.iterrows():
        u_id = row['User_id']
        m_id = row['Movie_id']
        dfdict = user_user_similarity.loc[:,u_id].to_dict()
        sorted_d = dict(sorted(dfdict.items(), key=operator.itemgetter(1),reverse=True))
        out = dict(itertools.islice(sorted_d.items(), 200))
        topusers=list(out.keys())
        dfsubset=user_rating_data[user_rating_data['User_id'].isin(topusers)]
        dfsubset=dfsubset[dfsubset['Movie_id']==m_id]
        #Calculate Weighted average
        '''rating = 0
        test = sum([out[x] for x in list(dfsubset['User_id'])])
        rating = 0
        for index, row in dfsubset.iterrows():
            rating = rating + (out.get(int(row['User_id'])) * row['Rating'])
        if(test!=0):
            predrating = rating / test
            pred.append(predrating)
        else:
            pred.append(np.nan)'''
        #Calculate Mean 
        pred.append(dfsubset['Rating'].mean())
    return pred


# Call get_ratings() which return the predicted ratings for the user-movie pairs

# In[ ]:


pred = get_ratings(user_test_data)
pred_bkp = pred
len(pred)


# In[ ]:


pred = pred_bkp


# Get a dataframe containing records with ratigns predicted as Nan

# In[ ]:


def get_Nan(pred, user_test_data):    
    pred_Nan = pd.DataFrame(data=pred, index = user_test_data.index)
    is_NaN = pred_Nan.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = pred_Nan[row_has_NaN]
    test_nan = user_test_data.loc[rows_with_NaN.index.values]
    return test_nan


# Call the get_Nan()

# In[ ]:


test_nan = get_Nan(pred, user_test_data)


# In[ ]:


test_nan


# Import and read any of the movie feature files such as actor/director/genre. Modify the column names accordingly.

# In[ ]:


r_cols_generic = ['Movie_id','Actor_id','Actor_name','Ranking']
generic_additional_data = pd.read_csv('movie_actors.DAT', encoding='cp1252', skiprows = 1, sep='\t', header=None, names=r_cols_generic)


# Predict the ratigns for records using the content-based recommendation method

# In[ ]:


def get_preference_ratings():
    for index, row in test_nan.iterrows():
        pred_index = index
        print(pred_index)
        test_df=test_nan.loc[index,:]
        user_watch_movies = user_rating_data[user_rating_data.User_id==row['User_id']]
        user_movie_dict = {}
        for index, row in user_watch_movies.iterrows():
            additional_data_list = list(generic_additional_data.loc[generic_additional_data['Movie_id'] == row['Movie_id'], 'Actor_id'])
            user_movie_dict[int(row['Movie_id'])] = additional_data_list
        user_movie_df = pd.DataFrame.from_dict(user_movie_dict, orient='index')
        user_movie_df['Merged'] = user_movie_df[user_movie_df.columns[:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
        user_movie_df = user_movie_df[['Merged']]
        X = vectorizer.fit_transform(user_movie_df['Merged'])
        user_movie_dict_predict = {}
        additional_data_list = list(generic_additional_data.loc[generic_additional_data['Movie_id'] == row['Movie_id'], 'Actor_id'])
        user_movie_dict_predict[int(row['Movie_id'])] = additional_data_list
        user_movie_predict_df = pd.DataFrame.from_dict(user_movie_dict_predict, orient='index')
        user_movie_predict_df['Merged'] = user_movie_predict_df[user_movie_predict_df.columns[:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
        user_movie_predict_df = user_movie_predict_df[['Merged']]
        X_test = vectorizer.transform(user_movie_predict_df['Merged'])
        df_movie_similarity = cosine_similarity(X, X_test)
        df_movie_similarity = pd.DataFrame(data=df_movie_similarity, index = user_movie_df.index,copy=True)
        dfdict_Nan=df_movie_similarity.loc[:,0].to_dict()
        sorted_d_Nan = dict(sorted(dfdict_Nan.items(), key=operator.itemgetter(1),reverse=True))
        out_Nan = dict(itertools.islice(sorted_d_Nan.items(), 5))
        topusers_Nan=list(out_Nan.keys())
        dfsubset_Nan=user_rating_data[user_rating_data['Movie_id'].isin(topusers_Nan)]
        p_Nan=dfsubset_Nan['Rating'].mean()
        pred[pred_index] = p_Nan


# Call get_preference_ratings() function whcih replaces the Nan ratings with the predicted ratings in the prediction array

# In[ ]:


get_preference_ratings()


# Write the predictions to a file

# In[ ]:


with open('Assignment_final_predictions_mean_200.txt', 'w') as filehandle:
    for listitem in pred:
        filehandle.write('%s\n' % listitem)
    filehandle.close()


#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate The Movie Database (TMDB)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > This movie database contains information about approximately 10,000 movies including genres, ratings, revenue, budget, Cast, and more. It contains movies which are released between1960 and 2015, it also has two columns for budget and revenue in terms of 2010 dollars accounting for inflation over time which will be used in any comparisons in the analysis instead of unadjusted ones. 
# 
# 
# > In this Investigation two important questions will be answered
# 1. what are the factors related to the top 50 highest revenue movies ?
# 2. how did the movie industry evolve through years ? 

# In[1]:


## importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# ### General Properties

# In[2]:


#loading the data set 
movies_df = pd.read_csv('tmdb-movies.csv')
movies_df.head()


# **General info**

# In[3]:


movies_df.info()


# >Note: We have 21 columns in our data set but we will not use all of them in our analysis
# The Columns we will use are : popularity, original_title, Cast,runtime, genres, release_year, vote_average, budget_adj and revenue_adj 
# 

# **Null Values**

# In[4]:


movies_df.isnull().sum()


# >We only care about genres and Cast in our analysis so let's see the rows which have null values in these columns

# In[5]:


movies_df[movies_df.cast.isnull()]


# > There are 76 movies in this dataset that doesn't have any Cast and we can replace the Nan values with 'unknown' instead of deleting these entries

# In[6]:


movies_df[movies_df.genres.isnull()]


# > There are 23 movies in this dataset that doesn't have any genres and we can remove these entries from the dataset as we will make a lot of operations on genres in our analysis  

# **Duplicates**

# In[7]:


movies_df[movies_df.duplicated()]


# > Only one duplicated entry will be removed 

# **Descriptive statistics of the data set**

# In[8]:


movies_df.describe()


# > Most of the dataset entries have 0 at budget _adj and revenue_adj then we will not remove but will query the dataset in our analysis on some conditions on the revenue_adj and budget_adj in the second question we will answer 

# **Exploring important columns**

# In[9]:


movies_df.genres.value_counts()


# In[10]:


movies_df.cast.value_counts()


# >A lot of entries in genres and cast columns contain multiple values separated by pipe (|) characters and should be split

# 
# 
# ### Data Cleaning 

# In[11]:


movies_df.columns


# **drop the unused columns**

# In[12]:



unused = ['id', 'imdb_id','budget', 'revenue','homepage', 'director', 'tagline', 'keywords', 'overview',         'production_companies','vote_count', 'release_date']


# In[13]:


movies_df.drop(columns= unused, inplace =True)
movies_df.head()


# **Removing duplicates**

# In[14]:


movies_df.drop_duplicates(inplace =True)
movies_df.duplicated().sum()


# **Removing and Replacing Null Values**

# In[15]:


movies_df.dropna(inplace = True, subset=['genres']) #remove entries with Null values at genres
movies_df.genres.isnull().sum()


# In[16]:


movies_df.cast.fillna('Unknown',inplace = True) #replace entries with Null values at cast with unknown
movies_df.cast.isnull().sum()


# **Making a dataframe for cast and genres for the Exploratory data analysis phase**

# In[17]:


movies_df.reset_index(inplace= True,drop =True)

genres_data = movies_df.genres.str.split('|').to_list() # splitting the genres and making a list for each entry

# making a dataframe with genre and important columns
genres_df =pd.DataFrame(data = genres_data,index=[movies_df.release_year,movies_df.original_title,movies_df.revenue_adj]).stack() 
genres_df=genres_df.reset_index([0,1,2,3])
genres_df.drop(columns= 'level_3',inplace = True)
genres_df.rename({0:'genres'},axis = 1,inplace = True)
genres_df.head()


# In[18]:


movies_df.reset_index(inplace= True,drop =True)

cast_data = movies_df.cast.str.split('|').to_list() # splitting the cast and making a list for each entry

# making a dataframe with cast and important columns
cast_df =pd.DataFrame(data = cast_data,index=[movies_df.release_year,movies_df.original_title,movies_df.revenue_adj]).stack() 
cast_df=cast_df.reset_index([0,1,2,3])
cast_df.drop(columns= 'level_3',inplace = True)
cast_df.rename({0:'cast'},axis = 1,inplace = True)
cast_df.head()


# In[19]:


movies_df.shape


# In[20]:


cast_df.shape


# In[21]:


genres_df.shape


# > - The new data set has 10842 rows and 10 columns 
# - The cast data set has 52549 rows and 4 columns 
# - The genres data set has 26955 rows and 4 columns

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### what are the factors related to the top 100 highest revenue movies ?

# We will explore the top 100 highest revenue movies as they are the most successful movies.
# we will create the visualization to find insights about the factors relared to these movies and relationships between variables and revenue.

# **Add Decade Column to the data frame**

# In[22]:


movies_df['decade']= np.floor(movies_df.release_year / 10) * 10 
movies_df['decade'] = movies_df['decade'].astype('int64') ;


# In[23]:


top_100_df = movies_df.sort_values(by = "revenue_adj",ascending = False)[0:99]
top_100_df.head()


# In[24]:


plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.scatter(top_100_df.popularity,top_100_df.revenue_adj)
plt.xlabel("Popularity")
plt.ylabel("Revenue (2010) $");


# > As shown in the above figure, The popularity and the revenue have a linear relation between them as the revenue increases when the popularity increases

# In[25]:


plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.hist(top_100_df.budget_adj, alpha=0.5, label='budget_adj')
plt.hist(top_100_df.revenue_adj, alpha=0.5, label='revenue_adj');
plt.legend(loc='upper right')
plt.xlabel("Revenue (2010) $") ;


# >As shown in the figure above, All of the top 100 highest revenue movies have made high profit as the highest budget movie have a budget lower than the lowest revenue movie in these movies.

# In[26]:


plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.hist(top_100_df.runtime,alpha=0.5);
plt.xlabel("Run Time (Minutes)");


# >As shown in the figure above, Most of the top 100 movies have a run time from 120 minutes and 150 minutes.

# In[27]:


plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.plot((top_100_df.groupby(['decade'])['original_title'].count()),'-o');
plt.ylabel("Movies");


# >As shown in the figure above, The number of movies have increased through decades except for the 1980s and there were 25 movies in the years from 2010 to 2015 and it is expected that 2010s to have more of the highest revenue movie than 2000s.

# In[28]:


top_100_df = pd.Series(genres_df['revenue_adj'].unique()).sort_values(ascending = False)[0:99]
top_100_genres = genres_df [genres_df['revenue_adj'].isin(top_100_df)] 


# In[29]:


plt.figure(num=None, figsize=(8, 6), dpi=80)
genre_count = top_100_genres.groupby(["genres"])["original_title"].count().sort_values(ascending = False)
plt.bar(genre_count.index,genre_count.values)
plt.xticks(rotation = 90);


# >As showm in the figure above,
# 1. The most common genre in the top 100 movies is Adventure as over 70% of these movies have adventures. 
# 2. The second most common genre is Action as over 50% of these movies have action scenes 

# In[30]:


plt.figure(num=None, figsize=(8, 6), dpi=80)
top_100_rev = pd.Series(cast_df['revenue_adj'].unique()).sort_values(ascending = False)[0:99]
top_100_cast = cast_df [cast_df['revenue_adj'].isin(top_100_rev)] 
cast_count = top_100_cast.groupby(["cast"])["original_title"].count().sort_values(ascending = False)[0:15]
plt.bar(cast_count.index,cast_count.values)
plt.xticks(rotation = 90);


# In[31]:


top_100_cast[(top_100_cast["cast"] == "Emma Watson")]


# In[32]:


top_100_cast[(top_100_cast["cast"] == "Harrison Ford")]


# In[33]:


top_100_cast[(top_100_cast["cast"] == "Ian McKellen")]


# >From The tables above, It is recognised that the successful movie series like Harry Potter, Star Wars and The Lord of the Rings dominated the top 100 highest revenue movies

# ### how did the movie industry evolve through years ? 

# We will explore how the movie industry evolved through years in terms of a lot of factors like the number of movies, the most common genres, the run time and the popularity. 

# In[34]:


plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.plot(movies_df.groupby(['decade' ])["original_title"].count(),'-o')
plt.ylabel("Movies");


# >As shown in the figure above, The number of movies has increased through decades and it's expected to increase more in the 2010s as the movies made from 2010 to 2015 are more than the movies made in the 2010s

# In[35]:


plt.figure(num=None, figsize=(8, 6), dpi=80)
genres_df['decade']= np.floor(genres_df.release_year / 10) * 10 #making the decade column for genres data frame
genres_df['decade'] = genres_df['decade'].astype('int64') ;
x = genres_df.groupby(['genres' ])["original_title"].count().sort_values(ascending = False)
plt.bar(x.index,x.values)
plt.xticks(rotation = 90);


# > As shown in the figure above, The top 5 genres in our data set are Drama, Comedy, Thriller, Action and Romance  

# In[36]:


# plotting a line chart for the top 5 genres through years 
plt.figure(num=None, figsize=(8, 6), dpi=80)
for item in x.index[0:5] :
    item_df = genres_df[genres_df["genres"] == item]
    plt.plot(item_df.groupby(['decade' ])["original_title"].count(),label = item)
    plt.legend()


# >As shown in the figure above, 
# 1. The Drama genre was always the most common genre through decades except 1980's 
# 2. The Comedy genre had increased rapidly from 1970's to 1980's as it was the most common genre  

# In[37]:


plt.figure(num=None, figsize=(8, 6), dpi=80)
pop_year= movies_df.groupby(['release_year'])["popularity"].mean()
plt.plot(pop_year);
plt.ylabel("Popularity");


# >As shown in the figure above, The average popularity have been increased through years and it has been increased rapidly from 2010 to 2015

# In[38]:


plt.figure(num=None, figsize=(8, 6), dpi=80)
runtime_year= movies_df.groupby(['release_year'])["runtime"].mean()
plt.plot(runtime_year);
plt.ylabel("Run Time (Minutes)");


# >As shown in the figure above, The average run time have been decreased through years from 125 minutes in the 1960s to 95 minutes in 2015

# <a id='conclusions'></a>
# ## Conclusions
# 
# > The Factors related to the top 100 highest revenue movies are : 
# 1. The revenue increases when the popularity increases
# 2. All of these movies have made very high profit
# 3. Most of the top 100 movies have a run time from 120 minutes and 150 minutes.
# 4. Over 70% of these movies have adventures and 50% of these movies have action scenes. 
# 5. The successful movie series like Harry Potter, Star Wars and The Lord of the Rings dominated the list of the movies
# 
# > The movie industry evolved a lot through years as :
# 1. The number of movies has increased through decades and it's expected to increase more in the 2010s as the movies made from 2010 to 2015 are more than the movies made in the 2010s
# 2. The Drama genre was always the most common genre through decades except 1980's and The Comedy genre had increased rapidly from 1970's to 1980's as it was the most common genre
# 3. The average popularity have been increased through years and it has been increased rapidly from 2010 to 2015
# 4. The average run time have been decreased through years from 125 minutes in the 1960s to 95 minutes in 2015

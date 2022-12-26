#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[98]:


df = pd.read_csv('data.csv',header= 0,
                        encoding= 'unicode_escape')
print(df.head())
df.shape


# In[99]:



df.isnull().sum()  


# In[100]:


df=df.dropna(subset=['Description'])
  # dropping null values in Description column
df.isnull().sum()
     


# In[101]:


df.info()
df.shape


# In[102]:


df['Description']


# In[103]:


df['Description'] = df['Description'].str.replace("[^a-zA-Z0-9]", " ")


# In[104]:


df['Description']


# In[105]:


df['Description_processed'] = df['Description'].apply(lambda row: ' '.join([word for word in row.split() if len(word)>2]))


# In[106]:


df['Description_processed'] 


# In[107]:


df['Description_processed'] = [review.lower() for review in df['Description_processed']]


# In[108]:


df['Description_processed']


# In[109]:


# Removing Stopwords Begin
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize

stop_words = stopwords.words('english')


# In[110]:



# Making custom list of words to be removed 
add_words = ['felt','spot','colour']
# Adding to the list of words
stop_words.extend(add_words)

# Function to remove stop words 
def remove_stopwords(rev):
    # iNPUT : IT WILL TAKE ROW/REVIEW AS AN INPUT
    # take the paragraph, break into words, check if the word is a stop word, remove if stop word, combine the words into a para again
    review_tokenized = word_tokenize(rev)
    rev_new = " ".join([i for i in review_tokenized  if i not in stop_words])
    return rev_new

# Removing stopwords
df['Description_processed'] = [remove_stopwords(r) for r in df['Description_processed']]


# In[111]:


# Begin Lemmatization 
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# function to convert nltk tag to wordnet tag
lemmatizer = WordNetLemmatizer()

# Finds the part of speech tag
# Convert the detailed POS tag into a shallow information
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

# lemmatize sentence using pos tag
def lemmatize_sentence(sentence):
  # word tokenize -> pos tag (detailed) -> wordnet tag (shallow pos) -> lemmatizer -> root word
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


df['Description_processed'] =df['Description_processed'].apply(lambda x: lemmatize_sentence(x))


# In[112]:


df['Description_processed']


# In[113]:


df2 = df['Description_processed'].drop_duplicates()
df2 = pd.DataFrame(df2)
df2
     


# In[114]:


##
## Plotting most frequent words from positive reviews using bar chart
##
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white')
from nltk import FreqDist #function to find the frequent words in the data

# Subset positive review dataset
#all_words_df = df.loc[df['sentiment'] == 'negative',:]

#Extracts words into list and count frequency
all_words = ' '.join([text for text in df2['Description_processed']])
all_words = all_words.split()
words_df = FreqDist(all_words)

# Extracting words and frequency from words_df object
words_df = pd.DataFrame({'word':list(words_df.keys()), 'count':list(words_df.values())})
words_df
# Subsets top 30 words by frequency
words_df = words_df.nlargest(columns="count", n = 100) 

words_df.sort_values('count', inplace = True)

# Plotting 30 frequent words
plt.figure(figsize=(10,20))
ax = plt.barh(words_df['word'], width = words_df['count'])
plt.show()


# In[115]:


get_ipython().system('pip3 install wordcloud')


# In[118]:


import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[119]:


from wordcloud import WordCloud

all_words = ' '.join([text for text in df2['Description_processed']])
 
wordcloud = WordCloud(width = 1800, height = 800, 
                      background_color ='white', 
                      min_font_size = 10).generate(all_words)

#plot the WordCloud image                        
plt.figure(figsize = (13,5), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()
     


# # VECTORIZATION - Bag of Words Model
# Binary approach:
# Since the "product description" contains "unique words", I use Binary approach.

# In[121]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Step 1: Design the Vocabulary
count_vectorizer = CountVectorizer(binary = True) 

# Step 2: Create the Bag-of-Words Model
bag_of_words = count_vectorizer.fit_transform(df2['Description_processed']) # fit - design the vocbulary and transform will convert the text into numbers based on the presence of the word

# Show the Bag-of-Words Model as a pandas DataFrame
feature_names = count_vectorizer.get_feature_names()
df_binary = pd.DataFrame(bag_of_words.toarray(), columns = feature_names)


# In[122]:


df_binary


# In[123]:


# convert into numpy array:
x = bag_of_words.toarray()
x


# # Modelling

# In[124]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context('notebook')
plt.style.use('fivethirtyeight')

from warnings import filterwarnings
filterwarnings('ignore')


# # Elbow plot (K-MEANS):
# We make a plot between k value and inertia

# In[125]:


from sklearn.cluster import KMeans

list_k = list(range(1, 20))
inertias = []
for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(x)
    inertias.append(km.inertia_)

# Plotting
plt.figure(figsize=(4, 4))
plt.plot(list_k, inertias, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance'); 


# # INTERPRETATION of the above Elbow plot:
# 
# Since, we didnot get the optimum k-value, we need to reduce the number of feature using Dimensionality Reduction.
# Dimensionality reduction:
# Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by *transforming a large set of variables into a smaller one that still contains most of the information* in the large set.
# HOW DO YOU DO A PCA?
# Standardize the range of continuous initial variables
# Compute the covariance matrix to identify correlations
# Compute the eigenvectors and eigenvalues of the covariance matrix to identify the principal components
# Create a feature vector to decide which principal components to keep
# Recast the data along the principal components axes

# In[126]:


#PCA
from sklearn.decomposition import PCA


# finding the optimum number of components:
components = None
pca = PCA(n_components = components)  # components - hyperparameter
pca.fit(x)
     


# In[127]:


# printing the explained variances
print("Variances (Percentage):")
print(pca.explained_variance_ratio_ * 100)
     


# In[128]:


# printing the cumulative variances
print("Cumulative Variances (Percentage):")
print((pca.explained_variance_ratio_.cumsum() * 100))


# In[129]:


# plot a scree plot
components = len(pca.explained_variance_ratio_)     if components is None else components
plt.plot(range(1,components+1), 
         np.cumsum(pca.explained_variance_ratio_ * 100))

plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")


# In[130]:


# choosing only 85% of variations:
from sklearn.decomposition import PCA

pca = PCA(n_components = 0.85)
pca.fit(x)

# optimum no:of components
components = len(pca.explained_variance_ratio_)
print(f'Number of components: {components}')

# Make the scree plot
plt.plot(range(1, components + 1), np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")
     


# # *INTERPRETATION* of the above graph:
# 
# We can see that it takes 1250 components to reach around 100% variance.
# But I consider 85% variance is sufficient for modelling

# In[131]:


#APPLY PCA
from sklearn.decomposition import PCA

pca = PCA(450)
PCA_data = pca.fit_transform(x)
PCA_data.shape 


# In[132]:


get_ipython().system('pip install yellowbrick')


# In[133]:


# Elbow plot
from yellowbrick.cluster import KElbowVisualizer
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(1,12)).fit(PCA_data)
visualizer.show()


# In[134]:


#Interpretation* of the Elbow plot:

#optimal k-value occurs at 4
#ie. 4 clusters are required
km = KMeans(n_clusters=4,init="k-means++",random_state=42)    # applying k = 4
km.fit(PCA_data)           # fit the data - identify pattern, does everything

centroids = km.cluster_centers_  # final centroid points
print("centroids: \n",centroids)

print("\ninertia: ",km.inertia_)  # measures how tight my groups are. Lower the bett


# In[135]:


km.labels_   # shows which group each datapoint belongs to


# In[136]:


#predict the labels of clusters
label = km.fit_predict(PCA_data)  
print(label)
     


# # Visualizing the Product Clusters:
# 2D plotting:

# In[137]:


centroids = km.cluster_centers_   # Getting the Centroids
u_labels = np.unique(label)      # Getting the labels
 
# plotting the results:
plt.figure(figsize=(10, 6)) 
for i in u_labels:
    plt.scatter(PCA_data[label == i , 0] , PCA_data[label == i , 2] , label = i)
plt.scatter(centroids[:,0] , centroids[:,2] , marker="X", c="k", s=120, label="centroids")
plt.legend()
plt.show()
     


# In[138]:


labels = np.unique(label)  # Getting the labels

fig = plt.figure(figsize = (10,5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(PCA_data[label == 0,0],PCA_data[label == 0,1],PCA_data[label == 0,2], s = 20 , color = 'blue', label = "cluster 0")  
ax.scatter(PCA_data[label == 1,0],PCA_data[label == 1,1],PCA_data[label == 1,2], s = 20 , color = 'red', label = "cluster 1")  
ax.scatter(PCA_data[label == 2,0],PCA_data[label == 2,1],PCA_data[label == 2,2], s = 20 , color = 'yellow', label = "cluster 2")  
ax.scatter(PCA_data[label == 3,0],PCA_data[label == 3,1],PCA_data[label == 3,2], s = 20 , color = 'green', label = "cluster 3") 

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()
plt.show()
     


# In[139]:


# creating a dataframe for the cluster labels:
df5 = pd.DataFrame(km.labels_)
print(df5.shape)
df5


# In[140]:



# checking the shape of df2
print(df2.shape)
df2


# In[141]:


df2 = df2.reset_index(drop=True) # resetting the index to get uniform index values

df6 = df2.join(df5)  # joining the "product clusters" with the "Description_NLP"

df6.rename(columns={0:'Product Cluster'},inplace=True) # renaming the column
     


# In[142]:


# merging the "product clusters" with the original dataframe:
df8 = pd.merge(df, df6, how='left', on='Description_processed')
df8


# In[143]:


#Saving Scikitlearn models
import joblib
joblib.dump(label, "kmeans_model.pkl")


# In[144]:


df8.to_csv("Clustered_Customer_Data.csv")


# # Data Preprocessing:
# OneHot encoding the "Product clusters":

# In[145]:


df9 = pd.get_dummies(df8,columns=["Product Cluster"])
df9.head(10)
     


# In[146]:


# copying the dataframe into another variable:
df10 = df9.copy()


# In[147]:


# Dropping unnecessary features:
df10 = df10.drop(["InvoiceNo","StockCode","Description","InvoiceDate","Description_processed"],axis=1)
df10.head(3)
     


# # OneHot encoding "Country" feature:

# In[148]:


df11 = pd.get_dummies(df10,columns=["Country"])
df11.head(20)


# # Label encoding "customerID":

# In[149]:


from sklearn.preprocessing import LabelEncoder 
label_encoder = LabelEncoder() # label_encoder object knows how to understand word labels.
 
df11['CustomerID']= label_encoder.fit_transform(df11['CustomerID'])
df11.head(3)


# In[150]:


df11.shape


# # Grouping the Customers based on CustomerID:
# Taking the *mean* of the group

# In[151]:


df12 = df11.groupby(['CustomerID']).mean()
df12


# In[96]:


# statistical EDA on the dataframe:
df12.describe()


# In[153]:


y = df12.to_numpy()


# # Scaling: (MinMax scaler)
# It ranges from 0 to 1

# In[154]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = MinMaxScaler()
y_scaled = sc.fit_transform(y)
     


# # Modelling:
# Elbow plot

# In[155]:


from yellowbrick.cluster import KElbowVisualizer
model = KMeans(random_state=9)
visualizer = KElbowVisualizer(model, k=(1,15)).fit(y_scaled)
visualizer.show()


# # *Interpretation*:
# 
# The optimal value of k occurs at k=4
# k-means clustering:

# In[158]:


km = KMeans(n_clusters=4, init= "k-means++", random_state=9)     # applying k = 4
km.fit(y_scaled)          # fit the data - identify pattern, does everything

centroids = km.cluster_centers_   # final centroid points

# print("centroids: ",centroids)
print("inertia: ",km.inertia_)    # measures how tight my groups are. Lower the better


# In[159]:


km.labels_   # shows which group each datapoint belongs to


# In[173]:


# predict the labels of clusters
label = km.fit_predict(y_scaled)  
label


# # Visualizing the customer clusters:

# In[ ]:





# In[161]:


# Getting the Centroids and Cluster labels
centroids = km.cluster_centers_
labels = np.unique(label)

# 2D plotting
plt.figure(figsize=(5, 5)) 
for i in labels:
    plt.scatter(y_scaled[label == i , 3] , y_scaled[label == i , 4] , label = i)
plt.scatter(centroids[:,3] , centroids[:,4] , marker="X", c="k", s=120, label="centroids")
plt.legend()
plt.show()


# In[163]:


# 3D plotting
labels = np.unique(label)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(y_scaled[label == 0,3],y_scaled[label == 0,4],y_scaled[label == 0,2], s = 20 , color = 'purple', label = "cluster 0")  
ax.scatter(y_scaled[label == 1,3],y_scaled[label == 1,4],y_scaled[label == 1,2], s = 20 , color = 'green', label = "cluster 1")  
ax.scatter(y_scaled[label == 2,3],y_scaled[label == 2,4],y_scaled[label == 2,2], s = 20 , color = 'blue', label = "cluster 2")  
ax.scatter(y_scaled[label == 3,3],y_scaled[label == 3,4],y_scaled[label == 3,2], s = 20 , color = 'red', label = "cluster 3")  
ax.scatter(y_scaled[label == 4,3],y_scaled[label == 4,4],y_scaled[label == 4,2], s = 20 , color = 'yellow', label = "cluster 4")  

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()
     


# # *INTERPRETATION* of the above plot:
# 
# since the datapoints are closer to each other, there is no clear cluster formed.

# In[164]:


df13 = pd.DataFrame(label) # creating a dataframe for the "customer clusters"

df13.reset_index(level=0, inplace=True) # creating the "customerID" column

df13.rename(columns={'index':'CustomerID', 0:'Customer cluster'},inplace=True) # renaming the columns

df13.head()
     


# In[166]:


# count of each clusters
df14 = df13.groupby("Customer cluster").count()
df14
     


# # Plotting the Customer Clusters:

# In[167]:


# check the number of clusters and number of CUSTOMERS in each cluster
import seaborn as sns
sns.countplot(df13["Customer cluster"])
     


# # *INTERPRETATION*:
# 
# cluster-0 contains the maximum number of Customers, followed by cluster-3
# cluster-1,2 contains the least number of Customers.

# # Silhouette score:
# If the score is 1, the cluster is dense and well-separated from other clusters.
# A value near 0 represents overlapping clusters with samples very close to the decision boundary of the neighboring clusters.
# Look for a silhouette score closer to 1. This score varies from -1 to 1
# 

# In[168]:


from sklearn import metrics
score = metrics.silhouette_score(y_scaled, km.labels_ )
score


# # *Interpretation* of Silhouette score:
# 
# score of 0.81 shows that the clusters are not well seprated from each other, and it overlaps each other slightly.
# overall, it is a good score

# # INFERENCES:
# Thus we have grouped *Similar Customers* based on:
# Products they bought
# Quantity and Price of the purchase
# Country of origin of the customers
# We have found that the customers can be *segmented into 4 buckets*, based on their similarity.

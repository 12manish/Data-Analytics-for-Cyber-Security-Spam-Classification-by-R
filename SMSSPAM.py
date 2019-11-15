#!/usr/bin/env python
# coding: utf-8

# ## Project Objective
# ##
# 
# Spam detection is one of the major applications of Machine Learning in the interwebs today. Pretty much all of the major email service providers have spam detection systems built in and automatically classify such mail as 'Junk Mail'. 
# 
# In this project we will be using the Naive Bayes algorithm to create a model that can classify  SMS messages as spam or not spam, based on the training we give to the model. It is important to have some level of intuition as to what a spammy text message might look like. Usually they have words like 'free', 'win', 'winner', 'cash', 'prize' and the like in them as these texts are designed to catch your eye and in some sense tempt you to open them. Also, spam messages tend to have words written in all capitals and also tend to use a lot of exclamation marks. To the recipient, it is usually pretty straightforward to identify a spam text and our objective here is to train a model to do that for us!
# 
# Being able to identify spam messages is a binary classification problem as messages are classified as either 'Spam' or 'Not Spam' and nothing else. Also, this is a supervised learning problem, as we will be feeding a labelled dataset into the model, that it can learn from, to make future predictions. 

# 
# 
# 

# ### Step 1.1: Understanding our dataset ### 
# 
# The columns in the data set are currently not named and as you can see, there are 2 columns. 
# 
# The first column takes two values, 'ham' which signifies that the message is not spam, and 'spam' which signifies that the message is spam. 
# 
# The second column is the text content of the SMS message that is being classified.

# >** Instructions: **
# * Import the dataset into a pandas dataframe using the read_table method. Because this is a tab separated dataset we will be using '\t' as the value for the 'sep' argument which specifies this format. 
# * Also, rename the column names by specifying a list ['label, 'sms_message'] to the 'names' argument of read_table().
# * Print the first five values of the dataframe with the new column names.

# In[1]:


import pandas as pd
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir('C:\\Users\\SPEED TAIL\\Desktop\\Naive-Bayes-Spam')
df = pd.read_table('data', names = ['label', 'sms_message'])
##df.rename(columns = {'v1':'label','v2':'sms'},inplace=True)
# Output printing out first 5 columns
df.head()


# ### Step 1.2: Data Preprocessing ###
# 
# 

# >**Instructions: **
# * Convert the values in the 'label' column to numerical values using map method as follows:
# {'ham':0, 'spam':1} This maps the 'ham' value to 0 and the 'spam' value to 1.
# * Also, to get an idea of the size of the dataset we are dealing with, print out number of rows and columns using 
# 'shape'.

# In[2]:


df['label'] = df.label.map({'ham':0, 'spam':1})


# In[3]:


df.shape


# In[6]:


# adding a column to represent the length of the tweet

df['len'] = df['sms_message'].str.len()
df.head(10)


# In[7]:


# describing by labels

df.groupby('label').describe()
#So from the above results we can see that there is a message with 910 characters. 


# In[8]:


#Lets take a look at that message to see if the particular message is spam or ham
df[df['len']==910]['sms_message'].iloc[0]


# In[ ]:





# In[9]:


df['len'].plot(bins=50,kind='hist')
##If you see, length of the text goes beyond 800 characters (look at the x axis). 
##This means that there are some messages whose length is more than the others


# ## relation between spam messages and length

# In[10]:




plt.rcParams['figure.figsize'] = (10, 8)
sns.boxenplot(x = df['label'], y = df['len'])
plt.title('Relation between Messages and Length', fontsize = 25)
plt.show()


# ### checking the most common words in the whole dataset

# In[43]:


get_ipython().system('pip install WordCloud')


# In[12]:



from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'gray', width = 800, height = 800, max_words = 20).generate(str(df['sms_message']))

plt.rcParams['figure.figsize'] = (10, 10)
plt.title('Most Common words in the dataset', fontsize = 20)
plt.axis('off')
plt.imshow(wordcloud)


# ####  for ham and spam visualize it in pie chart

# In[13]:


sns.set(style="darkgrid")
plt.title('# of Spam vs Ham')
sns.countplot(df['label'])


# ## Now lets try to find some distinguishing feature between the messages of two sets of labels - ham and spam

# In[14]:


df.hist(column='len',by='label',bins=50,figsize=(12,7))


# In[ ]:





# In[ ]:





# In[15]:



size = [4825, 747]
labels = ['spam', 'ham']
colors = ['Cyan', 'lightblue']

plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%')
plt.axis('off')
plt.title('Pie Chart for Labels', fontsize = 20)
plt.legend()
plt.show()


# ## checking the most common words in spam messages

# In[16]:




spam = ' '.join(text for text in df['sms_message'][df['label'] == 0])

wordcloud = WordCloud(background_color = 'pink', max_words = 50, height = 1000, width = 1000).generate(spam)

plt.rcParams['figure.figsize'] = (10, 10)
plt.axis('off')
plt.title('Most Common Words in Spam Messages', fontsize = 20)
plt.imshow(wordcloud)


# ### checking the most common words in ham messages

# In[17]:




ham = ' '.join(text for text in df['sms_message'][df['label'] == 1])

wordcloud = WordCloud(background_color = 'purple', max_words = 50, height = 1000, width = 1000).generate(ham)

plt.rcParams['figure.figsize'] = (10, 10)
plt.axis('off')
plt.title('Most Common Words in Ham Messages', fontsize = 20)
plt.imshow(wordcloud)


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer()
words = cv.fit_transform(df.sms_message)

sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'Cyan')
plt.title("Most Frequently Occuring Words - Top 30")


# >>**Machine Learning Model Instructions:**
# Import the sklearn.feature_extraction.text.CountVectorizer method and create an instance of it called 'count_vector'. 

# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()


# In[20]:


'''
Practice node:
Print the 'count_vector' object which is an instance of 'CountVectorizer()'
'''
print(count_vector)


# >>**Instructions:**
# Convert the array we obtained, loaded into 'doc_array', into a dataframe and set the column names to 
# the word names(which you computed earlier using get_feature_names(). Call the dataframe 'frequency_matrix'.
# 

# ### Step 3.1: Training and testing sets ###
# 
# Now that we have understood how to deal with the Bag of Words problem we can get back to our dataset and proceed with our analysis. Our first step in this regard would be to split our dataset into a training and testing set so we can test our model later. 

# 
# >>**Instructions:**
# Split the dataset into a training and testing set by using the train_test_split method in sklearn. Split the data
# using the following variables:
# * `X_train` is our training data for the 'sms_message' column.
# * `y_train` is our training data for the 'label' column
# * `X_test` is our testing data for the 'sms_message' column.
# * `y_test` is our testing data for the 'label' column
# Print out the number of rows we have in each our training and testing data.
# 

# In[24]:


# split into training and testing sets
# USE from sklearn.model_selection import train_test_split to avoid seeing deprecation warning.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# In[27]:



# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)


# ### Step 5: Naive Bayes implementation using scikit-learn ###
# 
# Thankfully, sklearn has several Naive Bayes implementations that we can use and so we do not have to do the math from scratch. We will be using sklearns `sklearn.naive_bayes` method to make predictions on our dataset. 
# 
# Specifically, we will be using the multinomial Naive Bayes implementation. This particular classifier is suitable for classification with discrete features (such as in our case, word counts for text classification). It takes in integer word counts as its input. On the other hand Gaussian Naive Bayes is better suited for continuous data as it assumes that the input data has a Gaussian(normal) distribution.

# In[28]:


'''


We have loaded the training data into the variable 'training_data' and the testing data into the 
variable 'testing_data'.

Import the MultinomialNB classifier and fit the training data into the classifier using fit(). Name your classifier
'naive_bayes'. You will be training the classifier using 'training_data' and y_train' from our split earlier. 
'''


# In[29]:


from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)


# In[ ]:


'''
Instructions:
Now that our algorithm has been trained using the training data set we can now make some predictions on the test data
stored in 'testing_data' using predict(). Save your predictions into the 'predictions' variable.
'''


# In[30]:


'''
Solution
'''
predictions = naive_bayes.predict(testing_data)


# #### Now that predictions have been made on our test set, Now we need to check the accuracy of our predictions.

# ### Step 6: Evaluating our model ###
# 
# Now that we have made predictions on our test set, our next goal is to evaluate how well our model is doing. 
# 

# We will be using all 4 metrics to make sure our model does well. For all 4 metrics whose values can range from 0 to 1, having a score as close to 1 as possible is a good indicator of how well our model is doing.

# In[31]:


'''

Compute the accuracy, precision, recall and F1 scores of your model using your test data 'y_test' and the predictions
you made earlier stored in the 'predictions' variable.
'''


# In[32]:


'''
Solution
'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(predictions, y_test)))
print('Precision score: ', format(precision_score(predictions, y_test)))
print('Recall score: ', format(recall_score(predictions, y_test)))
print('F1 score: ', format(f1_score(predictions, y_test)))


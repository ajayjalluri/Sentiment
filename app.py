import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from wordcloud import WordCloud
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import nltk
import warnings
from PIL import Image
import pickle

nltk.download('punkt')
nltk.download('stopwords')

df1 = pd.read_csv('1st-part.csv')
df2 = pd.read_csv('2nd-part.csv')
df3 = pd.read_csv('3rd-part.csv')
df4 = pd.read_csv('4th-part.csv')
df5 = pd.read_csv('5th-part.csv')
df6 = pd.read_csv('6th-part.csv')
df7 = pd.read_csv('7th-part.csv')
df8 = pd.read_csv("8th-part.csv")

df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8])

page = st.sidebar.selectbox("Select Activity", ["Sentiment Analyser", "Visualization", "Word cloud"])



pkl_file1 = open('lr.pkl', 'rb')

lr = pickle.load(pkl_file1)



pkl_file2 = open('tfidf.pkl', 'rb')

tfidf = pickle.load(pkl_file2)


# st.write(page)

def label_data():
    rows = pd.read_csv('Amazon_Unlocked_Mobile.csv', header=0, index_col=False, delimiter=',')
    labels = []
    for cell in rows['Rating']:
        if cell >= 4:
            labels.append('2')  #Good
        elif cell == 3:
            labels.append('1')   #Neutral
        else:
            labels.append('0')   #Poor

    rows['Label'] = labels
    del rows['Review Votes']
    return rows


def clean_data(data):

    #replace blank values in all the cells with 'nan'
    df.replace('',np.nan,inplace=True)
    #delete all the rows which contain at least one cell with nan value
    df.dropna(axis=0, how='any', inplace=True)

    #save output csv file
    df.to_csv('labelled_dataset.csv', index=False)
    return data
clean_data(df)
df = pd.read_csv('labelled_dataset.csv')
df.head()




df = df.sample(frac=0.1, random_state=0) #uncomment to use full set of data

# Drop missing values
df.dropna(inplace=True)

# Remove any 'neutral' ratings equal to 3
df = df[df['Rating'] != 3]

# Encode 4s and 5s as 1 (positive sentiment) and 1s and 2s as 0 (negative sentiment)
df['Sentiment'] = np.where(df['Rating'] > 3, 1, 0)
df.head()




def cleanText(raw_text, remove_stopwords=False, stemming=False, split_text=False):
    '''
    Convert a raw review to a cleaned review
    '''
    text = BeautifulSoup(raw_text, "html.parser").get_text()  #remove html
    letters_only = re.sub("[^a-zA-Z]", " ", text)  # remove non-character
    words = letters_only.lower().split() # convert to lower case


    if remove_stopwords: # remove stopword
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    if stemming==True: # stemming
#         stemmer = PorterStemmer()
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]

    if split_text==True:  # split text
        return (words)

    return( " ".join(words))



if page == "Visualization" :
    st.header("Distribution of Rating")
    df1 = pd.DataFrame(data = [[1,5787],[2,2023],[4,5009],[5,17998]],columns = ["Rating","Count"])
    fig = px.pie(df1, values= "Count", names='Rating', title='Distribution of Rating')
    st.plotly_chart(fig,use_container_width=10)

    st.header("Number of Reviews for Top 20 Brands")
    brands = df["Brand Name"].value_counts()
    b = brands.to_frame()
    b = b.reset_index()
    b = b.iloc[0:20,:]
    b.columns = ["Brand Name","Number of Reviews"]

    fig = px.bar(b,x='Brand Name',y = "Number of Reviews",title="Number of Reviews for Top 20 Brands")
    st.plotly_chart(fig,use_container_width=10)

    st.header("Number of Reviews for Top 50 Brands")
    brands = df["Brand Name"].value_counts()
    b = brands.to_frame()
    b = b.reset_index()
    b = b.iloc[0:50,:]
    b.columns = ["Brand Name","Number of Reviews"]

    fig = px.bar(b,x='Brand Name',y = "Number of Reviews",title="Number of Reviews for Top 50 Brands")
    st.plotly_chart(fig,use_container_width=10)


    st.header("Number of Reviews for Top 20 products")
    brands = df["Product Name"].value_counts()
    b = brands.to_frame()
    b = b.reset_index()
    b = b.iloc[0:20,:]
    b.columns = ["Product Name","Number of Reviews"]
    fig = px.bar(b,x='Product Name',y = "Number of Reviews",title="Number of Reviews for Top 20 products")
    st.plotly_chart(fig,use_container_width=30)

    st.header("Number of Reviews for Top 50 products")
    brands = df["Product Name"].value_counts()
    b = brands.to_frame()
    b = b.reset_index()
    b = b.iloc[0:50,:]
    b.columns = ["Product Name","Number of Reviews"]

    fig = px.bar(b,x='Product Name',y = "Number of Reviews",title="Number of Reviews for Top 50 products")
    st.plotly_chart(fig,use_container_width=30)

    st.header("Distribution of Review Length")
    review_length = df["Reviews"].dropna().map(lambda x: len(x))
    df["Review length (Number of character)"] = review_length
    fig = px.histogram(df, x="Review length (Number of character)",title = "Distribution of Review Length" )

    st.plotly_chart(fig,use_container_width=20)

    st.header("Polarity Distribution")

    df3 =pd.DataFrame([["Positive",230674],["Neutral",26058],["Negative",77603]],columns= ["Polarity","Frequency"])

    fig = px.bar(df3,x='Polarity',y = "Frequency",title = "Polarity Distribution")

    st.plotly_chart(fig,use_container_width=20)



def create_word_cloud(brand, sentiment):

        df_brand = df.loc[df['Brand Name'].isin([brand])]
        df_brand_sample = df_brand.sample(frac=0.1)
        word_cloud_collection = ''

        if sentiment == 1:
            df_reviews = df_brand_sample[df_brand_sample["Sentiment"]==1]["Reviews"]

        if sentiment == 0:
            df_reviews = df_brand_sample[df_brand_sample["Sentiment"]==0]["Reviews"]

        for val in df_reviews.str.lower():
            tokens = nltk.word_tokenize(val)
            tokens = [word for word in tokens if word not in stopwords.words('english')]
            for words in tokens:
                word_cloud_collection = word_cloud_collection + words + ' '

        wordcloud = WordCloud(max_font_size=50, width=500, height=300).generate(word_cloud_collection)
        
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        plt.savefig('WC.jpg')
        img= Image.open("WC.jpg")
        return img

if page == "Word cloud" :

    st.header("Word cloud")



    form = st.form(key='my_form1')
    brand = form.text_input(label='Enter Brand Name')
    s = form.selectbox("Select The Sentiment",["Positive","Negative"])
    submit_button = form.form_submit_button(label='Plot Word Cloud')


    if submit_button:
        if s=="Positive" :
            img = create_word_cloud(brand,1 )
            st.image(img)
        else :
            img = create_word_cloud(brand,0 )
            st.image(img)


if page == "Sentiment Analyser":

    st.header("Product Review Prediction")


    form = st.form(key='my_form2')
    r = form.text_input(label='Enter Product Review')
    submit_button = form.form_submit_button(label='Predict Review')

    if submit_button :

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        S = cleanText(r)
        l = []
        l.append(S)
        pred = lr.predict(tfidf.transform(l))
        if int(pred) == 1 :
            st.header("Positive Sentiment")
        else :
            st.header("Negative Sentiment")

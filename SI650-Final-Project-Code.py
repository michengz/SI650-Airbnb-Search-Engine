#!/usr/bin/env python
# coding: utf-8

# # SI650 Final Project
# Michelle Cheng (michengz@umich.edu)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', 150)
import pyterrier as pt
from pyterrier.measures import *
import os
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-19.jdk/Contents/Home"
import warnings
warnings.filterwarnings('ignore')
import fastrank

#from pyterrier.batchretrieve import TextScorer


# ## Data Preparation

df_reviews = pd.read_csv("reviews.csv")
df_listings = pd.read_csv("listings.csv")
df_listings = df_listings.rename(columns={"id": "listing_id"})

# Concatenating Comments by Listings
df_reviews_concatenated = pd.DataFrame()
df_reviews['comments'] = df_reviews['comments'].apply(lambda x: str(x))
df_reviews_concatenated['listing_id'] = df_reviews.groupby(['listing_id'])['comments'].apply('\n'.join).index
df_reviews_concatenated['comments'] = df_reviews.groupby(['listing_id'])['comments'].apply('\n'.join).tolist()

df_joined = df_listings.join(df_reviews_concatenated.set_index('listing_id'), on = 'listing_id')
df = df_joined[['listing_id','name','description','neighborhood_overview','comments','neighbourhood',
                'neighbourhood_cleansed','neighbourhood_group_cleansed','property_type','room_type',
                'accommodates','beds','amenities','price','review_scores_rating','review_scores_cleanliness', 
                'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value', 
                'host_response_rate','listing_url','picture_url','host_url','host_name']]
df[['name', 'description','neighborhood_overview','comments']] = df[['name', 'description','neighborhood_overview','comments']].fillna('')
df['text'] = df['name'] + '; ' + df['description'] + '; ' + df['neighborhood_overview']+ '; ' + df['comments']

df = df.rename(columns={'listing_id':"docno"})
df['docno'] = df['docno'].apply(lambda x: str(x))

def airbnb_filter(location, room_type, accommodates, beds):
    return df[(df['neighbourhood_group_cleansed'] == location) & #'Manhattan'
              (df['room_type'] == room_type) & #'Private room'
              (df['accommodates'] == accommodates) & #2
              (df['beds'] == beds)] #1

df_filtered = airbnb_filter('Manhattan', 'Private room', 2, 1)



if not pt.started():
    pt.init(tqdm = 'notebook', logging='ERROR')

docs_df = df_filtered[['docno','text','description','amenities','review_scores_rating','review_scores_cleanliness','review_scores_value','review_scores_communication']]
docs_df['docno'] = docs_df['docno'].apply(lambda x: str(x))
docs_df['review_scores_rating'] = docs_df['review_scores_rating'].apply(lambda x: str(x))
docs_df['review_scores_cleanliness'] = docs_df['review_scores_cleanliness'].apply(lambda x: str(x))
docs_df['review_scores_value'] = docs_df['review_scores_value'].apply(lambda x: str(x))
docs_df['review_scores_communication'] = docs_df['review_scores_communication'].apply(lambda x: str(x))

index_dir = './airbnb_index'
if not os.path.exists(index_dir + "/data.properties"):
    pt.set_property("termpipelines", "Stopwords,PorterStemmer")
    indexer = pt.DFIndexer(index_dir, overwrite = True, stemmer = 'PorterStemmer',stopwords = 'Stopwords', tokeniser="UTFTokeniser")
    #indexer.setProperty("termpipelines", "Stopwords, PorterStemmer")
    #   indexer.setProperty("termpipelines","Stopwords,PorterStemmer")
    indexer.setProperties(**{
            "indexer.meta.forward.keylens":"26,2048",
            'metaindex.compressed.crop.long' : 'true'
        })
    index_ref = indexer.index(docs_df["text"], docs_df["docno"], docs_df["description"],docs_df["amenities"],docs_df["review_scores_rating"],docs_df["review_scores_cleanliness"],docs_df["review_scores_value"],docs_df["review_scores_communication"])
    
else:
    index_ref = pt.IndexRef.of(index_dir + "/data.properties")

index = pt.IndexFactory.of(index_ref)


# ## Preparing for Evaluation - Annotation

# Loading Queries
queries_df = pd.read_csv("queries.csv")

def remove_punc(q):
    return "".join([x if x!=',' else "" for x in q])
queries_df['query'] = queries_df['query'].apply(remove_punc)

# Retrieving documents using multiple models
models = ['BM25','PL2','TF_IDF','DPH']
df_retrieved = pd.DataFrame()
for model in models:
    br = pt.BatchRetrieve(index, wmodel = model, num_results = 100)
    temp_df = br(queries_df)
    df_retrieved = pd.concat([df_retrieved, temp_df])

# Sort and retrieve the top 100 docs
df_retrieved = df_retrieved.join(df_filtered[['docno','text','listing_url','price','host_name']].set_index('docno'), on = 'docno')
df_retrieved_cleansed = df_retrieved.drop_duplicates(subset=['qid','docno'], keep="first").drop_duplicates(subset=['qid','score','host_name'], keep="first")
df_retrieved_cleansed['qid'] = df_retrieved_cleansed['qid'].apply(lambda x:int(x))
df_annotation = df_retrieved_cleansed.sort_values(by=['qid','rank']).groupby('qid').head(100)


# Export annotation file
df_annotation.to_csv("annotation.csv", index = False)

# Query-doc pairs
n = 1
docs_df_dup = docs_df
while n < 20:
    docs_df_dup = pd.concat([docs_df_dup,docs_df], axis = 0)
    n += 1
queries_df_dup = queries_df.loc[queries_df.index.repeat(len(docs_df))]
query_doc_df = pd.concat([queries_df_dup.reset_index(),docs_df_dup.reset_index()],axis = 1)[['qid','query','docno','text']]

# Load annotated labels
annotated_df = pd.read_csv('annotated.csv')
annotated_df['docno'] = annotated_df['docno'].apply(lambda x: str(x))
annotated_df['qid'] = annotated_df['qid'].apply(lambda x: str(x))

# Construct qrels dataframe
qrels = query_doc_df.merge(annotated_df, on=['docno','qid'],how = 'left')[['qid','docno','label']]
qrels['label'] = qrels['label'].fillna(1)
qrels['label'] = qrels['label'].apply(lambda x: 1 if x==0 else x)
qrels['label'] = qrels['label'].astype(int)


# Train-test Split
from sklearn.model_selection import train_test_split

tr_va_topics, test_topics = train_test_split(queries_df, test_size=0.5, random_state=42)
train_topics, valid_topics =  train_test_split(tr_va_topics, test_size=0.33, random_state=42)


# ### Baseline Model

bm25 = pt.BatchRetrieve(index, wmodel = 'BM25')
bm25_qe = pt.BatchRetrieve(index, wmodel = 'BM25', controls={"qe":"on", "qemodel" : "Bo1"})

# ### Custom BM25

def bm25_custom_weighting(keyFreq, posting, entryStats, collStats):

    N = collStats.getNumberOfDocuments() # number of documents
    df = entryStats.getDocumentFrequency() # number of documents that contain the term
    tf = posting.getFrequency() # term frequency in document
    dl = posting.getDocumentLength() # document length
    avdl = collStats.getAverageDocumentLength() # average document length
    qtf = keyFreq # term frequency in query
    
    mf = entryStats.getMaxFrequencyInDocuments() # maximum in-document term frequency of the term among all documents
    tt = entryStats.getFrequency() # total number of occurrences of the term
    W = collStats.getNumberOfTokens() # total number of tokens
    avtf = (tt/df)/(W/N) # average term frequency
    
    k1 = 1.2
    k3 = 8
    b = 0.75
    a = 0.5
    c = 1

    idf = np.log((N-df+0.5)/(df+0.5))
    normalized_qtf = ((k3+1)*qtf)/(k3+qtf)
    normalized_tf  = ((k1+1)*tf)/(k1*(3-(a+b+c)+b*(dl/avdl)+c*(avtf/(mf-avtf))+a*(avtf/tf))+tf) 
    
    score =  idf * normalized_qtf * normalized_tf
    
    return score

custom_bm25_qe = pt.BatchRetrieve(index, wmodel = bm25_custom_weighting, controls={"qe":"on", "qemodel" : "Bo1"})


# ### Learning to Rank

def get_ratings(row):
    total = 0
    for i in ['review_scores_rating','review_scores_cleanliness','review_scores_value','review_scores_communication']:
        try: 
            total += float(row[i])
        except:
            total += 0
    if str(total).isnumeric() == False:
        total = 0
#     print(total)
    return total

ltr_feats1 = (custom_bm25_qe) >> pt.text.get_text(index, ["description",'amenities','review_scores_rating','review_scores_cleanliness','review_scores_value','review_scores_communication']) >> (
    pt.transformer.IdentityTransformer()
    ** # Score of Description
    pt.text.scorer(body_attr="description", takes='docs', wmodel=bm25_custom_weighting) 
    ** # Abstract Coordinate Match
    pt.BatchRetrieve(index, wmodel="CoordinateMatch")
    ** # Get ratings
    pt.apply.doc_score(get_ratings)
)
fnames=["BM25", 'description', "CoordinateMatch",'ratings']


qrels['label'] = qrels['label'].apply(lambda x: float(x))


train_request = fastrank.TrainRequest.coordinate_ascent()

params = train_request.params
params.init_random = True
params.normalize = True
params.seed = 1234567

ca_pipe = ltr_feats1 >> pt.ltr.apply_learned_model(train_request, form='fastrank')
ca_pipe.fit(train_topics, qrels)


# ## Interaction

# def retrieve_airbnb():
#     print('Please enter your Airbnb search:')
#     x = input()
#     print('Retrieving Results...\n')
#     retrieved_docnos = ca_pipe(x)[['docno']].head(10)
#     output = retrieved_docnos.merge(df_filtered, on = 'docno', how = 'left')[['name','description','neighbourhood_cleansed','host_name','property_type','listing_url','picture_url']]
#     for i in range(10):
#         print(f'{output.iloc[i,:]["name"]}\n')
#         print(f'{output.iloc[i,:]["neighbourhood_cleansed"]}\n')
#         print(f'{output.iloc[i,:]["description"]}\n')
#         print(f'{output.iloc[i,:]["listing_url"]}\n')
#         print('\n')
#    return output


# if __name__ == '__main__':
#     retrieve_airbnb()


import streamlit as st

def main():
    st.title('New York Airbnb Search')
    search = st.text_input('Enter your search (for example: "modern luxurious hotel near subway station"):')
    if search:
        retrieved_docnos = ca_pipe(search)[['docno']].head(10)
        outputs = retrieved_docnos.merge(df_filtered, on = 'docno', how = 'left')[['name','description','neighbourhood_cleansed','host_name','property_type','listing_url','picture_url']]
        for i in range(10):
            # col1, col2, col3 = st.columns([1,6,1])
            # with col1:
            #     st.write("")
            # with col2:
            #     st.image(outputs.iloc[i,:]['picture_url'], width=600)
            # with col3:
            #     st.write("")
            st.image(outputs.iloc[i,:]['picture_url'], width=700)
            st.write(f"[{outputs.iloc[i,:]['name']}]({outputs.iloc[i,:]['listing_url']})")
            st.write(outputs.iloc[i,:]['neighbourhood_cleansed'])
            st.write(outputs.iloc[i,:]['description'])
            st.write(" ")
            # st.write(outputs.iloc[i,:]['listing_url'])

if __name__ == '__main__':
    main()

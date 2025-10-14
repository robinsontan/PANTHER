"""
Data preprocessing script for the pre-training on CCT behavior sequence data.
"""

import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch

CCT_BEHSEQ_DATA_PATH = os.getenv('CCT_BEHSEQ_DATA_PATH')

# ========== Generate behavior sequence data for CCT ==========
# - Download the raw data from https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset

DATA_PATH = "./dataset/credit-card-transactions/credit_card_transactions-ibm_v2.csv"
CARD_ATTR_DATA_PATH = "./dataset/credit-card-transactions/sd254_cards.csv"
USER_ATTR_DATA_PATH = "./dataset/credit-card-transactions/sd254_users.csv"

df = pl.read_csv(DATA_PATH)
df_card = pl.read_csv(CARD_ATTR_DATA_PATH)
df_user = pl.read_csv(USER_ATTR_DATA_PATH)

df_full = df.join(df_card, left_on=["User", "Card"], right_on=["User", "CARD INDEX"], how="left")\
            .join(df_user.with_row_index().cast({"index": pl.Int64}), left_on="User", right_on="index", how="left")

df_full = df_full.filter(pl.col("Is Fraud?") == "No")  # Use non-fraud transactions to train the model

beh_feature_columns = ["Amount", "Use Chip", "Merchant Name", "Merchant City", "Merchant State", "MCC"]
card_attr_columns = ["Card Number", "Card Brand", "Card Type", "Has Chip", "Cards Issued", "Credit Limit", "FICO Score", "Num Credit Cards", "Current Age", "Gender", "Per Capita Income - Zipcode", "Yearly Income - Person"]


df_seq = df_full.with_columns(pl.col(c).cast(str) for c in beh_feature_columns)\
                       .group_by(card_attr_columns)\
                       .agg(beh_feature_columns)\
                       .with_columns(pl.col("Amount").list.len().alias("seq_length"))\
                       .with_columns(pl.col(c).list.join(",") for c in beh_feature_columns)\
                       .with_columns(pl.col("Gender").replace({"Male":0, "Female":1}).cast(int))\
                       .sample(fraction=1, shuffle=True, seed=42)


df_seq.write_parquet(CCT_BEHSEQ_DATA_PATH)

# ========== Pre-process the behavior sequence data ==========
# - Bucketize numerical features

bin_dict=dict()

num_bins_dict={  # bucketize configurations for each feature
    'Amount':100,
    'Credit Limit':100,
    'Per Capita Income - Zipcode':100,
    'Yearly Income - Person':100,
}

def parse_str(x, d, column, num_bins, min_val=None, max_val=None):
    if column == 'Amount':
        x = [use_bin(x,column,num_bins) for x in x.split(',')]
        return ','.join([str(d[str(k)]) for k in x])
    if column in ['Credit Limit','Per Capita Income - Zipcode','Yearly Income - Person',]:
        x = my_float(x)
        x = use_bin(x,column,num_bins)
        return x
    return ','.join([str(d[k]) for k in x.split(',')])

def log_binning(min_val, max_val, num_bins, data, column,eps=1e-1):
    """
    Binning function using logarithmic mapping for data distribution.
    
    Args:
        min_val (float): Minimum value
        max_val (float): Maximum value
        num_bins (int): Number of bins
        data (list or numpy array): Data to be binned
        column (str): Column name for binning
    
    Returns:
        list: Bin indices for each data point
    """
    data_sorted = np.sort(data)
    buckets = np.array_split(data_sorted, num_bins)
    bins=[]
    for i in buckets:
        if len(i)>0:
            bins.append(i[0])
    np.array(bins)
    bin_dict[column]=bins
    bin_indices = np.digitize(data, bins, right=True)
    bin_indices+=1
    assert bin_indices.min() >= 1
    bin_counts = Counter(bin_indices)
    return bin_indices, bin_counts

def use_bin(x,column,num_bins):
    x=my_float(x)
    bins = bin_dict[column]
    bin_indices = np.digitize(x, bins, right=True)
    bin_indices = np.clip(bin_indices, 1, num_bins)
    assert bin_indices.min() >= 1
    assert bin_indices.max() <= num_bins
    return bin_indices

def plot_histogram(bin_counts, num_bins,name):
    """
    Plot frequency distribution histogram.
    
    Args:
        bin_counts (dict): Frequency count for each bin
        num_bins (int): Number of bins
        name (str): Name for saving the plot
    """
    bins = list(range(1, num_bins + 1))
    counts = [bin_counts[bin_num] for bin_num in bins]
    plt.bar(bins, counts, align='center')
    plt.xlabel('bins #')
    plt.ylabel('counts')
    plt.title('counts distribution')
    plt.xticks(bins)
    plt.savefig(f'fig/{name}')

def my_float(x):
    if x[0]=='$':
        return float(x[1:])
    else:
        return float(x)

def process_dataframe(df, element_column=set(),ignore_columns=set(), seq_columns=set()):
    encoding_dict = dict()
    min_dict, max_dict = dict(), dict()
    total_length = df['seq_length'].sum()
    print('total_length: ',total_length)
    for column in df.columns:
        print(column,end=':')
        if column in num_bins_dict.keys():
            num_bins = num_bins_dict[column]
        if column in ignore_columns:
            df.drop(column, axis=1, inplace=True)
            print('droped')
            continue
        
        if column in element_column:
            if column in ['Credit Limit','Per Capita Income - Zipcode','Yearly Income - Person',]:
                all_str = df[column].str.cat(sep=',').split(',')
                all_str = [my_float(x) for x in all_str]
                min_dict[f"{column}_min"] = min(all_str)
                max_dict[f"{column}_max"] = max(all_str)
                all_str, bin_counts = log_binning(min_val=min(all_str), max_val=max(all_str), num_bins=num_bins, data=all_str,column=column)
                plot_histogram(bin_counts, num_bins,column)
                unique_values=set(all_str)
            else:
                unique_values = df[column].unique()
        else:
            all_str = df[column].str.cat(sep=',').split(',')
            if column == 'Amount':
                all_str = [my_float(x) for x in all_str]
                min_dict[f"{column}_min"] = min(all_str)
                max_dict[f"{column}_max"] = max(all_str)
                all_str, bin_counts = log_binning(min_val=min(all_str), max_val=max(all_str), num_bins=num_bins, data=all_str,column=column)
                plot_histogram(bin_counts, num_bins,column)
            assert len(all_str) == total_length
            unique_values = set(all_str)
        print(len(unique_values))
        encoding_dict[column] = {str(value): idx + 1 for idx, value in enumerate(unique_values)}
        # Apply changes
        print('apply changes...')
        if column in element_column:
            if column in ['Credit Limit','Per Capita Income - Zipcode','Yearly Income - Person',]:
                df[column] = df[column].apply(lambda x: encoding_dict[column][str(use_bin(x,column,num_bins))])
            else:
                df[column] = df[column].apply(lambda x: encoding_dict[column][str(x)])
        else:
            if column == 'Amount':
                df[column] = df[column].apply(lambda x: parse_str(x, encoding_dict[column], column, num_bins, min_val=min_dict[f"{column}_min"], max_val=max_dict[f"{column}_max"]))
            else:
                df[column] = df[column].apply(lambda x: parse_str(x, encoding_dict[column], column, num_bins))
            assert (df[column].apply(lambda x: len(x.split(','))) == df['seq_length']).all()
    # Count attribute pairs
    print('count attr pairs...')
    attr_pair = set(zip(*[df[i].str.cat(sep=',').split(',') for i in seq_columns]))
    attr_pair = {i:idx+1 for idx, i in enumerate(attr_pair)}
    encoding_dict['beh_seq'] = attr_pair
    return encoding_dict, min_dict, max_dict

df = pd.read_parquet(CCT_BEHSEQ_DATA_PATH)

print(df.info())
print(df.head())
print(df['seq_length'].sum())

print('delete short seq')
df = df[df['seq_length']>=30]
print(df['Card Number'].nunique())
print(df.info())
print(df['seq_length'].sum())

print(df['seq_length'].min())

element_column = ['Card Number','Card Brand', 'Card Type', 'Has Chip', 
                  'Cards Issued', 'Num Credit Cards', 'Current Age',
                  'Gender','Credit Limit','FICO Score','Per Capita Income - Zipcode',
                  'Yearly Income - Person'
                  ]
ignore_columns = {'seq_length','Merchant State', 'Merchant Name','Merchant City',}
seq_columns = ['Amount','MCC','Use Chip']

encoding_dict, min_dict, max_dict = process_dataframe(df, element_column=element_column, ignore_columns=ignore_columns, seq_columns=seq_columns)


# Save the auxiliary files for pre-training
with open('./encoding_dict.pkl', 'wb') as f:
    pickle.dump(encoding_dict, f)

with open('./bin_dict.pkl', 'wb') as f:
    pickle.dump(bin_dict, f)

print("Encoding Dictionary:")
print(df.head(2))

dim_dict = {
    'beh_seq': 50,
}
for i in element_column:
    dim_dict[i] = 4
for i in seq_columns:
    dim_dict[i] = 8
print('dim_dict.keys:',len(dim_dict.keys()))

embedding_num_dict = {k: len(v) for k, v in encoding_dict.items()}
for k,v in embedding_num_dict.items():
    print(f"length of dicts {k}:", v)
    
torch.save(
    {
        'vocab_size': embedding_num_dict,
        'emb_dim':dim_dict,
        'seq_columns':seq_columns,
        'element_column':element_column,
    }
    , './embedding_num_dict.pt')


def map_dataframe(df, map_dict, seq_columns):
    attr_pair = map_dict['beh_seq']
    def get_beh_idx(row):
        ans = [row[i].split(',') for i in seq_columns]
        ans = list(zip(*ans))
        ans = [str(attr_pair[i]) for i in ans]
        return ','.join(ans)
    df['beh_seq'] = df.apply(get_beh_idx, axis=1)
    return df


df_processed = map_dataframe(df, map_dict=encoding_dict,seq_columns=seq_columns)

# Save the processed behavior sequence
df_processed.to_parquet('./beh_seq_combined.parquet')

print(df_processed.info())
print(df_processed.head(2))


# ====== Data pre-process for the downstream fraud detection task =======

for dirname, _, filenames in os.walk('./dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df0=pd.read_csv("./dataset/sd254_users.csv")
df0.head()

df0['areaincome']=df0['Per Capita Income - Zipcode'].replace({'\$':''},regex=True).astype(float)
df0['income']=df0['Yearly Income - Person'].replace({'\$':''},regex=True).astype(float)
df0['debt']=df0['Total Debt'].replace({'\$':''},regex=True).astype(float)
df0['retire']=(df0['Current Age']>df0['Retirement Age']).astype(int)
print(df0.info())
df0.head()

customer = df0[['Person', 'Current Age', 'areaincome', 'income', 'debt', 'FICO Score', 'Num Credit Cards']]
customer.head()

df2=pd.read_csv("./dataset/sd254_cards.csv")
df2.head(10)

df2.describe(include='all')

df2['creditlimit']=df2['Credit Limit'].replace({'\$':''},regex=True).astype(float)
df2['opendate']=pd.to_datetime(df2['Acct Open Date'], format="%m/%Y")
card =df2[['User', 'CARD INDEX', 'Card Brand','Card Type','Has Chip', 'Cards Issued', 'creditlimit', 'opendate']]
df2.head()


df3=pd.read_csv("./dataset/credit_card_transactions-ibm_v2.csv")
df3.head()

# add card number to df3
ucn=df2[['User', 'CARD INDEX','Card Number']]
ucn=ucn.rename(columns={'CARD INDEX':'Card'})
df3 = pd.merge(df3, ucn, on=['User', 'Card'], how='left')
null_count = df3['Card Number'].isna().sum()
print(null_count)

df3.info()

alltxn=df3.copy()
# create date time column
alltxn['datetime']= pd.to_datetime(alltxn['Year'].astype('str')+'-'+
                                   alltxn['Month'].astype('str')+'-'+
                                   alltxn['Day'].astype('str')+' '+ alltxn['Time'].astype('str'))
alltxn=alltxn.sort_values(['User', 'datetime']).reset_index(drop=True)

# create the variable calculate days since last transaction
alltxn['hour_last_txn']=(alltxn.sort_values(['User', 'datetime']).groupby('User')['datetime'].diff().dt.total_seconds()/3600)

alltxn['amount']=alltxn['Amount'].replace({'\$':''},regex=True).astype(float)
alltxn.head()

alltxn.drop(columns=['Year', 'Month', 'Day', 'Time', 'Amount', 'Merchant Name','Merchant State', 'Zip'], inplace=True)
alltxn.head()

# Find the transaction amount pattern, average, median, standard deviation
# For better model, one might want to test different time frame, such as 7-day, 60 day etc
alltxn['ave30']=alltxn.groupby('User').rolling('30D', on='datetime')[['amount']].mean().reset_index().amount.shift()
alltxn['median30']=alltxn.groupby('User').rolling('30D', on='datetime')[['amount']].mean().reset_index().amount.shift()
alltxn['std30']=alltxn.groupby('User').rolling('30D', on='datetime')[['amount']].std().reset_index().amount.shift()
alltxn

# Impute missing value
alltxn.fillna(alltxn.median(numeric_only=True), inplace=True)

# Select Online transactions
online =alltxn[alltxn['Use Chip']=='Online Transaction'].reset_index(drop=True)
online.head()

# Sampling - select all fraud and select 5 non-fraud for every fraud.
# Would like to use more non-raud, such as 10 non-fraud for every fraud, but it appears to be running too long in the grid  search
fraud = online[online['Is Fraud?']=='Yes'].copy()
nonfraud=online[online['Is Fraud?']=='No'].sample(n=len(fraud)*5,random_state=42)
# Assign weight
fraud['weight']=1
nonfraud['weight']=len(online[online['Is Fraud?']=='No'])/len(nonfraud)

# Create development/ training sample
devsample = pd.concat([fraud, nonfraud])
devsample['fraud']=devsample['Is Fraud?'].apply(lambda x: 1 if x=='Yes' else 0)
devsample.head()

devsample.tail()

devsample.info()

#Check the columns and rows are the same, before and after the merge
# merge with customer information
print(devsample.shape)
devsample=devsample.merge(customer[['areaincome', 'income', 'debt', 'FICO Score', 'Num Credit Cards']],
                                   how='left', left_on='User', right_index=True, indicator=True)
print(devsample.shape)
devsample.head()

# Check the merger result
devsample._merge.value_counts()

# Merge the Card Information
#Check the columns and rows are the same, before and after the merge
devsample.drop(columns='_merge', inplace=True)
print(devsample.shape)
devsample=devsample.merge(card, how='left', left_on=['User', 'Card'], right_on=['User', 'CARD INDEX'], indicator=True)
print(devsample.shape)
devsample.head()

'Card Number' in devsample.columns

devsample.info()

# delete unecessary columns
devsample.drop(columns=['CARD INDEX', '_merge'], inplace=True)

# create additional features/ variables for the model, such as how the compare with historical information
devsample['days_since_open']=(devsample['datetime']-devsample['opendate']).dt.days
devsample['pct_over30ave']=devsample['amount']/devsample['ave30']*100
devsample['pct_over30med']=devsample['amount']/devsample['median30']*100
devsample['std_30hist']=(devsample['amount']-devsample['ave30'])/devsample['std30']
devsample.head()

# check for any missing value
print(devsample.isnull().sum().sum())

devsample.info()

devsample.to_csv('./devsample.csv', index=False)

import numpy as np
import pandas as pd

#read dataset
df = pd.read_csv('C:\\Users\\Bosch\\Desktop\\Mestrado\\MLFA\\2º Semestre\\CSC\\aula1\\flights_dataset.csv')
#dataset info
print(df.info())
print('--------')
#drop unwanted columns
df.drop(['hour','minute','tailnum'], 1, inplace=True)
#infer objects type
df.infer_objects()
#check and replace missing with -99 (masking)
print(df.isnull().sum())
print('--------')
df.fillna(-99, inplace=True)
#frequency distribution of categories within a feature
print(df['dest'].unique())
print('Unique count: %d' %df['dest'].value_counts().count())
print('--------')
print(df['dest'].value_counts())
print('--------')
'''
Function to encode all non-(int/float) features in a dataframe.
For each column, if its dtype is neither int or float, get the list of unique values,
store the relation between the label and the integer that encodes it and apply it.
Return a labelled dataframe and a dictionary label to be able to restore the original value.
'''
def label_encoding(df):
    dic = {}
    for col in df.columns:
        if df[col].dtype == np.object:
            dic[col] = {}
    for col,dicCol in dic.items():
        i = 0
        while i < df[col].value_counts().count():
            dicCol[df[col].unique()[i]] = i
            i += 1
    df_labelled = df.replace(to_replace=dic, value=None)
    #df_labelled = pd.get_dummies(df,columns=dic.keys())
    return df_labelled,dic

'''
Function to decode what was previously encoded - get the original value!
'''
def label_decoding(df_labelled, label_dictionary):
    dic = {}
    for col,dicCol in label_dictionary.items():
        dic[col] = dict((y,x) for x,y in dicCol.items())
    df = df_labelled.replace(to_replace=dic, value=None)
    '''
    cols = df_labelled.columns.to_series().values
    newDf = pd.DataFrame(np.repeat(cols[None, :], len(df_labelled), 0)[df_labelled.astype(bool).values], df_labelled.index[df_labelled.any(1)])
    '''
    return df

df_labelled, label_dictionary = label_encoding(df)
print(df_labelled['dest'].unique())
print('Unique count after Label Encoding: %d' %df_labelled['dest'].value_counts().count())
df_labelled_decoded = label_decoding(df_labelled, label_dictionary)
print(df_labelled_decoded['dest'].unique())
print('Unique count after dec.: %d' %df_labelled_decoded['dest'].value_counts().count())
print('--------')
'''
Use a pandas' function to apply one-hot encoding to the origin column
'''
print(df.columns.values)
df_pandas_ohe = pd.get_dummies(df,columns=['origin'])
print(df_pandas_ohe.columns.values)
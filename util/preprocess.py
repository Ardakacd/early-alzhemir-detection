import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# preprocess.py file includes variety of helper functions that can be used for data preprocessing
# e.g. - normalization, encoding, etc.

"""
Def: Below function can be used to ONE-HOT encode desired columns
Params: 
-   dataframe : dataframe that is desired to be operated on
-   encode_list : list of columns (column names - string) to be one-hot encoded 
Returns:
- dataframe that consists encoded columns (provided) and already existing columns (concatenated)
"""
def one_hot_encode(dataframe, encode_list):
    # Select the columns to encode
    df_to_encode = dataframe[encode_list]

    other_columns = dataframe.drop(encode_list, axis=1)

    # Initialize OneHotEncoder
    encoder = OneHotEncoder()

    # Fit and transform the encoded DataFrame
    encoded_array = encoder.fit_transform(df_to_encode)

    # Convert the encoded array back to a DataFrame
    df_encoded_onehot = pd.DataFrame(encoded_array.toarray(), columns=encoder.get_feature_names_out(df_to_encode.columns))

    return pd.concat([other_columns, df_encoded_onehot], axis=1)

"""
Def: sep_column() can be used to seperate one specific column from dataframe (mostly column which is the target column/label)
Params: 
-   dataframe : dataframe that is desired to be operated on
-   column_name : column name (string) to be seperated from the datframe
Returns:
- returns X and y where X is the remaining columns of dataframe and y is the single column which is the seperated one
"""
def sep_column(dataframe, column_name):
    X = dataframe.drop(column_name, axis=1)
    y = dataframe[column_name]
    return X, y


"""
Def: min_max_norm() is simply used to apply min-max normalization on selected columns*
* selected columns - columns that we wish to normalize (e.g. : consists too small/large values)
Params: 
-   dataframe : list of dataframes that is desired to be operated on (they will be normalized sequentially)
-   columns : list of columns (column names - string) to be normalized
-   min_vals : minimum values of each attribute  
-   max_vals : maximum values of each attribute  
** min_vals and max_vals are pandas objects (dictionary-like object such as (att. name : value) pairs)
Returns:
- list of dataframes where provided columns of each dataframe is normalized
(although a value is being returned, operations are actually in-place)
"""
def min_max_norm(dataframes, columns, min_vals, max_vals):
    
    for col in columns:
        for df in dataframes:
            if col in df:
                df[col] = (df[col] - min_vals[col]) / (max_vals[col] - min_vals[col])

    return dataframes

"""
Def: given dataframe, list of word embeddings and column list - this function simply replaces each columns in the column list
     with the corresponding word embedding (vector)
Params: 
-   dataframe : dataframe that is desired to be operated on
-   word_embeddings : dictionary consists of "word : vector" pairs
-   columns : list of columns (column names - string) to be replaced with word embedding vector
Returns:
- dataframe where provided columns are replaced with their word embedding vectors
(although a value is being returned, operations are actually in-place)
"""

"""
TODO : dimension of vectors is hardcoded (100), it may be implemented in a dynamic way
"""

def replace_with_word_embeddings(dataframe, word_embeddings, columns):
    value_dict = {}

    for col in columns:
        for col_values in dataframe[col].unique():
            value_dict[col_values] = word_embeddings[col_values.lower()].numpy()

        replaced_col = dataframe[col].map(value_dict).values.tolist()
        holder_df = pd.DataFrame()

        for i in range(100):
            holder_df[col + str(i)]  = [column[i] for column in replaced_col]

        dataframe.drop(col,axis=1,inplace=True)
        dataframe = pd.concat([dataframe.reset_index(drop=True),holder_df],axis=1)

    return dataframe

"""
Def: carry the provided column to the end of the dataframe
Params: 
-   dataframe : dataframe that is desired to be operated on
-   column_name : (string) column name
Returns:
- dataframe where provided column is moved to the end (right) of the dataframe
(although a value is being returned, operations are actually in-place)
"""
def put_the_column_at_end(dataframe,column_name):
    column_to_move = dataframe.pop(column_name)
    dataframe[column_name] = column_to_move
    return dataframe
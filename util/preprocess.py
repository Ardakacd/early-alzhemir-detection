import pandas as pd
from sklearn.preprocessing import OneHotEncoder

"""
TODO
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
TODO
"""
def sep_column(dataframe, column_name):
    X = dataframe.drop(column_name, axis=1)
    y = dataframe[column_name]
    return X, y


"""
TODO
"""
def min_max_norm(dataframes, columns, min_vals, max_vals):
    
    for col in columns:
        for df in dataframes:
            if col in df:
                df[col] = (df[col] - min_vals[col]) / (max_vals[col] - min_vals[col])

    return dataframes

"""
TODO

value_dict = {}
for col in word_embedded_columns:
    for col_values in X_train[col].unique():
        value_dict[col_values] = word_embeddings[col_values.lower()].numpy()

gender_column_train = X_train['Gender'].map(value_dict).values.tolist()
gender_column_test = X_test['Gender'].map(value_dict).values.tolist()

gender_train_dataset = pd.DataFrame()
gender_test_dataset = pd.DataFrame()

for i in range(100):
    gender_train_dataset["Gender" + str(i)]  = [column[i] for column in gender_column_train]
    gender_test_dataset["Gender" + str(i)]  = [column[i] for column in gender_column_test]

X_train.drop("Gender",axis=1,inplace=True)
X_test.drop("Gender",axis=1,inplace=True)

X_train = pd.concat([X_train.reset_index(drop=True),gender_train_dataset],axis=1)
X_test = pd.concat([X_test.reset_index(drop=True),gender_test_dataset],axis=1)

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

def put_the_column_at_end(dataframe,column_name):
    column_to_move = dataframe.pop(column_name)
    dataframe[column_name] = column_to_move
    
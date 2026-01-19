import pandas as pd

def preprocess_data(df):
    """
    Basic preprocessing: remove duplicates, handle missing values, encode categorical features.
    """
    df = df.drop_duplicates()
    df = df.fillna(0)  # or a more sophisticated strategy
    # Example: convert categorical columns to numeric if needed
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes
    return df

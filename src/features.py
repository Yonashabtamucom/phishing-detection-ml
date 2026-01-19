def create_features(df, target_column="Result"):
    """
    Split the DataFrame into features and target.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

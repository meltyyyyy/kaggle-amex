
def transform_dtype(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float16')
        if df[col].dtype == 'float32':
            df[col] = df[col].astype('float16')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int8')
        if df[col].dtype == 'int32':
            df[col] = df[col].astype('int8')
    return df

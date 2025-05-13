import os, pandas as pd, numpy as np, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load data
def load_data(path, encoding='utf-8'):
    return pd.read_csv(path, encoding=encoding)

# Drop duplicates
def drop_duplicates(df):
    return df.drop_duplicates(keep='first').reset_index(drop=True)

# IQR clipping
def clip_outliers_iqr(df, cols, factor=1.5):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        
        df[col] = np.clip(df[col], lower, upper)
    
    return df

# Build preprocessor
def build_preprocessor(numeric_feats, categorical_feats):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('scaler', StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats)
    ])

# Preprocessing
def preprocess(input_csv, output_dir):
    # Load & clean
    df = load_data(input_csv)
    df = drop_duplicates(df)

    # Identify columns
    target = 'Obesity'
    numeric = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col != target]
    categorical = [col for col in df.select_dtypes(include=['object']).columns if col != target]
    
    # Outlier clipping
    df = clip_outliers_iqr(df, numeric)
    
    # Encode target
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    
    # Preprocessing pipeline
    preproc = build_preprocessor(numeric, categorical)
    X = df.drop(columns=[target])
    y = df[target]
    X_proc = preproc.fit_transform(X)
    
    # Assemble output
    ohe_cols = preproc.named_transformers_['cat']['onehot'].get_feature_names_out(categorical)
    cols = numeric + list(ohe_cols)
    df_out = pd.DataFrame(X_proc, columns=cols)
    df_out[target] = y.values
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'Obesity_preprocessing.csv')
    df_out.to_csv(csv_path, index=False)
    
    joblib.dump(preproc, os.path.join(output_dir, 'obesity_preprocessor.joblib'))
    joblib.dump(le, os.path.join(output_dir, 'obesity_label_encoder.joblib'))
    
    print(f"✅ Preprocessed data saved to {csv_path}")

if __name__ == '__main__':
    import argparse
    from sklearn.pipeline import Pipeline

    parser = argparse.ArgumentParser(description='Automate Obesity Data Preprocessing')
    parser.add_argument('--input',  required=True, help='Path to obesity.csv')
    parser.add_argument('--output', default='preprocessing', help='Folder output')
    args = parser.parse_args()

    preprocess(args.input, args.output)

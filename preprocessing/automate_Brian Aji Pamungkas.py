import os, pandas as pd, numpy as np, joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load a CSV file into a DataFrame.
def load_data(path, encoding='utf-8'):
    return pd.read_csv(path, encoding=encoding)

# Drop duplicates
def drop_duplicates(df):
    return df.drop_duplicates(keep='first').reset_index(drop=True)

# Clip outliers using the IQR method
def clip_outliers_iqr(df, cols, factor=1.5):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        df[col] = np.clip(df[col], lower, upper)
    
    return df

# Build preprocessing pipeline
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

# Main preprocessing workflow
def preprocess(input_csv, output_dir):
    df = load_data(input_csv)
    df = drop_duplicates(df)

    target = 'Obesity'
    numeric = [c for c in df.select_dtypes(include=['int64', 'float64']).columns if c != target]
    categorical = [c for c in df.select_dtypes(include=['object']).columns if c != target]

    df = clip_outliers_iqr(df, numeric)

    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

    preproc = build_preprocessor(numeric, categorical)
    X = df.drop(columns=[target])
    y = df[target]
    X_proc = preproc.fit_transform(X)

    ohe_cols = preproc.named_transformers_['cat']['onehot'].get_feature_names_out(categorical)
    cols = numeric + list(ohe_cols)
    df_out = pd.DataFrame(X_proc, columns=cols)
    df_out[target] = y.values

    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'Obesity_preprocessed.csv')
    df_out.to_csv(csv_path, index=False)

    # Persist artifacts
    joblib.dump(preproc, os.path.join(output_dir, 'obesity_preprocessor.joblib'))
    joblib.dump(le, os.path.join(output_dir, 'obesity_label_encoder.joblib'))

    print(f"✅ Preprocessed data saved to {csv_path}")
    print(f"🔖 Preprocessor saved to {os.path.join(output_dir, 'obesity_preprocessor.joblib')}")
    print(f"🔖 Label encoder saved to {os.path.join(output_dir, 'obesity_label_encoder.joblib')}")

    return preproc, le, df_out

# Reverse processed data back to original feature space
def reverse_preprocessing(processed_csv, preprocessor_joblib, label_enc_joblib, output_dir):
    df_proc = pd.read_csv(processed_csv)
    preproc = joblib.load(preprocessor_joblib)
    le = joblib.load(label_enc_joblib)

    target = 'Obesity'
    numeric = preproc.transformers_[0][2]
    categorical = preproc.transformers_[1][2]
    ohe_cols = preproc.named_transformers_['cat']['onehot'].get_feature_names_out(categorical)

    X_proc = df_proc[numeric + list(ohe_cols)]
    y_proc = df_proc[target]

    X_num_inv = preproc.named_transformers_['num']['scaler'].inverse_transform(X_proc[numeric])
    X_num_df = pd.DataFrame(X_num_inv, columns=numeric)

    X_cat_inv = preproc.named_transformers_['cat']['onehot'].inverse_transform(X_proc[ohe_cols])
    X_cat_df = pd.DataFrame(X_cat_inv, columns=categorical)

    y_inv = le.inverse_transform(y_proc)

    df_rev = pd.concat([X_num_df, X_cat_df], axis=1)
    df_rev[target] = y_inv

    os.makedirs(output_dir, exist_ok=True)
    
    rev_path = os.path.join(output_dir, 'Obesity_reversed.csv')
    df_rev.to_csv(rev_path, index=False)

    print(f"🔄 Reversed data saved to {rev_path}")
    
    return df_rev

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Automate Obesity Data Preprocessing & Artifact Saving')
    parser.add_argument('--input', required=True, help='Path to raw obesity.csv')
    parser.add_argument('--output', default='preprocessing', help='Output folder')
    parser.add_argument('--reverse', action='store_true', help='Also perform reversal to original space')
    args = parser.parse_args()

    preproc, le, df_out = preprocess(args.input, args.output)

    if args.reverse:
        processed_csv = os.path.join(args.output, 'Obesity_preprocessed.csv')
        preproc_joblib = os.path.join(args.output, 'obesity_preprocessor.joblib')
        label_enc_joblib = os.path.join(args.output, 'obesity_label_encoder.joblib')
        
        reverse_preprocessing(processed_csv, preproc_joblib, label_enc_joblib, args.output)

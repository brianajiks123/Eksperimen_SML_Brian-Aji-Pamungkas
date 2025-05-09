import os, pandas as pd, numpy as np, joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from category_encoders import TargetEncoder

# Membaca file CSV dan mengembalikan DataFrame
def load_data(path, encoding = 'ISO-8859-1'):
    return pd.read_csv(path, encoding=encoding)

# Menangani outlier dengan metode IQR clipping
def detect_and_clip_outliers(df, cols, factor = 1.5):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df[col] = np.clip(df[col], lower, upper)
    
    return df

# Membangun ColumnTransformer sesuai eksperimen
def build_preprocessor(ordinal_features, binary_nominal, low_nominal, high_nominal, numerical):
    return ColumnTransformer(transformers=[
        ('ord', 'passthrough', ordinal_features),
        ('bin', OrdinalEncoder(), binary_nominal),
        ('low', OneHotEncoder(handle_unknown='ignore'), low_nominal),
        ('high', TargetEncoder(), high_nominal),
        ('num', StandardScaler(), numerical)
    ])

# Pipeline preprocessing otomatis dari data mentah hingga siap model training
def preprocess(input_path, raw_cols_to_drop, ordinal_features, binary_nominal, low_nominal, high_nominal, numerical, target_col, output_dir):
    # Load dataset
    df = load_data(input_path)

    # Drop ID if exists
    df = df.drop(columns=raw_cols_to_drop)

    # Handle outliers
    df = detect_and_clip_outliers(df, numerical)

    # Split X & y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Create & fit preprocessor
    preprocessor = build_preprocessor(
        ordinal_features, binary_nominal, low_nominal, high_nominal, numerical
    )
    
    X_trans = preprocessor.fit_transform(X, y)

    # Save pipeline
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(preprocessor, os.path.join(output_dir, 'Employees Stress Level_preprocessor.joblib'))
  
    # Generate feature names after encoding
    try:
        ohe = preprocessor.named_transformers_['low']
        low_names = list(ohe.get_feature_names_out(low_nominal))
    except Exception:
        low_names = []
    
    feature_names = (
        ordinal_features + binary_nominal +
        low_names + high_nominal + numerical
    )
    
    df_out = pd.DataFrame(X_trans, columns=feature_names)
    df_out[target_col] = y.values
    out_csv = os.path.join(output_dir, 'Employees Stress Level_preprocessing.csv')
    df_out.to_csv(out_csv, index=False)
    
    print(f"Preprocessed data disimpan di: {out_csv}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Automate preprocessing Employees Stress Level'
    )
    
    parser.add_argument('--input', required=True, help='Path ke file CSV raw')
    parser.add_argument('--output_dir', default='Employees Stress Level_preprocessing',
                        help='Folder untuk menyimpan hasil')
    
    args = parser.parse_args()

    # Features
    raw_cols_to_drop = ['Employee_Id']
    ordinal_features = ['Work_Pressure', 'Manager_Support', 'Sleeping_Habit',
                        'Exercise_Habit', 'Job_Satisfaction', 'Social_Person']
    binary_nominal = ['Work_Life_Balance', 'Lives_With_Family']
    low_nominal = ['Work_From']
    high_nominal = ['Working_State']
    numerical = ['Avg_Working_Hours_Per_Day']
    target_col = 'Stress_Level'

    preprocess(
        input_path=args.input,
        raw_cols_to_drop=raw_cols_to_drop,
        ordinal_features=ordinal_features,
        binary_nominal=binary_nominal,
        low_nominal=low_nominal,
        high_nominal=high_nominal,
        numerical=numerical,
        target_col=target_col,
        output_dir=args.output_dir
    )

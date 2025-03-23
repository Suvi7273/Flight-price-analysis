import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def convert_duration(duration):
    try:
        if isinstance(duration, (int, float)):
            return duration
        parts = duration.split('h')
        hours = int(parts[0]) if 'h' in duration else 0
        minutes = int(parts[1].replace('m', '').strip()) if len(parts) > 1 and 'm' in parts[1] else 0
        return hours * 60 + minutes
    except Exception as e:
        print(f"Error in duration conversion: {duration} -> {e}")
        return np.nan

df = pd.read_csv(r'D:\Documents\sem4-docs\proj\da_flight\dataset\Clean_Dataset.csv')

print("Missing values before processing:\n", df.isnull().sum())

if 'flight' in df.columns:
    df = df.drop(columns=['flight'])

df['duration_minutes'] = df['duration'].apply(convert_duration)

invalid_durations = df[df['duration_minutes'].isna()]
if not invalid_durations.empty:
    print("Invalid durations detected:\n", invalid_durations[['duration']].head())

df.drop(columns=['duration'], inplace=True)

print("Unique values in days_left:", df['days_left'].unique())

categorical_cols = ['airline', 'source_city', 'destination_city', 'departure_time', 'arrival_time', 'class', 'stops']
numerical_cols = ['duration_minutes', 'days_left']

df[categorical_cols] = df[categorical_cols].astype(str)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))

encoded_features.columns = encoder.get_feature_names_out(categorical_cols)

df = df.drop(columns=categorical_cols).reset_index(drop=True)
df = pd.concat([df, encoded_features], axis=1)

print(f"Dataset shape after preprocessing: {df.shape}")

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "LinearRegression": LinearRegression()
}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'{name}_model.pkl')
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name} MAE: {mae:.2f}")

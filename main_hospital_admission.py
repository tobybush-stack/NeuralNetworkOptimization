import numpy as np
import pandas as pd
import math
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

max_iterations = 50000
num_calls = 20

space = [
    Integer(1, 5, name="num_layers"),
    Integer(5, 50, name="neurons_per_layer"),
    Real(1e-3, 1e-1, prior="log-uniform", name="learning_rate_init"),
    Categorical(["relu", "tanh"], name="activation"),
]

cwd = os.getcwd()
path = cwd + '\\data_admission.csv'

# Ingest data and define y variable
raw_columns = [
    "SNO",
    "MRD No.",
    "D.O.A",
    "D.O.D",
    "AGE",
    "GENDER",
    "RURAL",
    "TYPE OF ADMISSION-EMERGENCY/OPD",
    "month year",
    "DURATION OF STAY",
    "duration of intensive unit stay",
    "OUTCOME",
    "SMOKING",
    "ALCOHOL",
    "DM",
    "HTN",
    "CAD",
    "PRIOR CMP",
    "CKD",
    "HB",
    "TLC",
    "PLATELETS",
    "GLUCOSE",
    "UREA",
    "CREATININE",
    "BNP",
    "RAISED CARDIAC ENZYMES",
    "EF",
    "SEVERE ANAEMIA",
    "ANAEMIA",
    "STABLE ANGINA",
    "ACS",
    "STEMI",
    "ATYPICAL CHEST PAIN",
    "HEART FAILURE",
    "HFREF",
    "HFNEF",
    "VALVULAR",
    "CHB",
    "SSS",
    "AKI",
    "CVA INFRACT",
    "CVA BLEED",
    "AF",
    "VT",
    "PSVT",
    "CONGENITAL",
    "UTI",
    "NEURO CARDIOGENIC SYNCOPE",
    "ORTHOSTATIC",
    "INFECTIVE ENDOCARDITIS",
    "DVT",
    "CARDIOGENIC SHOCK",
    "SHOCK",
    "PULMONARY EMBOLISM",
    "CHEST INFECTION"    
]

df_raw = pd.read_csv(path, header='infer')
df = pd.DataFrame()

# transform columns

date_columns = [
    'D.O.A'
    , 'D.O.D'
]

ignore_columns = [
    'SNO'
    , 'MRD No.'
    , 'month year'
]

target_column_raw = "DURATION OF STAY"
target_column = target_column_raw.replace('.', '').replace('-', '').replace('//', '').replace(' ', '_').lower()

new_columns = []
logged_columns = []
cat_columns = df_raw.select_dtypes(include=["object", "category"]).columns
is_log_target = False

for key in df_raw.columns:
    if(key not in ignore_columns):
        new_key = key.replace('.', '').replace('-', '').replace('//', '').replace(' ', '_').lower()
        new_columns.append(new_key)
        if(key in date_columns):
            df[new_key] = (pd.to_datetime("2026-01-01") - pd.to_datetime(df_raw[key], errors='coerce')).dt.days
        elif(key not in cat_columns):
            column_data = df_raw[key]
            data_skew = abs(column_data.dropna().skew())
            if(data_skew > 1):
                df[new_key] = np.log1p(df_raw[key])
                if(key == target_column):
                    is_log_target = True
                else:
                    logged_columns.append(new_key)
            else:
                df[new_key] = df_raw[key]
        else:
            num_unique_values = df_raw[key].nunique()
            if(num_unique_values < 12):
                df_onehot = pd.get_dummies(df_raw[key], prefix=new_key, dtype=int)
                df = pd.concat([df, df_onehot], axis=1)
            else:
                 df[new_key] = df_raw[key].astype('category').cat.codes

df = df.dropna(subset=[target_column])

X_raw_df = df.drop(target_column, axis=1)

correlations = X_raw_df.corrwith(df[target_column])

threshold = 0.3
selected_features = correlations[abs(correlations) >= threshold].index.tolist()

feature_names = X_raw_df.columns.tolist()

# X_raw = X_raw_df.values
y = df[target_column].values

cols_with_nan = X_raw_df.columns[X_raw_df.isna().any()].tolist()
cols_without_nan = X_raw_df.columns[~X_raw_df.isna().any()].tolist()

impute_transformer = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("impute_cols", impute_transformer, cols_with_nan)
    ],
    remainder="passthrough" # Leaves clean columns untouched
)

ordered_feature_names = cols_with_nan + cols_without_nan

# Define objective function for optimizer
@use_named_args(space)
def objective(num_layers, neurons_per_layer, learning_rate_init, activation):
    
    hidden_layers = tuple([neurons_per_layer] * num_layers)
    
    # Optimize the Squared Error (using adam stochastic gradient descent)
    # Use Multi-layer perceptron regressor (neural network)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=learning_rate_init,
        activation=activation,
        solver="adam",
        max_iter=max_iterations,
        random_state=42,
    )

    medical_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("regressor", mlp),
    ])

    scores = cross_val_score(
        medical_pipeline, X_raw_df, y, cv=3, n_jobs=-1, scoring="neg_mean_squared_error"
    )

    mean_mse = -np.mean(scores)

    print(f"Trial -> Layers: {num_layers} | Neurons: {neurons_per_layer} | LR: {learning_rate_init:.4f} | Act: {activation} -> MSE: {mean_mse:.4f}")

    return mean_mse

print("Beginning Optimization...")
result = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=num_calls,
    n_random_starts=15,
    random_state=42,
)

num_layers = int(result.x[0])
num_neurons = int(result.x[1])
learning_rate = result.x[2]
activation_function = result.x[3]

print("\n--- Optimization Complete ---")
print(f"Optimal Number of Layers:     {num_layers} layers")
print(f"Optimal Neurons per Layer:    {num_neurons} per layer")
print(f"Optimal Learning Rate:        {learning_rate}")
print(f"Best Activation Function:     {activation_function}")
print(f"Error: {result.fun}")

print("\n--- Beginning Test ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_raw_df, y, test_size=0.30, random_state=42
)

final_preprocessor = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler())
])

X_train_scaled = final_preprocessor.fit_transform(X_train)
X_test_scaled = final_preprocessor.transform(X_test)

optimized_hidden_layers = tuple([num_neurons] * num_layers)

final_model = MLPRegressor(
    hidden_layer_sizes=optimized_hidden_layers,
    learning_rate_init=learning_rate,
    activation=activation_function,
    solver="adam",
    max_iter=max_iterations,
    random_state=42,
)

final_model.fit(X_train_scaled, y_train)

predictions = final_model.predict(X_test_scaled)
if(is_log_target):
    predictions_orig = np.expm1(predictions)
    y_test_orig = np.expm1(y_test)
else:
    predictions_orig = predictions
    y_test_orig = y_test

rmse = np.sqrt(mean_squared_error(y_test_orig, predictions_orig))
r2 = r2_score(y_test_orig, predictions_orig)

print("\n=== Final Model Performance Evaluation ===")
print(f"Average Prediction Deviation (RMSE): {rmse:.2f} days")
print(f"Model Variance Explanation Score (R²): {r2:.4f}")

print("\n--- Individual Sample Breakdown ---")
for idx in range(min(len(y_test), 100)):
    print(f"Patient {idx+1} -> Actual Duration: {y_test_orig[idx]:.2f} days | AI Predicted Duration: {predictions_orig[idx]:.2f} days")

n = len(X_raw_df)
k = len(feature_names)
r2 = r2_score(y_test_orig, predictions_orig)
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
mape = mean_absolute_percentage_error(y_test_orig, predictions_orig) * 100

print("--- Overall Metrics ---")
print(f"\nAdjusted R2:                {adjusted_r2}")
print(f"\nMAPE:                       {mape}")

# Build Plot of neural network prediction
plt.figure(figsize=(7, 6))

# Plot the patient data points
plt.scatter(y_test_orig, predictions_orig, color="blue", alpha=0.7, label="Patient Records")

# Plot the Prediction
perfect_line_range = [min(y_test_orig.min(), predictions_orig.min()), max(y_test_orig.max(), predictions_orig.max())]
plt.plot(
    perfect_line_range,
    perfect_line_range,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction Line",
)

# Format chart
plt.title("Visual Proof: Actual vs. AI Predicted Duration of Stay", fontsize=12)
plt.xlabel("Actual Duration of Stay (Days)", fontsize=10)
plt.ylabel("AI Predicted Duration of Stay (Days)", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

# Display chart
plt.show()


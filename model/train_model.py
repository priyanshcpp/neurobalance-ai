import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import joblib

# Load
df = pd.read_csv("data/dosha_data.csv")
df = pd.get_dummies(df, columns=["region", "temp_feel"], drop_first=True)

X = df.drop(columns=["vata", "pitta", "kapha"])
y_vata = df["vata"]
y_pitta = df["pitta"]
y_kapha = df["kapha"]

X_train, X_test, yv_train, yv_test = train_test_split(X, y_vata, test_size=0.2)
_, _, yp_train, yp_test = train_test_split(X, y_pitta, test_size=0.2)
_, _, yk_train, yk_test = train_test_split(X, y_kapha, test_size=0.2)

model_vata = LinearRegression().fit(X_train, yv_train)
model_pitta = LinearRegression().fit(X_train, yp_train)
model_kapha = LinearRegression().fit(X_train, yk_train)

print("✅ Models Trained")
print("Vata R2:", r2_score(yv_test, model_vata.predict(X_test)))
print("Pitta R2:", r2_score(yp_test, model_pitta.predict(X_test)))
print("Kapha R2:", r2_score(yk_test, model_kapha.predict(X_test)))

# Save models
joblib.dump(model_vata, "model/vata_model.pkl")
joblib.dump(model_pitta, "model/pitta_model.pkl")
joblib.dump(model_kapha, "model/kapha_model.pkl")
joblib.dump(X.columns.tolist(), "model/features.pkl")

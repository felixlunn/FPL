import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# ================================
# Load dataset
# ================================

file_path = ['all-euro-data-2024-2025.csv', 'all-euro-data-2023-2024.csv', 'all-euro-data-2022-2023.csv', 'all-euro-data-2021-2022.csv']
df = pd.concat([pd.read_csv(file)for file in file_path], ignore_index=True)

print("Shape:", df.shape)

# ================================
# Preprocess columns
# ================================
df = df.rename(columns={
    "FTR": "result", "FTHG": "home_goals", "FTAG": "away_goals",
    "HomeTeam": "home_team", "AwayTeam": "away_team", "Date": "date", "HTHG": "HalfTimeHomeGoals", "HTAG": "HalfTimeAwayGoals", "HS": "HomeShots",
    "AS": "AwayShots", "HST": "HomeShotsOnTarget", "AST": "AwayShotsOnTarget", "HF": "HomeFouls", "AF": "AwayFouls",
    "HC": "HomeCorners", "AC": "AwayCorners", "HY": "HomeYellowCards", "AY": "AwayYellowCards", "HR": "HomeRedCards", "AR": "AwayRedCards"
})
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

label_map = {"H": "Home", "A": "Away", "D": "Draw"}
df["result_label"] = df["result"].map(label_map)

# ================================
# Add bookmaker implied probabilities (Bet365 odds)
# ================================
for col in ["B365H", "B365D", "B365A"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["book_home"] = 1 / df["B365H"]
df["book_draw"] = 1 / df["B365D"]
df["book_away"] = 1 / df["B365A"]

total = df[["book_home", "book_draw", "book_away"]].sum(axis=1)
df["book_home"] /= total
df["book_draw"] /= total
df["book_away"] /= total

# ================================
# Feature engineering: simple Elo ratings
# ================================
teams = pd.concat([df["home_team"], df["away_team"]]).unique()
elo = {t: 1500 for t in teams}
k = 20
elo_history = []

for _, row in df.iterrows():
    ht, at = row["home_team"], row["away_team"]
    hg, ag = row["home_goals"], row["away_goals"]
    exp_h = 1 / (1 + 10 ** ((elo[at] - elo[ht]) / 400))
    exp_a = 1 - exp_h
    if hg > ag:
        s_h, s_a = 1, 0
    elif hg < ag:
        s_h, s_a = 0, 1
    else:
        s_h, s_a = 0.5, 0.5
    elo[ht] += k * (s_h - exp_h)
    elo[at] += k * (s_a - exp_a)
    elo_history.append({"home_elo": elo[ht], "away_elo": elo[at]})

elo_df = pd.DataFrame(elo_history)
df = pd.concat([df.reset_index(drop=True), elo_df], axis=1)
df["elo_diff"] = df["home_elo"] - df["away_elo"]

# ================================
# Train/test split (chronological 80/20)
# ================================
df = df.sort_values("date").reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

features = ["home_team", "away_team", "home_elo", "away_elo", "elo_diff",
            "book_home", "book_draw", "book_away","HalfTimeHomeGoals", "HalfTimeAwayGoals", "HomeShots", "AwayShots",
            "HomeShotsOnTarget", "AwayShotsOnTarget", "HomeFouls", "AwayFouls", "HomeCorners", "AwayCorners", "HomeYellowCards",
            "HomeRedCards", "AwayYellowCards", "AwayRedCards"]
X_train = train_df[features]
y_train = train_df["result_label"]
X_test = test_df[features]
y_test = test_df["result_label"]

# ================================
# Preprocessing + Calibrated Model
# ================================
numeric_features = ["home_elo", "away_elo", "elo_diff", "book_home", "book_draw", "book_away"]
cat_features = ["home_team", "away_team"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", cat_transformer, cat_features)
    ]
)

base_clf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced_subsample"
)

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", CalibratedClassifierCV(base_clf, method="isotonic", cv=5))
])

pipe.fit(X_train, y_train)

# ================================
# Evaluation
# ================================
probs = pipe.predict_proba(X_test)
preds = pipe.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("Log-loss:", log_loss(y_test, probs))
print("Classification report:\n", classification_report(y_test, preds))
print("Confusion matrix:\n", confusion_matrix(y_test, preds))

# Calibration curve for Home
prob_true, prob_pred = calibration_curve((y_test == "Home"), probs[:, 0], n_bins=10)
plt.plot(prob_pred, prob_true, marker="o", label="Home (calibrated)")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Predicted probability")
plt.ylabel("True frequency")
plt.title("Calibration curve")
plt.legend()
plt.show()

# ================================
# Feature importance (from base model, not calibrated wrapper)
# ================================
base_clf.fit(preprocessor.fit_transform(X_train), y_train)

# Get feature names from preprocessing
ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_names = ohe.get_feature_names_out(cat_features)
all_features = numeric_features + list(cat_names)

importances = base_clf.feature_importances_
indices = np.argsort(importances)[-15:]  # top 15

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [all_features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.title("Top Feature Importances")
plt.show()

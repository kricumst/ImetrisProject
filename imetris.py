from dotenv import load_dotenv
from os import getenv
from mssql_python import connect
import pandas as pd
import matplotlib as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)

load_dotenv()

def create_df(table_name):

    query = f"SELECT * FROM [{table_name}]"
    conn_str = getenv("sql_conn_str")
    with connect(conn_str) as conn:
      df = pd.read_sql(query, conn)
    return df

train = create_df("provider-train")
train_out = create_df("outpatient-train")
train_in = create_df("inpatient-train")
train_bene = create_df("beneficiary-train")

test = create_df("provider-test")
test_out = create_df("outpatient-test")
test_in = create_df("inpatient-test")
test_bene = create_df("beneficiary-test")

print("Train providers:", train.shape)
print("Train outpatient:", train_out.shape)
print("Train inpatient:", train_in.shape)
print("Train beneficiaries:", train_bene.shape)
print()
print("Test providers:", test.shape)
print("Test outpatient:", test_out.shape)
print("Test inpatient:", test_in.shape)
print("Test beneficiaries:", test_bene.shape)

# 2. Basic label EDA

print(train["PotentialFraud"].value_counts())
print("Fraud rate:", (train["PotentialFraud"] == "Yes").mean())

# Parse claim dates for inpatient/outpatient so we can do LOS and time-based features

for df in [train_out, train_in, test_out, test_in]:
    df["ClaimStartDt"] = pd.to_datetime(df["ClaimStartDt"])
    df["ClaimEndDt"] = pd.to_datetime(df["ClaimEndDt"])

for df in [train_in, test_in]:
    df["AdmissionDt"] = pd.to_datetime(df["AdmissionDt"])
    df["DischargeDt"] = pd.to_datetime(df["DischargeDt"])
    df["LOS_days"] = (df["DischargeDt"] - df["AdmissionDt"]).dt.days

print("Outpatient date range:", train_out["ClaimStartDt"].min(), "to", train_out["ClaimEndDt"].max())
print("Inpatient date range:", train_in["ClaimStartDt"].min(), "to", train_in["ClaimEndDt"].max())

# Simple distributions

train_out["InscClaimAmtReimbursed"].hist(bins=50)
plt.title("Outpatient Reimbursement")
plt.xlabel("Amount")
plt.ylabel("Count")
plt.show()

train_in["LOS_days"].hist(bins=40)
plt.title("Inpatient Length of Stay (days)")
plt.xlabel("LOS_days")
plt.ylabel("Count")
plt.show()

# 3. Beneficiary feature cleaning: convert chronic conditions 1/2 -> 0/1

chronic_cols = [c for c in train_bene.columns if c.startswith("ChronicCond_")]

def convert_chronic(df):
    out = df.copy()
    for c in chronic_cols:
        # 1 = no condition, 2 = condition; make new binary col
        out[c + "_bin"] = (out[c] == 2).astype(int)
    return out

train_bene_clean = convert_chronic(train_bene)
test_bene_clean = convert_chronic(test_bene)

print("Example chronic counts (Diabetes):")
print(train_bene["ChronicCond_Diabetes"].value_counts())
print(train_bene_clean["ChronicCond_Diabetes_bin"].value_counts())

# 4. Provider-level aggregation for train

# Outpatient aggregation
prov_out = train_out.groupby("Provider").agg(
    n_out_claims=("ClaimID", "nunique"),
    out_total_reimb=("InscClaimAmtReimbursed", "sum"),
    out_mean_reimb=("InscClaimAmtReimbursed", "mean"),
    out_max_reimb=("InscClaimAmtReimbursed", "max"),
    out_std_reimb=("InscClaimAmtReimbursed", "std"),
)

# Inpatient aggregation
prov_in = train_in.groupby("Provider").agg(
    n_in_claims=("ClaimID", "nunique"),
    in_total_reimb=("InscClaimAmtReimbursed", "sum"),
    in_mean_reimb=("InscClaimAmtReimbursed", "mean"),
    in_max_reimb=("InscClaimAmtReimbursed", "max"),
    in_std_reimb=("InscClaimAmtReimbursed", "std"),
    in_mean_LOS=("LOS_days", "mean"),
    in_max_LOS=("LOS_days", "max"),
)

# Merge with labels
provider_train_features = (
    train[["Provider", "PotentialFraud"]]
    .merge(prov_out, on="Provider", how="left")
    .merge(prov_in, on="Provider", how="left")
)

# Fill missing (providers with no in/out claims)
num_cols = provider_train_features.columns.difference(["Provider", "PotentialFraud"])
provider_train_features[num_cols] = provider_train_features[num_cols].fillna(0)

provider_train_features.head()

provider_train_features.groupby("PotentialFraud")[
    ["n_out_claims", "n_in_claims", "out_total_reimb", "in_total_reimb", "in_mean_LOS"]
].mean()


# 5. Provider-level aggregation for test

prov_out_test = test_out.groupby("Provider").agg(
    n_out_claims=("ClaimID", "nunique"),
    out_total_reimb=("InscClaimAmtReimbursed", "sum"),
    out_mean_reimb=("InscClaimAmtReimbursed", "mean"),
    out_max_reimb=("InscClaimAmtReimbursed", "max"),
    out_std_reimb=("InscClaimAmtReimbursed", "std"),
)

prov_in_test = test_in.groupby("Provider").agg(
    n_in_claims=("ClaimID", "nunique"),
    in_total_reimb=("InscClaimAmtReimbursed", "sum"),
    in_mean_reimb=("InscClaimAmtReimbursed", "mean"),
    in_max_reimb=("InscClaimAmtReimbursed", "max"),
    in_std_reimb=("InscClaimAmtReimbursed", "std"),
    in_mean_LOS=("LOS_days", "mean"),
    in_max_LOS=("LOS_days", "max"),
)

provider_test_features = (
    test.merge(prov_out_test, on="Provider", how="left")
        .merge(prov_in_test, on="Provider", how="left")
)

# ===================== SAFE FEATURE ENGINEERING BLOCK (FINAL) ===================== #

# -------- 0) PRE-CLEANUP (REMOVE ANY OLD DUPLICATES IF RERUN) -------- #

drop_cols = [c for c in provider_train_features.columns if c.endswith("_x") or c.endswith("_y") or c in [
    "unique_diag_out", "unique_diag_in", "unique_diag_total",
    "physician_missing_out", "physician_missing_in"
]]

if drop_cols:
    print("Dropping stale columns:", drop_cols)
    provider_train_features.drop(columns=drop_cols, inplace=True, errors="ignore")


# ===================== A) BASIC COUNT & RATIO FEATURES ===================== #

provider_train_features["total_claims"] = (
    provider_train_features["n_out_claims"] + provider_train_features["n_in_claims"]
)

provider_train_features["claims_ratio_in_to_out"] = (
    provider_train_features["n_in_claims"] / (provider_train_features["n_out_claims"] + 1)
)

provider_train_features["out_reimb_per_claim"] = (
    provider_train_features["out_total_reimb"] / (provider_train_features["n_out_claims"] + 1)
)

provider_train_features["in_reimb_per_claim"] = (
    provider_train_features["in_total_reimb"] / (provider_train_features["n_in_claims"] + 1)
)

provider_train_features["reimb_ratio_in_to_out"] = (
    provider_train_features["in_total_reimb"] / (provider_train_features["out_total_reimb"] + 1)
)


# ===================== B) PERCENTILE-BASED RISK FLAGS ===================== #

in_mean_reimb_90 = provider_train_features["in_mean_reimb"].quantile(0.90)
los_mean_90 = provider_train_features["in_mean_LOS"].quantile(0.90)

provider_train_features["high_cost_provider"] = (
    (provider_train_features["in_mean_reimb"] > in_mean_reimb_90).astype(int)
)

provider_train_features["high_LOS_provider"] = (
    (provider_train_features["in_mean_LOS"] > los_mean_90).astype(int)
)


# ===================== C) DIAGNOSIS UNIQUENESS ===================== #

out_diag_cols = [c for c in train_out.columns if c.startswith("ClmDiagnosisCode")]
in_diag_cols = [c for c in train_in.columns if c.startswith("ClmDiagnosisCode")]

prov_diag_out = (
    train_out.groupby("Provider")[out_diag_cols]
    .apply(lambda df: df.stack().nunique())
    .rename("unique_diag_out")
)

prov_diag_in = (
    train_in.groupby("Provider")[in_diag_cols]
    .apply(lambda df: df.stack().nunique())
    .rename("unique_diag_in")
)

provider_train_features = (
    provider_train_features
    .merge(prov_diag_out, on="Provider", how="left")
    .merge(prov_diag_in, on="Provider", how="left")
)

provider_train_features["unique_diag_out"] = provider_train_features["unique_diag_out"].fillna(0)
provider_train_features["unique_diag_in"] = provider_train_features["unique_diag_in"].fillna(0)
provider_train_features["unique_diag_total"] = (
    provider_train_features["unique_diag_out"] + provider_train_features["unique_diag_in"]
)


# ===================== D) PHYSICIAN MISSINGNESS ===================== #

phys_cols = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]

phys_missing_out = (
    train_out[["Provider"] + phys_cols]
    .groupby("Provider")[phys_cols]
    .apply(lambda x: x.isna().mean().mean())
    .rename("physician_missing_out")
)

phys_missing_in = (
    train_in[["Provider"] + phys_cols]
    .groupby("Provider")[phys_cols]
    .apply(lambda x: x.isna().mean().mean())
    .rename("physician_missing_in")
)

provider_train_features = (
    provider_train_features
    .merge(phys_missing_out, on="Provider", how="left")
    .merge(phys_missing_in, on="Provider", how="left")
)

provider_train_features["physician_missing_out"] = provider_train_features["physician_missing_out"].fillna(1)
provider_train_features["physician_missing_in"] = provider_train_features["physician_missing_in"].fillna(1)


# --------  FINAL POST-CLEANUP (REMOVE ANY MERGE SUFFIXES) -------- #

drop_cols = [c for c in provider_train_features.columns if c.endswith("_x") or c.endswith("_y")]
if drop_cols:
    print("Removing merge suffix columns:", drop_cols)
    provider_train_features.drop(columns=drop_cols, inplace=True, errors="ignore")


# ===================== END CHECK ===================== #
print("\n🎉 Feature engineering complete! 🎉")
print(provider_train_features.head())
print(provider_train_features.shape)
print(provider_train_features.isna().sum())



num_cols_test = provider_test_features.columns.difference(["Provider"])
provider_test_features[num_cols_test] = provider_test_features[num_cols_test].fillna(0)

provider_test_features.head()

# ===================== ADDITIONAL FEATURE ENGINEERING ===================== #
# ============ PROCEDURE CODE UNIQUENESS + BENEFICIARY RISK SCORES ============ #

print("🔧 Starting advanced feature engineering (procedures + case-mix)...")

# --------------------------------------------------------------------------- #
# 1️⃣ PROCEDURE CODE UNIQUENESS / DIVERSITY
# --------------------------------------------------------------------------- #

proc_cols_out = [c for c in train_out.columns if c.startswith("ClmProcedureCode")]
proc_cols_in  = [c for c in train_in.columns  if c.startswith("ClmProcedureCode")]

# Compute unique procedure codes per provider (stack + nunique)
prov_proc_out = (
    train_out.groupby("Provider")[proc_cols_out]
    .apply(lambda df: df.stack().nunique())
    .rename("unique_proc_out")
)

prov_proc_in = (
    train_in.groupby("Provider")[proc_cols_in]
    .apply(lambda df: df.stack().nunique())
    .rename("unique_proc_in")
)

# Merge into provider table
provider_train_features = (
    provider_train_features
    .merge(prov_proc_out, on="Provider", how="left")
    .merge(prov_proc_in, on="Provider", how="left")
)

# Fill NA
provider_train_features["unique_proc_out"] = provider_train_features["unique_proc_out"].fillna(0)
provider_train_features["unique_proc_in"]  = provider_train_features["unique_proc_in"].fillna(0)

# Total procedure diversity
provider_train_features["unique_proc_total"] = (
    provider_train_features["unique_proc_out"] + provider_train_features["unique_proc_in"]
)

# --------------------------------------------------------------------------- #
# 2️⃣ BENEFICIARY RISK SEVERITY SCORE (CHRONIC CONDITION INDEX)
# --------------------------------------------------------------------------- #

# Create chronic condition binary columns if not already present
chronic_cols_bin = [c for c in train_bene.columns if c.startswith("ChronicCond_")]
if not any("_bin" in c for c in train_bene.columns):
    train_bene_clean = train_bene.copy()
    for c in chronic_cols_bin:
        train_bene_clean[c + "_bin"] = (train_bene_clean[c] == 2).astype(int)
else:
    train_bene_clean = train_bene.copy()

# Compute an overall chronic severity score per beneficiary
chronic_bin_cols = [c for c in train_bene_clean.columns if c.endswith("_bin")]
train_bene_clean["chronic_severity_score"] = train_bene_clean[chronic_bin_cols].sum(axis=1)

# Merge beneficiary severity into claims tables for provider-level aggregation
train_out_bene = train_out.merge(train_bene_clean[["BeneID", "chronic_severity_score"]], on="BeneID", how="left")
train_in_bene  = train_in.merge(train_bene_clean[["BeneID", "chronic_severity_score"]], on="BeneID", how="left")

# Provider-level severity (mean + max + variance)
provider_case_mix = (
    pd.concat([
        train_out_bene[["Provider", "chronic_severity_score"]],
        train_in_bene[["Provider", "chronic_severity_score"]]
    ])
    .groupby("Provider")["chronic_severity_score"]
    .agg(["mean", "max", "std"])
    .rename(columns={"mean":"bene_risk_mean", "max":"bene_risk_max", "std":"bene_risk_std"})
)

provider_case_mix["bene_risk_std"] = provider_case_mix["bene_risk_std"].fillna(0)

# Merge into main table
provider_train_features = provider_train_features.merge(provider_case_mix, on="Provider", how="left")

# Fill NA (providers with no beneficiaries)
for col in ["bene_risk_mean", "bene_risk_max", "bene_risk_std"]:
    provider_train_features[col] = provider_train_features[col].fillna(0)

# --------------------------------------------------------------------------- #
# 3️⃣ OPTIONAL — PROCEDURE / DIAGNOSIS COMPLEXITY RATIOS
# --------------------------------------------------------------------------- #

provider_train_features["diag_per_claim"] = (
    provider_train_features["unique_diag_total"] / (provider_train_features["total_claims"] + 1)
)

provider_train_features["proc_per_claim"] = (
    provider_train_features["unique_proc_total"] / (provider_train_features["total_claims"] + 1)
)

provider_train_features["proc_to_diag_ratio"] = (
    provider_train_features["unique_proc_total"] / (provider_train_features["unique_diag_total"] + 1)
)

# --------------------------------------------------------------------------- #
# DONE
# --------------------------------------------------------------------------- #

print("Advanced feature engineering completed!")
print(provider_train_features.head())
print(provider_train_features.shape)
print(provider_train_features.isna().sum())

# ===================== 6. MODELING SETUP & TRAINING ===================== #

# ---------- 6.1 Build X, y ---------- #

# Encode target: Fraud = 1, No Fraud = 0
provider_train_features["FraudFlag"] = (provider_train_features["PotentialFraud"] == "Yes").astype(int)

# Select feature columns: all numeric except ID/label
exclude_cols = ["Provider", "PotentialFraud", "FraudFlag"]
feature_cols = [c for c in provider_train_features.columns if c not in exclude_cols]

X = provider_train_features[feature_cols].values
y = provider_train_features["FraudFlag"].values

print("X shape:", X.shape)
print("y positive rate (fraud):", y.mean())

# ---------- 6.2 Train/validation split ---------- #

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0], "Val size:", X_val.shape[0])

# ---------- 6.3 Logistic Regression (class-weighted) ---------- #

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

log_reg = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
    n_jobs=-1
)

log_reg.fit(X_train_scaled, y_train)

# Probabilities for fraud class
y_val_proba_lr = log_reg.predict_proba(X_val_scaled)[:, 1]

auc_lr = roc_auc_score(y_val, y_val_proba_lr)
print("\n🔹 Logistic Regression ROC-AUC:", round(auc_lr, 4))


# ---------- 6.4 Random Forest (stronger non-linear model) ---------- #

rf = RandomForestClassifier(
    n_estimators=350,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight={0: 1, 1: 4},  # upweight fraud class
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_val_proba_rf = rf.predict_proba(X_val)[:, 1]

auc_rf = roc_auc_score(y_val, y_val_proba_rf)
print("🔹 Random Forest ROC-AUC:", round(auc_rf, 4))


# ---------- 6.5 Threshold sweep (focus on fraud recall but not insane FP) ---------- #

def sweep_thresholds(y_true, y_proba, model_name="model"):
    print(f"\n===== Threshold sweep for {model_name} =====")
    best_f2 = 0
    best_thr = 0.5
    records = []

    for thr in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_proba >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        )

        # F2-score weights recall higher than precision
        beta = 2
        if precision + recall == 0:
            f2 = 0
        else:
            f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

        records.append((thr, precision, recall, f1, f2))

        # You can also require a minimum precision floor if you want, e.g. precision > 0.15
        if f2 > best_f2:
            best_f2 = f2
            best_thr = thr

    # Print summary of sweep
    df_thr = pd.DataFrame(records, columns=["threshold", "precision", "recall", "f1", "f2"])
    print(df_thr)

    print(f"\n⭐ Best threshold for {model_name} by F2:", round(best_thr, 2))
    print(df_thr.loc[df_thr["threshold"] == best_thr])

    return best_thr, df_thr


best_thr_lr, thr_table_lr = sweep_thresholds(y_val, y_val_proba_lr, model_name="Logistic Regression")
best_thr_rf, thr_table_rf = sweep_thresholds(y_val, y_val_proba_rf, model_name="Random Forest")


# ---------- 6.6 Final evaluation at chosen threshold (Random Forest) ---------- #

chosen_thr = best_thr_rf  # you can manually set e.g. 0.35 if you want more precision
print("\n💡 Using threshold =", round(chosen_thr, 2), "for Random Forest")

y_val_pred_rf = (y_val_proba_rf >= chosen_thr).astype(int)

print("\nConfusion Matrix (RF):")
print(confusion_matrix(y_val, y_val_pred_rf))

print("\nClassification Report (RF):")
print(classification_report(y_val, y_val_pred_rf, digits=4))

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "log_coef": log_reg.coef_[0],
    "rf_importance": rf.feature_importances_
})

# Sort by Random Forest importance
importance_sorted = importance_df.sort_values("rf_importance", ascending=False)
print("\nTop 15 features by Random Forest importance:")
print(importance_sorted.head(15))

# Sort by absolute Logistic regression coefficients
importance_sorted_lr = importance_df.reindex(
    importance_df.log_coef.abs().sort_values(ascending=False).index
)
print("\nTop 15 features by Logistic Regression coefficient magnitude:")
print(importance_sorted_lr.head(15))

# ===================== FINAL MODEL SELECTION & SUMMARY ===================== #

print("\n================ FINAL MODEL EVALUATION SUMMARY ================\n")

# 1️⃣ Compare ROC-AUC
print(f"Logistic Regression ROC-AUC: {auc_lr:.4f}")
print(f"Random Forest ROC-AUC:       {auc_rf:.4f}\n")

best_model_name = "Logistic Regression" if auc_lr >= auc_rf else "Random Forest"
best_model = log_reg if best_model_name == "Logistic Regression" else rf
best_proba = y_val_proba_lr if best_model_name == "Logistic Regression" else y_val_proba_rf
best_threshold = best_thr_lr if best_model_name == "Logistic Regression" else best_thr_rf

print(f"📌 Selected Model: **{best_model_name}**")
print(f"📌 Decision Threshold Chosen (from F2 optimization): {round(best_threshold, 3)}\n")

# 2️⃣ Final classification using chosen model + threshold
y_val_pred = (best_proba >= best_threshold).astype(int)

# Display results
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, digits=4))

# 3️⃣ Business Interpretation
fraud_detect_rate = recall_score = precision_recall_fscore_support(
    y_val, y_val_pred, average="binary", zero_division=0
)[1]

precision_score = precision_recall_fscore_support(
    y_val, y_val_pred, average="binary", zero_division=0
)[0]

print("--------------------------------------------------")
print("📈 Model Business Interpretation")
print(f"✔ Probability-based fraud detector using {best_model_name}")
print(f"✔ Detects approximately {fraud_detect_rate*100:.1f}% of fraud cases (recall)")
print(f"✔ About {precision_score*100:.1f}% of flagged providers are likely fraud (precision)")
print(f"✔ ROC-AUC score of {auc_rf if best_model_name=='Random Forest' else auc_lr:.3f} shows strong class separation")
print("--------------------------------------------------")

if fraud_detect_rate >= 0.80 and precision_score >= 0.25:
    print("✅ Conclusion: Model is suitable for **real-world SIU pre-screening** (not final guilt).")
    print("   - Use for **risk-based triage** or **tiered audit review**.")
    print("   - Not intended as **automatic denial engine**.\n")
else:
    print("⚠️ Conclusion: Model needs **further tuning** before business deployment.\n")

print("=================================================================\n")


# ===================== STEP B: MODEL EXPLAINABILITY ===================== #

print("\n📌 Initializing SHAP explainability...")

# Use model chosen earlier (best_model)
model_for_shap = best_model

# Train SHAP explainer (tree or linear mode auto-detect)
explainer = shap.Explainer(model_for_shap, X_train_scaled if best_model_name=="Logistic Regression" else X_train)
shap_values = explainer(X_val_scaled if best_model_name=="Logistic Regression" else X_val)

# 1️⃣ Global importance plot
print("\n📌 Displaying global feature importance...")
shap.plots.bar(shap_values, max_display=15)

# 2️⃣ Beeswarm: shows feature interaction + direction
print("\n📌 Displaying beeswarm feature effects...")
shap.plots.beeswarm(shap_values, max_display=20)

# 3️⃣ Single provider local explanation example
idx = np.argmax(y_val_proba_lr if best_model_name=="Logistic Regression" else y_val_proba_rf)
print("\n📌 SHAP local explanation for the most suspicious validation provider:")
shap.plots.waterfall(shap_values[idx], max_display=12)

print("\n🎯 SHAP explainability completed.")


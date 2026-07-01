from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA = Path("data")
MODEL_DIR = Path("models")
IN = DATA / "racer_rank_dataset.csv"
OUT_PRED = DATA / "finish_class_predictions_valid.csv"
OUT_MODEL = MODEL_DIR / "finish_class_random_forest.joblib"

DROP_COLS = {
    "race_id", "race_date", "race_start_date", "date", "name",
    "finish_position", "finish_class",
    "margin", "agari", "decision", "sb", "comment",
    "combination", "payout", "popularity",
    "ticket_1st", "ticket_2nd", "ticket_3rd", "ticket_top2", "ticket_top3",
    "official_is_1st", "official_top2", "official_top3",
    "ticket_1st_car", "ticket_2nd_car", "ticket_3rd_car",
    "has_numeric_finish", "is_finished",
    "finish_status", "finish_status_detail", "same_finish_count", "raw_finish_position",
    "exception_name_from_html", "exception_margin", "exception_agari",
    "exception_decision", "exception_sb", "exception_comment",
    "is_dead_heat", "is_absent", "is_dnf", "is_crash", "is_disqualified",
    "is_accident_finish", "is_unknown_exception",
}

POST_RACE_PATTERNS = [
    "finish", "ticket", "official", "payout", "popularity", "combination",
    "result", "着", "順位", "払戻", "決まり", "decision", "margin",
    "agari", "comment", "exception", "disqualified", "dead_heat",
    "dnf", "crash",
]

ALLOW_FEATURE_COLS = {
    "score_rank_in_race",
    "win_rate_rank_in_race",
    "place_rate_rank_in_race",
}

def is_post_race_col(col):
    if col in ALLOW_FEATURE_COLS:
        return False
    low = col.lower()
    return any(p.lower() in low for p in POST_RACE_PATTERNS)

def make_x(data, columns=None):
    drop_cols = set(DROP_COLS)
    drop_cols.update(c for c in data.columns if is_post_race_col(c))
    x = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore").copy()
    cats = [c for c in x.columns if x[c].dtype == "object"]
    x = pd.get_dummies(x, columns=cats, dummy_na=True)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0)
    return x

df = pd.read_csv(IN, dtype={"race_id": str}, low_memory=False)
df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

df = df[df["finish_class"] != "absent"].copy()

train = df[df["race_date"] <= "2025-12-31"].copy()
valid = df[(df["race_date"] >= "2026-01-01") & (df["race_date"] <= "2026-06-26")].copy()

X_train = make_x(train)
X_valid = make_x(valid, list(X_train.columns))
y_train = train["finish_class"]
y_valid = valid["finish_class"]

print(f"train rows: {len(train):,}")
print(f"valid rows: {len(valid):,}")
print(f"features: {len(X_train.columns):,}")

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=20,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

pred = model.predict(X_valid)
proba = model.predict_proba(X_valid)

print(f"accuracy: {accuracy_score(y_valid, pred):.4f}")
print("")
print(classification_report(y_valid, pred, digits=4))

out = valid[["race_id", "race_date", "car_no", "name", "finish_class"]].copy()
for i, cls in enumerate(model.classes_):
    out[f"p_{cls}"] = proba[:, i]

for col in ["p_1st", "p_2nd", "p_3rd", "p_4plus", "p_exception"]:
    if col not in out.columns:
        out[col] = 0.0

out["p_top2"] = out["p_1st"] + out["p_2nd"]
out["p_top3"] = out["p_1st"] + out["p_2nd"] + out["p_3rd"]

OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

out.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
joblib.dump(
    {"model": model, "columns": list(X_train.columns), "classes": list(model.classes_)},
    OUT_MODEL,
)

print(f"saved: {OUT_PRED}")
print(f"saved: {OUT_MODEL}")
print("")
print(out.sort_values(["race_date", "race_id", "p_top3"], ascending=[True, True, False]).head(30).to_string(index=False))

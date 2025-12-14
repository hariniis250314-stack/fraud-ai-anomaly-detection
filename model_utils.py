 import numpy as np

def feature_engineering(df):
    engineered = df.copy()

    for col in engineered.columns:
        engineered[f"{col}_log"] = np.log1p(engineered[col])

    z_scores = abs((engineered - engineered.mean()) / engineered.std())
    engineered["max_z_score"] = z_scores.max(axis=1)
    engineered["row_variance"] = engineered.var(axis=1)
    engineered["transaction_intensity"] = engineered.sum(axis=1)

    return engineered

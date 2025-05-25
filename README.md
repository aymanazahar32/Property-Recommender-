# Property-Recommender-

This project builds a robust comparable property (comp) selection system from a messy real estate dataset of appraisals. The goal is to train a learning-to-rank model (e.g., LambdaMART) that identifies which candidate properties are most comparable to a given subject property.

**Features**
Flexible JSON Loader: Handles multiple formats including:

JSON lists ([{}])

ND-JSON (newline-delimited)

Back-to-back multi-line JSON objects

Robust Parsing: Falls back to json5 for malformed JSON with unquoted keys.

Flattening Logic: Converts nested dictionaries into flat, prefixed columns.

Missing Value Preservation: All missing values are preserved as NaN, making the data suitable for models like LightGBM or XGBoost.

**Pipeline Overview**

**Load and Parse Data**

Uses a custom streaming parser with bracket-counting to handle malformed or inconsistent formats.

Each appraisal expands into one row per subject-candidate pair.

**Data Output**

Final output is a DataFrame with shape approximately (9800, 120), ready for feature engineering and modeling.

**Next Steps**

Engineer delta features between subject and candidate properties.

Train a LambdaMART ranker for comp selection.

Add SHAP-based model interpretation and feature importance analysis.

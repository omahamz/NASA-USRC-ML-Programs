import json
from src.ml_models.evaluate import comparison_bar_chart

# Load saved metrics
with open("models/mlp_comparison_metrics.json") as f:
    mlp_m = json.load(f)
with open("models/gp_comparison_metrics.json") as f:
    gp_m = json.load(f)

# Combine all variants
all_metrics = {**mlp_m, **gp_m}
comparison_bar_chart(all_metrics, title="All Models — R² on HF Test Set")
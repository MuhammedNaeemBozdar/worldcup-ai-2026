"""
train.py
Standalone script to train and save all ML models.
Run: python train.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import train_and_save_models

print("=" * 60)
print("FIFA World Cup 2026 AI — Model Training")
print("=" * 60)

models, scaler, feat_cols, metrics = train_and_save_models()

print("\n📊 Model Performance:")
for name, m in metrics.items():
    print(f"  {name:25s} Accuracy: {m['accuracy']:.3f}  Precision: {m['precision']:.3f}  Recall: {m['recall']:.3f}")

print(f"\n✅ Models saved to models/trained_models/")
print("   Run `streamlit run app.py` to launch the dashboard.")

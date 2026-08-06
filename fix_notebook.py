import json

with open('road_accident_dl.ipynb') as f:
    nb = json.load(f)

# Update Section 20 code cell (cell 206)
cell = nb['cells'][206]
cell['source'] = [
    "# 1. Full Dataset Predictions (using balanced models)\n",
    "probs_xgb_full = xgb_balanced.predict_proba(X_bal_imputed)\n",
    "probs_lgb_full = lgb_balanced.predict_proba(X_bal_imputed)\n",
    "\n",
    "mlp_balanced.eval()\n",
    "with torch.no_grad():\n",
    "    inputs_full = torch.tensor(scaler_bal.transform(X_bal_imputed.values), dtype=torch.float32)\n",
    "    probs_mlp_full = torch.softmax(mlp_balanced(inputs_full), dim=1).numpy()\n",
    "\n",
    "# 2. Ensemble Predictions\n",
    "probs_ensemble_full = 0.35 * probs_xgb_full + 0.45 * probs_lgb_full + 0.20 * probs_mlp_full\n",
    "preds_ensemble_full = np.argmax(probs_ensemble_full, axis=1)\n",
    "\n",
    "# 3. Evaluation\n",
    "print('Hybrid Blending Ensemble Performance on BALANCED Complete Dataset:')\n",
    "print(f'Overall Accuracy: {accuracy_score(le_bal.transform(df_fe[\"Accident_severity\"]), preds_ensemble_full)*100:.2f}%')\n",
    "print(classification_report(le_bal.transform(df_fe['Accident_severity']), preds_ensemble_full, target_names=le_bal.classes_, zero_division=0))\n",
    "\n",
    "# 4. Confusion Matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import matplotlib.pyplot as plt\n",
    "cm = confusion_matrix(le_bal.transform(df_fe['Accident_severity']), preds_ensemble_full)\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
    "            xticklabels=le_bal.classes_, yticklabels=le_bal.classes_)\n",
    "plt.title('Hybrid Ensemble - Balanced Complete Dataset')\n",
    "plt.ylabel('Actual')\n",
    "plt.xlabel('Predicted')\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

# Update Section 20 markdown
nb['cells'][205]['source'] = [
    "# Section 20: Evaluation on the Balanced Complete Dataset\n",
    "\n",
    "We evaluate the retrained hybrid ensemble on the complete balanced dataset to report final metrics."
]

# Update Section 21 markdown
nb['cells'][207]['source'] = [
    "# Section 21: Final Model Comparison (Balanced Models)\n",
    "\n",
    "We compare the balanced models on the test set."
]

# Update Section 21 code
nb['cells'][208]['source'] = [
    "from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score\n",
    "\n",
    "model_names = [\n",
    "    'XGBoost (Balanced)',\n",
    "    'LightGBM (Balanced)',\n",
    "    'MLP (Balanced)',\n",
    "    'Hybrid Ensemble (Balanced)'\n",
    "]\n",
    "\n",
    "# Get predictions from each model\n",
    "y_pred_xgb_bal_test = xgb_balanced.predict(X_test_bal)\n",
    "y_pred_lgb_bal_test = lgb_balanced.predict(X_test_bal)\n",
    "\n",
    "mlp_balanced.eval()\n",
    "with torch.no_grad():\n",
    "    y_pred_mlp_bal_test = mlp_balanced(torch.tensor(X_test_bal_scaled, dtype=torch.float32)).argmax(dim=1).numpy()\n",
    "\n",
    "y_pred_ensemble_bal_test = np.argmax(0.35 * probs_xgb_bal + 0.45 * probs_lgb_bal + 0.20 * probs_mlp_bal, axis=1)\n",
    "\n",
    "y_preds = [y_pred_xgb_bal_test, y_pred_lgb_bal_test, y_pred_mlp_bal_test, y_pred_ensemble_bal_test]\n",
    "\n",
    "# Metrics\n",
    "accuracy_scores = [accuracy_score(y_test_bal, p) for p in y_preds]\n",
    "precision_scores = [precision_score(y_test_bal, p, average='weighted') for p in y_preds]\n",
    "recall_scores = [recall_score(y_test_bal, p, average='weighted') for p in y_preds]\n",
    "f1_scores = [f1_score(y_test_bal, p, average='weighted') for p in y_preds]\n",
    "\n",
    "fatal_recalls = [recall_score(y_test_bal, p, average=None)[0] for p in y_preds]\n",
    "serious_recalls = [recall_score(y_test_bal, p, average=None)[1] for p in y_preds]\n",
    "\n",
    "# Display results\n",
    "results_df = pd.DataFrame({\n",
    "    'Model': model_names,\n",
    "    'Accuracy': [f'{s*100:.2f}%' for s in accuracy_scores],\n",
    "    'Precision (W)': [f'{s*100:.2f}%' for s in precision_scores],\n",
    "    'Recall (W)': [f'{s*100:.2f}%' for s in recall_scores],\n",
    "    'F1 (W)': [f'{s*100:.2f}%' for s in f1_scores],\n",
    "    'Fatal Recall': [f'{s*100:.2f}%' for s in fatal_recalls],\n",
    "    'Serious Recall': [f'{s*100:.2f}%' for s in serious_recalls]\n",
    "})\n",
    "\n",
    "print('Final Model Comparison (Balanced Data)')\n",
    "print('=' * 100)\n",
    "print(results_df.to_string(index=False))\n"
]

# Clear old outputs from Section 20 and 21
nb['cells'][206]['outputs'] = []
nb['cells'][208]['outputs'] = []

with open('road_accident_dl.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Sections 20 and 21 updated to use balanced models')

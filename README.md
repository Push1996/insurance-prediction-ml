# Insurance Claim and Age Prediction using ML

This project builds two machine learning models to predict:

- **Part 1 (Regression)**: Estimate the age of an insurance policyholder.
- **Part 2 (Classification)**: Predict whether a policyholder will make an insurance claim.

## 📦 Files Included

- `insurance_predictor.py`: Main script for both regression and classification tasks.
- `train.csv`, `test.csv`: Simulated training/testing data (non-sensitive, provided by instructor).
- `demo_notebook.ipynb`: Jupyter notebook for EDA, model development, and visualization.
- `project_report.pdf`: Summarizes methodology, evaluation, and findings.
- `predicted_ages.csv`: Output for Part 1 (regression).
- `predicted_claims.csv`: Output for Part 2 (classification).

## 🚀 Run Instructions

Ensure all files are in the same directory. Then run:

```bash
python3 insurance_predictor.py train.csv test.csv
```

## 🛠️ Features Used

- Feature selection and engineering
- Random Forests, XGBoost, and ensemble methods
- Handling imbalanced datasets
- Evaluation with MSE and Macro F1-score

## 📈 Evaluation Metrics

- **Part 1** (Regression): Mean Squared Error (MSE)
- **Part 2** (Classification): Macro F1-Score

## 📄 Report

See `project_report.pdf` for:

- Data cleaning and preprocessing strategies
- Model selection and hyperparameter tuning
- Business implications and insights

---

## 👨‍💻 Author

Po-Hsun Chang  
Master of Information Technology (Database Systems)  
University of New South Wales  
📧 chris89019@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/pohsunchang)

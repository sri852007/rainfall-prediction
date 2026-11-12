# Rainfall Prediction Using Machine Learning

A beginner-friendly B.Sc. Data Science internship project that classifies rainfall as **Yes (1)** or **No (0)** using weather measurements.

## Project structure

```text
rainfall-prediction/
├── data/
│   └── Rainfall.csv
├── model/
│   └── best_model.pkl
├── notebook/
│   └── Rainfall_Prediction.ipynb
├── inference.py
├── requirements.txt
└── README.md
```

## Dataset

The supplied `Rainfall.csv` contains 366 rows and 12 columns. The notebook strips whitespace from column names, converts the `rainfall` target (`yes`/`no`) to 1/0, and uses these eight input features:

1. `pressure`
2. `temparature`
3. `dewpoint`
4. `humidity`
5. `cloud`
6. `sunshine`
7. `winddirection`
8. `windspeed`

The notebook drops `day`, `maxtemp`, and `mintemp` before training. Missing values in `winddirection` and `windspeed` are filled with their medians.

## Machine Learning workflow

The notebook compares seven classifiers:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (RBF)
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- Gradient Boosting

It then tunes a Random Forest with `GridSearchCV` and evaluates the selected model using accuracy, ROC-AUC, a classification report, a confusion matrix, and an ROC curve. Run the notebook to view the actual metrics; this README intentionally does not invent or hard-code performance scores.

The supplied serialized model is a `RandomForestClassifier` with these saved settings:

- `n_estimators=400`
- `max_depth=5`
- `min_samples_leaf=5`
- `class_weight='balanced'`
- `random_state=42`

## Setup on Windows

Open Command Prompt in the project folder and create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `notebook/Rainfall_Prediction.ipynb`.

## Run a new prediction

From the project root folder:

```bash
python inference.py
```

Enter numeric values when prompted. The script uses the same eight feature names and order expected by the saved model and prints the predicted class and estimated probability.

**Important:** Enter measurements using the same units and conventions as the training dataset. This is an educational machine-learning estimate, not an official weather forecast.

## Model compatibility

The supplied model was serialized with scikit-learn **1.9.1**. Scikit-learn model files are not guaranteed to be compatible across versions. Install the matching version listed in `requirements.txt` when possible. Only load model files you trust.

## Notes

- Keep `data/Rainfall.csv`, `model/best_model.pkl`, and `inference.py` in their shown locations so the script can find the files.
- If you retrain the model in the notebook, save the selected estimator to `model/best_model.pkl` to update the model used by `inference.py`.

# Hugging Face Space URL: https://huggingface.co/spaces/vipulSP21/urban-nest-rent-predictor

# STTAI2026-miniproject
Urban Nest Rent Predictor implements the UrbanNest Analytics rent prediction workflow

# PropTech Startup Strategy - Rent Prediction Pipeline

This repository implements the UrbanNest Analytics rent prediction workflow required for the assignment. It compares Grid Search, Random Search, and Bayesian Optimisation for a `RandomForestRegressor`, tracks experiments with `trackio`, serves predictions through Streamlit, and includes a Docker setup for local runs and Hugging Face Spaces deployment.

## Repository Structure

```text
Assignment_4/
|-- README.md
|-- requirements.txt
|-- train.ipynb
|-- app.py
|-- Dockerfile
|-- models/
|-- plots/
|-- Dataset/
`-- screenshots/
```

## Implemented Deliverables

1. `train.ipynb` performs label encoding, 5-fold cross-validation, hyperparameter search, `trackio` logging, plot generation, final retraining, and artefact export.
2. `app.py` loads the saved model plus encoders with `pickle.load` and renders widget inputs for every feature.
3. `Dockerfile` packages the Streamlit app on port `8501`.

## Local Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

Run all cells in `train.ipynb` to generate:

- `models/best_rf_model.pkl`
- `models/label_encoders.pkl`
- `models/experiment_summary.json`
- `plots/trials_vs_error.png`
- `plots/optuna_hyperparameter_space.png`

Then launch the app and the tracking dashboard:

```bash
streamlit run app.py
trackio show --project urban-nest-rent-prediction
```

For the Task 1 screenshot requirement, open the Trackio dashboard after the notebook run, make sure the compared runs for Grid Search, Random Search, and Bayesian Optimization are visible with their scores and hyperparameters, and save the screenshot as `screenshots/trackio_dashboard.png`.

## Docker Usage

```bash
docker build -t urban-nest-rent-predictor
docker run -p 8501:8501 urban-nest-rent-predictor
```

Visit `http://localhost:8501` after the container starts.

## Manual Submission Steps Still Needed

- Deploy the project to a Hugging Face Docker Space and replace the placeholder URL at the top of this README.
- Save the required screenshots inside `screenshots/`:
  - `trackio_dashboard.png`
  - `docker_build.png`
  - `docker_ps.png`
  - `streamlit_working.png`
- Push the project to a private GitHub repository and add your TA as a collaborator.

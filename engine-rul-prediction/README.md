# Engine Remaining Useful Life (RUL) Prediction

## Objective
Predict the remaining useful life of aircraft engines using sensor data.

## Dataset
NASA CMAPSS Turbofan Engine Degradation Dataset (FD001).

## Pipeline
1. Exploratory Data Analysis
2. RUL label generation
3. Train-test split
4. Baseline Linear Regression
5. Random Forest Regression
6. Feature engineering with rolling statistics
7. Model saving and loading
8. RUL prediction for new engine data

## Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

## Results
- Baseline MAE: ~25 cycles
- Final MAE: ~2.1 cycles
- R² Score: ~0.99

## Output
The trained model can predict remaining engine life in operational cycles.

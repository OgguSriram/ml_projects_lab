## 🤖 Model Information

The final model used in this project is a **Random Forest Classifier** trained on an imbalanced dataset using **SMOTE** for oversampling.

### Final Model Configuration
- **Algorithm:** Random Forest
- **Number of trees:** 300
- **Maximum depth:** 30
- **Minimum samples per leaf:** 2
- **Max features:** sqrt
- **Random state:** 42

### Training Strategy
- Stratified train-test split (80/20)
- Feature scaling applied only to `Time` and `Amount`
- SMOTE applied **only on training data**
- Hyperparameters optimized using `RandomizedSearchCV`
- Decision threshold tuned using Precision–Recall curve

### Final Decision Threshold
- **Threshold:** 0.44  
- Selected to maximize recall while maintaining acceptable precision

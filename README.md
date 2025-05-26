# Forest Cover Type Prediction Using Machine Learning

## Overview
This project utilizes the Covertype Dataset from the UCI Machine Learning Repository to predict forest cover types in Roosevelt National Forest, Colorado. It implements and compares machine learning models including Random Forest, XGBoost, and a Neural Network, achieving an accuracy of approximately 0.96 with the tuned XGBoost model. The notebook includes data preprocessing, model training, evaluation, and visualization of results, making it a comprehensive demonstration of machine learning techniques.

## Key Features
- Handles class imbalance with class weights
- Implements Random Forest, XGBoost, and Neural Network models
- Includes visualizations for EDA and model performance
- Achieves high accuracy with tuned XGBoost

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kimkmathews/forest-cover-type-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the Dataset**:
   - The Covertype Dataset (`covtype.csv`) is not included in this repository due to its size. Download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Covertype).
   - Place the downloaded `covtype.csv` file in the `data/` directory of this repository.
4. Run the notebook:
   ```bash
   jupyter notebook covtype_model.ipynb
   ```

## Usage
- Explore the `covtype_model.ipynb` notebook for the complete workflow.
- Ensure the `data/` directory contains `covtype.csv` (see Installation step 3).
- Review the `images/` directory for generated visualizations.
- Note: The trained model file (`best_xgb_model.pkl`) is not included due to its size. You can retrain the model by running the notebook.

## Results
### Exploratory Data Analysis (EDA)
![EDA Distribution Plot](images/eda_distribution.png)
*Description: Distribution of key numerical features like elevation and slope.*

### Modeling Performance
![Confusion Matrix](images/confusion_matrix.png)
*Description: Confusion matrix for the tuned XGBoost model showing classification performance across cover types.*

![Model Accuracy Comparison](images/model_accuracy_comparison.png)
*Description: Comparison of accuracy scores for Random Forest, XGBoost, and Neural Network models.*

## Key Insights
- Random Forest provides a robust baseline with 0.95 accuracy.
- Tuned XGBoost achieves up to 0.96 accuracy, outperforming other models.
- Neural Network underperforms due to dataset complexity and imbalance.
- Class weighting improves recall for underrepresented classes.
- Misclassifications often occur between similar cover types.

# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model Name:** Logistic Regression Classifier
- **Model Type:** Classification
- **Training Algorithm:** Logistic Regression
- **Framework:** Scikit-learn

## Intended Use
This model predicts whether an individual's income exceeds $50,000 based on demographic and employment information. It can be used by organizations seeking to analyze workforce demographics and salary distributions or for educational purposes in data analysis.

## Training Data
The model was trained on publicly available Census Bureau data. This dataset includes various features such as age, work class, education, marital status, occupation, relationship status, race, sex, and native country.

## Evaluation Data
The evaluation was conducted on a test set equal to 20% of the data. The test set contains the same features as the training set, ensuring a fair assessment of the model's performance.

## Metrics
- **Precision:** 0.7490
- **Recall:** 0.6136
- **F1:** 0.6746

These metrics indicate that while the model has decent precision, there is room for improvement in recall. This suggests that the model may miss some positive cases.

## Ethical Considerations
As this model is trained on demographic data, it is important to consider the potential bias as reflected in historical inequalities. Care should be taken in how predictions are used, as they could reinforce biases against certain groups.

## Caveats and Recommendations
- This model is best suited for demographic analysis where context is understood.
- Further tuning and possibly using more advanced algorithms may improve performance, especially in recall.
- Continuous monitoring of the model's performance is recommended to ensure it remains valid as population demographics change.
# Machine Learning Models

This directory contains the serialized machine learning artifacts used by the `ClassificationAgent`.

## Files

*   **`model.pkl`**: A pre-trained XGBoost Classifier model.
*   **`label_encoder.pkl`**: A scikit-learn LabelEncoder used to decode the numeric predictions into human-readable labels (e.g., Low, Medium, High).
*   **`test_model.py`**: A standalone script to test the model's predictions on sample data without running the full agent pipeline.

## Usage

To test the model independently:

```bash
python test_model.py
```


need to test the new model , new classifiation agent , new test.py
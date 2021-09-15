# COPD Readmission Modelling

### Setup
Create a new environment and simply run `pip install -r requiremtns.txt` to install all needed libraries

### Training
`full_train_pipeline` is a self-contained training script for the XGBoost model from the SQL data. Simply run
`python full_train_pipeline` to train a new model.

The model will be saved in a new directory `models` containing all normalizers and saved XGBoost models.

After training is complete, an out of fold (OOF) AUC score and confusion matrix will be returned. Verify these are similar to previous runs and within expectations. 

Note that `full_train_pipeline` can use data directly from the sql server or from a csv file.

### Inference
`infer_xgb` can be used a few ways - either returning the output to a single line or returning a csv of the entire
database/input csv.

`python infer_xgb --row all` to infer on all the rows provided through SQL/CSV, and output a CSV with patient_id and confidence

`python infer_xgb --row 12` or another integer in order to infer on only a given row and return/print the confidence. 

Note that `infer_xgb.py` can use data directly fomr the sql server or from a csv file.

Inference can only happen following training - it loads models from `models/` so training must be run first.


### SHAP
`shap.ipynb` presents Shapley values in an easy to interpret method. There is currently the option to create a waterfall plot for one patient from inference. 
There also is the option to use the Jupyter Notebook file to create both a summary waterfall plot for feature importance as well as individual plots for specific patients.

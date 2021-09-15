import pyodbc
import numpy as np
from imblearn.over_sampling import SVMSMOTE
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, Normalizer
import xgboost as xgb
import pandas as pd
import pgeocode
import random
import os
from gauss_rank_scaler import GaussRankScaler
import pickle


def seed_everything(seed):
    """
    :param seed: int, random seed
    :return: None
    Sets random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# Pulls all the data from the sql server and saves it as a dataframe
def get_sql_dataframe():
    # return: Dataframe from server
    server = '10.100.201.31'
    database = 'surgery'
    username = 'kingds'
    password = ''
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID='
                          + username + ';PWD=' + password)
    cursor = cnxn.cursor()

    tableResult = pd.read_sql(
        "SELECT * FROM [surgery].[dbo].[dm_prediction_data_v5]", cnxn)
    df = pd.DataFrame(tableResult)
    return df


def clean_data(data):
    # Param data: Takes in the dataframe from the sql server
    # Returns:  the cleaned up dataframe

    # Filter for data before Covid
    # data = data[~(data['unit_date'] > '2020-03-15')]

    # Drop blanks for gender, postal code, diagnosis code
    data['gender'].replace('', np.nan, inplace=True)
    data.dropna(subset=['gender'], inplace=True)
    data = data[data.gender != 'U']

    data['postal_code'].replace('', np.nan, inplace=True)
    data.dropna(subset=['postal_code'], inplace=True)

    data['diagnosis_code'].replace('', np.nan, inplace=True)
    data.dropna(subset=['diagnosis_code'], inplace=True)

    # Fill empty spots with the mean value
    data['day_from_er_visit'].fillna((data['day_from_er_visit'].mean()), inplace=True)
    data['day_from_acute_visit'].fillna((data['day_from_acute_visit'].mean()), inplace=True)

    # Keep certain practices
    services = ['FAMILY PRACTICE', 'GENERAL MEDICINE', 'EMERGENCY MEDICINE']
    data = data[data['service_admit'].isin(services)]

    # Create new variable initial month of visit
    data['initial_month_of_visit'] = pd.DatetimeIndex(data['unit_date']).month

    # Create new variable initial year of visit
    data['initial_year_of_visit'] = pd.DatetimeIndex(data['unit_date']).year

    # Take provider family and create flag of if one exists for patient or not
    # if rows of that column are empty than make new column a 0 otherwise make it a 1
    data['have_family_doctor'] = 1
    data.loc[data['provider_family'].isnull(), 'have_family_doctor'] = 0

    # Get dummies for certain variables
    data = pd.get_dummies(data, columns=['marital_status', 'gender', 'facility', 'service_admit'])

    # Create new variable for how long a stay is
    data['length_of_stay'] = 0
    total_visit = data['epiurn'].value_counts()
    df = pd.DataFrame({'epiurn': total_visit.index, 'length_of_stay': total_visit.values})
    data = data.merge(df, on='epiurn')

    # get cummulative days spent in each hospital section by visit
    grouped_visits = data.groupby(by=["epiurn"])
    # Creates a series? which has the epiurn and a list of the mt location codes applied to that epiurn
    grouped_hospital_rooms = grouped_visits["mt_location_code"].apply(list)
    # Counts how many time each mt location code comes up in the list for each epiurn
    counted_hospital_stay = grouped_hospital_rooms.apply(pd.Series.value_counts)
    # Fills na's with 0's for rooms not visited by patients during visit
    clean_hospital_stays = counted_hospital_stay.fillna(0)

    # merge the full data with the new visit separated date
    non_dup_data = data.drop_duplicates(subset=['epiurn'])

    result = non_dup_data.join(clean_hospital_stays, on='epiurn', lsuffix='_left', rsuffix='_right')

    # Dropping non Canadian postal code
    postal_code = result['postal_code'].map(lambda a: a if len(a) == 7 and a[3] == ' ' else None)
    postal_code = postal_code.dropna()

    # Get the longitude and latitude for each postal code
    nomi = pgeocode.Nominatim('ca')

    pc_data = postal_code.map(lambda pc: nomi.query_postal_code(pc))

    # Create array for longitude and latitiude
    lat = np.array([i['latitude'] for i in pc_data])
    long = np.array([i['longitude'] for i in pc_data])

    # Change the arrays to Dataframes
    lat = pd.DataFrame(lat, columns=['latitude'])
    long = pd.DataFrame(long, columns=['longitude'])

    # Concat those Dataframes
    long_lat = pd.concat([long, lat], axis=1)

    data = result.drop(
        columns=['visit_type', 'triage_level', 'bone_density', 'ct', 'echo', 'interventional', 'mammography', 'mri',
                 'nuclear_medicine', 'respiratory', 'ultrasound', 'xray', 'north_hastings_lab_test',
                 'miscellaneous', 'blood_bank_charge_only', 'pku_test', 'blood_gas_cord_venous',
                 'bone_marrow_aspirate', 'blood_gas_cord_arterial', 'immuno_histo_chemistry', 'chemistry',
                 'send_out', 'infection_control', 'no_prefix', 'surgical', 'congulation', 'routine_microbiology',
                 'blood_gas', 'cytology', 'blood_bank_units', 'urinalysis', 'hematology', 'serology_for_poc',
                 'referred_out_cystology', 'virology', 'serology', 'miscellaneous_send_out',
                 'chemistry_miscellaneous', 'autopsy', 'blood_bank', 'body_fluids',
                 'referred_out_microbiology', 'surgical_send_out_to_bgh', 'institution_from', 'institution_to',
                 'lvl0_diagnosis', 'lvl1_diagnosis', 'lvl2_diagnosis', 'earlier_of_triage_and_registration_datetime',
                 'alc_care_type_required', 'is_vent_longterm', 'is_vent', 'is_conversion',
                 'weight', 'height', 'residence', 'pain_scale', 'id', 'is_coded', 'day_to_acute_visit',
                 'previous_er_diagnosis', 'admit_diagnosis', 'palliative_score', 'unit_date',
                 'day_to_er_visit', 'has_telemetry_on_day', 'is_to_er_similar_icd10_diagnosis',
                 'is_to_acute_similar_icd10_diagnosis', 'provider_most_responsible',
                 'provider_family', 'reason_for_visit', 'diagnosis', 'length_of_stay_x',
                 'mt_location_code', 'is_death', 'postal_code'
                 ])

    # Combines the data with the lat long Dataframe
    data.reset_index(drop=True, inplace=True)
    long_lat.reset_index(drop=True, inplace=True)
    cleaned_data = pd.concat([data, long_lat], axis=1)

    # Not all postal codes could find a longitude and latitiude hence dropping the few that don't have them
    cleaned_data = cleaned_data.dropna()

    # Filter for only COPD patients
    codes = ['J44.0', 'J44.1', 'J44.9']
    copd_data = cleaned_data.loc[cleaned_data['diagnosis_code'].isin(codes)]

    del copd_data['diagnosis_code']

    # Get list of call the columns to use in inference to be sure the columns are the same for training and inference
    cols = list(copd_data)

    # Create a readmitted feature
    copd_data['readmitted'] = 0
    # Adding readmission criteria
    # Those who were in hospital less than or equal to 30 days ago, COPD patients will be filtered later
    copd_data.loc[(copd_data['day_from_acute_visit'] <= 30), 'readmitted'] = 1

    # Save the columns to a pickle to be opened in inference file
    with open('cols_list.pickle', 'wb') as handle:
        pickle.dump(cols, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del copd_data['epiurn'], copd_data['patient_id']

    return copd_data


def train_xgb(x_train, y_train, fold, x_val=None, y_val=None):
    """
    :param x_train: np.ndarray, train features
    :param y_train: np.ndarray, train targets
    :param fold: int, fold number (saves model/normalizer to ./models/fold.{json, pkl})
    :param x_val: optional np.ndarray, validation features
    :param y_val: optional np.ndarray, validation targets
    :return: validation predictions (np.ndarray), feature importance (dict)
    """
    params = {
        'objective': 'binary:logistic',
        'learning_rate': 0.04,
        'max_depth': 5,
        'n_estimators': 1000,
        'min_child_weight': 5,
        'colsample_bytree': 0.5,
        'eval_metric': 'auc',
        'nthread': 24,
        'verbosity': 0,
        'seed': 42,
        'num_class': 1,
        'use_label_encoder': False
    }

    # smart weights initialization, converges faster.
    base_pred = np.log((y_train).mean())

    xgb_model = xgb.XGBClassifier(**params)
    if x_val is not None:
        xgb_model.fit(x_train, y_train, base_margin=np.ones(x_train.shape[0]) * base_pred,
                      eval_set=[(x_val, y_val)], early_stopping_rounds=100, verbose=False)
        val = xgb_model.predict_proba(x_val)
    else:
        xgb_model.fit(x_train, y_train, base_margin=np.ones(x_train.shape[0]) * base_pred)
        val = None
    xgb_model.save_model(f'./models/{fold}.json')

    # get feature importance from gradient booster
    importance = xgb_model.get_booster().get_score(importance_type='gain')
    return val, importance


def main():
    random_seed = 42
    seed_everything(random_seed)

    # make models directory if it does not exist
    if not os.path.exists('./models'):
        os.makedirs('./models')

    # set number of fold and scaling method
    NFOLD = 5
    SCALING = 'norm'  # norm or rankgauss

    # Reads in data from server or csv file
    # Ensure password is included
    #df = get_sql_dataframe()
    df = pd.read_csv('~/PycharmProjects/COPD/GitLab/copd/xgboost_deploy/full_data.csv')
    df = clean_data(df)

    # get features and targets
    features = df.loc[:, df.columns != 'readmitted'].values
    targets = df['readmitted'].values

    # initialize stratified k fold cross validation
    skf = StratifiedKFold(n_splits=NFOLD, random_state= random_seed, shuffle=True)

    oof_preds = []
    oof_targets = []

    feature_gain = []
    onehot = OneHotEncoder()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(features, targets)):
        if SCALING == 'norm':
            normalizer = Normalizer().fit(features[tr_idx])

            x_train = normalizer.transform(features[tr_idx])
            x_val = normalizer.transform(features[val_idx])

            # save normalizer for future use
            pickle.dump(normalizer, open(f'./models/{fold}.pkl', 'wb'))

        if SCALING == 'rankgauss':
            # evil hack to make sure columns are not all zeros
            # sets the last value of an all-zero column to one for rankgauss
            x_train = features[tr_idx]
            for i in np.where(~x_train.any(axis=0))[0]:
                x_train[-1, i] = 1
            scaler = GaussRankScaler().fit(x_train)
            x_train = scaler.transform(x_train)

            x_val = features[val_idx]
            for i in np.where(~x_val.any(axis=0))[0]:
                x_val[-1, i] = 1
            x_val = scaler.transform(x_val)

        # separate into train/val targets
        y_train = targets[tr_idx]
        y_val = targets[val_idx]

        # SVM SMOTE (minority oversampling) on both train and val
        oversample = SVMSMOTE()
        x_train, y_train = oversample.fit_resample(x_train, y_train)
        x_val, y_val = oversample.fit_resample(x_val, y_val)

        # train xgb model
        preds, importance = train_xgb(x_train, y_train, fold, x_val=x_val, y_val=y_val)
        feature_gain.append(importance)

        # one hot validation targets to calculate AUC with.
        oh_y_val = onehot.fit_transform(y_val.reshape(-1, 1)).toarray()
        print(f"Fold {fold+1} AUC", roc_auc_score(y_true=oh_y_val, y_score=preds))

        # append to out of fold (for future validation, or to form a csv of predictions)
        oof_targets.append(y_val)
        oof_preds.append(preds)

    # combine out of fold (OOF) predictions and targets
    oof_preds_all = np.concatenate(oof_preds)
    oof_targets_all = np.concatenate(oof_targets)

    # reshape onehot targets
    onehot_targets = onehot.fit_transform(oof_targets_all.reshape(-1, 1)).toarray()

    # calculate auc/confusion matrix on OOF.
    print("OOF AUC:", roc_auc_score(y_true=onehot_targets, y_score=oof_preds_all))
    print("OOF Confusion Matrix:", confusion_matrix(oof_targets_all, oof_preds_all.argmax(1), normalize='true'))

    # pretty print the feature importance with the corresponding column names
    imp = pd.DataFrame(feature_gain)
    answer = dict(imp.mean())
    importance = {}
    for i in answer.keys():
        importance[df.columns[int(i[1:3])]] = answer[i] / sum(answer.values())
    importance = dict(sorted(importance.items(), key=lambda item: item[1])[::-1])

    print(importance)

    # train final model on all data, save normalizeir as well.

    print("Training final model")
    normalizer = Normalizer().fit(features)
    features = normalizer.transform(features)
    pickle.dump(normalizer, open('./models/final.pkl', 'wb'))
    _, importance = train_xgb(features, targets, 'final')


if __name__ == '__main__':
    main()

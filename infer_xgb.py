import xgboost as xgb
import argparse
import pickle
import pandas as pd
import numpy as np
import pgeocode
import matplotlib.pyplot as plt
import shap
import os
import pyodbc


def get_sql_dataframe():
    """
    :return: Dataframe from server
    """
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
    """
    :param data: Dataframe from sql server or csv file
    :return: clean Dataframe, epiurn column and patient id column for prediction output
    """

    # Drop blanks for gender, postal code, diagnosis code
    data['gender'].replace('', np.nan, inplace=True)
    data.dropna(subset=['gender'], inplace=True)

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

    # get cummulative days spent in each hospital section by visit=
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

    result.reset_index(drop=True, inplace=True)
    long_lat.reset_index(drop=True, inplace=True)
    data = pd.concat([result, long_lat], axis=1)

    data = data.drop(
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
                 'mt_location_code', 'is_death', 'postal_code', 'diagnosis_code'
                 ])

    infer_cols = list(data)

    # Open the pickle file with the list of total columns used from training
    with open('cols_list.pickle', 'rb') as handle:
        train_cols = pickle.load(handle)

    # Check if the list of columns in the inference Dataframe matches the list of columns from the training Dataframe
    # If it matches continue, if inference is missing a column add it and fill with zeros
    for i in train_cols:
        if i in list(data):
            continue
        else:
            data[i] = 0

    # Check if the list of columns in the training Dataframe matches the list of columns from the inference Dataframe
    # If it matches continue, if inference has an extra column drop it
    for j in infer_cols:
        if j in train_cols:
            continue
        else:
           del data[j]

    clean_data = data.dropna()
    clean_data.reset_index(drop=True, inplace=True)

    # Get the columns epiurn and patient id for the predictino output csv
    cleaned_epiurn = clean_data['epiurn']
    cleaned_patient = clean_data['patient_id']

    del clean_data['epiurn'], clean_data['patient_id']

    return clean_data, cleaned_epiurn, cleaned_patient


def infer(row, clean_df, epiurn, patient, thresh, csv=False):
    '''
    :param row: int, row to infer on. Only used if csv=False
    :param clean_df: the Dataframe from clean_data()
    :param epiurn: the columnn of epiurn for the prediction output csv
    :param patient: the columnn of patient id for the prediction output csv
    :param thresh: determines at what value a person will be classified as readmitted or not
    :param csv: bool, whether to infer on entire dataframe and output csv
    :return: row if csv=False, otherwise None (csv outputted to out.csv)
    '''

    # load the normalizer from the final model
    normalizer = pickle.load(open('./models/final.pkl', 'rb'))

    # load final xgboost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(f'./models/final.json')
    if not csv:
        # locate single row
        feature = clean_df.loc[row, :].values.reshape(1, -1)
        # normalise row and predict
        feature = normalizer.transform(feature)
        return xgb_model.predict_proba(feature)[0]

    if csv:
        # get patient id (since the csv is precleaned, in the future, this can just use patients from sql)
        patient_id = patient
        epiurn = epiurn

        # locate features, normalise, and predict
        features = clean_df.loc[:, :]
        features = normalizer.transform(features)
        out = xgb_model.predict_proba(features)

        # output to csv (out.csv) with patient_id and readmission probability
        df = pd.DataFrame({
            'patient_id': patient_id,
            'readmission probability': out[:, 1],
            'readmitted': np.where(out[:, 1] > thresh, 1, 0),
            'epiurn': epiurn
        })
        df.to_csv('out.csv', index=False)


def generate_shap(row, clean_df):

    normalizer = pickle.load(open('./models/final.pkl', 'rb'))
    clean_df.loc[:, clean_df.columns != 'readmitted'] = normalizer.transform(clean_df.loc[:, clean_df.columns != 'readmitted'])
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('./models/final.json')

    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(clean_df.loc[:, clean_df.columns != 'readmitted'])

    print(xgb_model.predict_proba(clean_df.loc[:, clean_df.columns != 'readmitted'].iloc[row].values.reshape(1, -1)))
    shap.plots.waterfall(shap_values[row], show=False)

    # make shap directory if it does not exist
    if not os.path.exists('./shap'):
        os.makedirs('./shap')

    # save shap plot
    plt.savefig(f"./shap/{row}.png", dpi=100, bbox_inches="tight")
    plt.close()
    return f"./shap/{row}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--row', type=str, default='all',
                        help='row of csv to infer on (or all to infer on entire database)')
    parser.add_argument('--thresh', type=str, default=0.5,
                        help='Set threshold of "argmax" in CSV (default 0.5), only works with --row all')
    parser.add_argument('--shap', action='store_true',
                        help='saves SHAP graph in ./graphs - only works when --row is an int (not all)')

    args = parser.parse_args()

    # load data (either from CSV or from SQL)
    df = pd.read_csv('~/PycharmProjects/COPD/GitLab/copd/xgboost_deploy/last_year.csv')
    # Ensure password is there to reach server
    #df = get_sql_dataframe()

    clean_df, epiurn, patient = clean_data(df)

    if args.row == 'all' or args.row == 'csv':
        # infer on all rows and output a csv
        infer(args.row, clean_df, epiurn, patient, args.thresh, csv=True)
    else:
        # infer on a single row, and output shap if so desired.
        out = infer(int(args.row), clean_df, epiurn, patient, args.thresh)
        print(f'Probability for no readmission: {100 * out[0]:.2f}, probability for readmission: {100 * out[1]:.2f}')
        if args.shap:
            shap_path = generate_shap(int(args.row), clean_df)
            print(f"shape saved to {shap_path}")


if __name__ == '__main__':
    main()

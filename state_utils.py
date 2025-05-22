import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta, datetime, date
from datetime import time as dttime
from zoneinfo import ZoneInfo

def get_dbs_on_date(pt_data):
    """
    Retrieves date that stimulation was turned on from info dict.

    Parameters:
    - pt_data (dictionary): patient info dictionary containing DBS on date in string format.
    
    Returns:
    - dbs_on_date (datetime.date): DBS on date in the datetime.date type.
    """
    return datetime.strptime(pt_data['dbs_date'], '%Y-%m-%d').date()

def get_state_labels(pt_df, scale_val_colname, scale_name, pt_scale_df, dbs_on_date, pt_id, date_today, response_pct_reduction=0.35):
    """
    Calculate the patient's state at each time point based on their scale scores.
    State key:
        0 = Pre-DBS
        1 = Disinhibited
        2 = Non-responder
        3 = Responder
        4 = Transition
        5 = Unknown
    
    Parameters:
    - pt_df (pd.DataFrame): DataFrame containing all data for this patient.
    - scale_val_colname (str): Column name in pt_df containing the most recent scale value.
    - scale_name (str): Name of the scale to match.
    - pt_scale_df (pd.DataFrame): DataFrame containing scale data for this patient.
    - dbs_on_date (datetime.date): DBS on date in the datetime.date type.
    - pt_id (str): Patient ID.
    - response_pct_reduction (float): Percentage reduction in scale score to define response. Default is 0.35 (35%).

    Returns:
    - pt_df (pd.DataFrame): Updated DataFrame containing state labels and strings.
    """
    predbs_scores = pt_scale_df.query('date <= @dbs_on_date')[scale_name].dropna()
    baseline_score = np.max(predbs_scores)
    target_score = baseline_score * (1 - response_pct_reduction) # 35% reduction in YBOCS defines response

    # Assign all data points a default starting label of 4 ('unknown').
    pt_df['state_label'] = 5 # unknown/unlabeled
    pt_df['state_label_str'] = 'Unknown'

    # Assign pre-DBS points a label of 0.
    pt_df.loc[pt_df['days_since_dbs'] < 0, 'state_label'] = 0 # pre-DBS
    pt_df.loc[pt_df['days_since_dbs'] < 0, 'state_label_str'] = 'Pre-DBS' # pre-DBS

    pt_df.loc[pt_df[scale_val_colname] < target_score, 'state_label'] = 3 # Responder
    pt_df.loc[pt_df[scale_val_colname] < target_score, 'state_label_str'] = 'Responder' # Responder
    if (pt_scale_df[scale_name] < target_score).any(): # If they are/were a responder
        pt_df.loc[(pt_df[scale_val_colname] > target_score) & (pt_df['days_since_dbs'] > 0), 'state_label'] = 2 # Non-Responder
        pt_df.loc[(pt_df[scale_val_colname] > target_score) & (pt_df['days_since_dbs'] > 0), 'state_label_str'] = 'Non-Responder' # Non-Responder
        first_response_date = pt_scale_df.loc[pt_scale_df[scale_name] < target_score, 'date'].min().tz_localize('US/Central')
        pt_df.loc[(pt_df['CT_timestamp'] < first_response_date) & (pt_df['days_since_dbs'] > 0), 'state_label'] = 4 # Transition
        pt_df.loc[(pt_df['CT_timestamp'] < first_response_date) & (pt_df['days_since_dbs'] > 0), 'state_label_str'] = 'Transition' # Transition
    elif date_today - dbs_on_date > timedelta(days=6*30): # If they are not a responder and it has been more than 6 months since DBS
        pt_df.loc[pt_df['days_since_dbs'] > 0, 'state_label'] = 2 # Non-Responder
        pt_df.loc[pt_df['days_since_dbs'] > 0, 'state_label_str'] = 'Non-Responder' # Non-Responder
    else: # If they haven't responded yet but stimulation began less than 6 months ago, label as unknown.
        pt_df.loc[pt_df['days_since_dbs'] > 0, 'state_label'] = 5 # Unknown
        pt_df.loc[pt_df['days_since_dbs'] > 0, 'state_label_str'] = 'Unknown' # Unknown
    pt_df['state_label'] = pt_df['state_label'].astype('int').astype('category')
    pt_df['state_label_str'] = pt_df['state_label_str'].astype('category')
    pt_df['state_label'] = pt_df['state_label'].cat.set_categories([0, 1, 2, 3, 4, 5], ordered=False)
    pt_df['state_label_str'] = pt_df['state_label_str'].cat.set_categories(['Pre-DBS', 'Disinhibited', 'Non-Responder', 'Responder', 'Transition', 'Unknown'], ordered=False)
    return pt_df

def add_disinhibited(pt_df, pt_dict):
    """
    Add disinhibited state labels to the patient's DataFrame.

    Parameters:
    - pt_df (pd.DataFrame): DataFrame containing all data for this patient.
    - pt_dict (dict): patient info dictionary containing disinhibited time range.

    Returns:
    - new_state_labels (pd.DataFrame): DataFrame containing updated state labels and strings.
    """
    if 'disinhibited' not in pt_dict:
        return pt_df[['state_label', 'state_label_str']]
    
    new_state_labels = pd.DataFrame({'state_label': pt_df['state_label'].values, 'state_label_str': pt_df['state_label_str'].values}, index=pt_df.index)
    # Get the disinhibited time range for this patient
    start_disinhibited = pt_dict['disinhibited'][0]
    end_disinhibited = pt_dict['disinhibited'][1]
    disinhibited_days = np.arange(start_disinhibited, end_disinhibited + 1)
    inds = pt_df[pt_df['days_since_dbs'].isin(disinhibited_days)].index
    new_state_labels.loc[inds, 'state_label'] = 1
    new_state_labels.loc[inds, 'state_label_str'] = 'Disinhibited'
    return new_state_labels

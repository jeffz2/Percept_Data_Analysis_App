import json
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, date
from datetime import time as dttime
from zoneinfo import ZoneInfo
import adbs_ofc
import json_utils


def generate_raw(pt_name: str, patient_dict: dict):

    # Read json files from provided directory

    pt_raw_df = []

    jsons = json_utils.get_json_filenames(patient_dict['directory'])
    pt_changes_df = pd.DataFrame(columns=['timestamp', 'pt_id'] + [f'{side}_{attr}' for side in ['left', 'right'] for attr in ['amplitude', 'pulse_width', 'freq', 'contacts']] + ['source_file'])
    
    raw_data_list = []
    for filename in jsons:
        try:
            with open(filename, 'r') as f:
                js = json.load(f)
        except (PermissionError, json.JSONDecodeError) as e:
            print(e)
            continue
    
        raw = json_utils.chronic_lfp_from_json(js, filename)
        if not raw.empty:
            raw_data_list.append(raw)

        changes_dict = json_utils.get_param_changes(js)
        if changes_dict is not None:
            for t in changes_dict.keys():
                changes = changes_dict[t]
                for hem in ['Left', 'Right']:
                    hem_changes = changes[hem]
                    if hem_changes is not None:
                        for attr in ['amplitude', 'pulse_width', 'freq', 'contacts']:
                            if attr in hem_changes:
                                pt_changes_df.loc[t, f'{hem.lower()}_{attr}'] = hem_changes[attr]
                pt_changes_df.loc[t, 'timestamp'] = t
                pt_changes_df.loc[t, 'source_file'] = filename
    pt_changes_df['pt_id'] = pt_name
    try:
        raw_data = pd.concat(raw_data_list, ignore_index=True)
        raw_data['pt_id'] = pt_name
        if raw_data.size != 0:
            pt_raw_df.append(raw_data)
    except ValueError as e:
        print(f'No chronic LFP power data from {pt_name}')
        return ProcessLookupError

    raw_df = pd.concat(pt_raw_df, ignore_index=True)

    # clean data specific to aDBS OFC
    if pt_name in ['B014', 'B015', 'B016', 'B017', 'B018']:
        raw_df = adbs_ofc.clean(raw_df, pt_name)

    # Relabel all remaining "OTHER" lead locations to VC/VS
    raw_df.loc[raw_df['left_lead_location'] == "OTHER", 'left_lead_location'] = 'VC/VS'
    raw_df.loc[raw_df['right_lead_location'] == "OTHER", 'right_lead_location'] = 'VC/VS'

    pt_changes_df.sort_values(by=['pt_id', 'timestamp'], inplace=True, ignore_index=True)

    assert (raw_df['left_lead_location'] == raw_df['right_lead_location']).all()
    raw_df['lead_location'] = raw_df['left_lead_location'].where(raw_df['left_lead_location'] == raw_df['right_lead_location'], None)
    raw_df.drop(columns=['left_lead_location', 'right_lead_location'], inplace=True)

    assert (raw_df['left_lead_model'] == raw_df['right_lead_model']).all()
    raw_df['lead_model'] = raw_df['left_lead_model'].where(raw_df['left_lead_model'] == raw_df['right_lead_model'], None)
    raw_df.drop(columns=['left_lead_model', 'right_lead_model'], inplace=True)

    return raw_df, pt_changes_df
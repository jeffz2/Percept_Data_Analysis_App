import datetime
from datetime import datetime
import pandas as pd
import plotly.io as pio
import tempfile
from PySide6.QtWidgets import QFileDialog
import numpy as np

DATE_FORMAT = '%Y-%m-%d'
HEMI_DICT = {0: "left", 1: "right"}
INTERP_MODEL = 'SLOvER+'

def translate_param_dict(input_data):  
    translated_data = {
        "dbs_date": input_data['Initial_DBS_programming_date'],
        "subject_name": input_data['subject_name'],
        "directory": input_data['directory'],
        "responder": input_data['responder'],
        "responder_date": input_data['responder_date'],
        "hemisphere": 0
    }

    return translated_data

def add_extension(filename, extension):
    return filename if filename.lower().endswith(extension.lower()) else filename + extension

def save_lin_ar_feature(data, filename):
    ext = filename.split('.')[-1].lower()
    
    if ext == 'json':
        data.to_json(filename, orient='records', lines=True)
    elif ext == 'xlsx':
        data.to_excel(filename, index=False)
    elif ext == 'tsv':
        data.to_csv(filename, sep='\t', index=False)
    elif ext == 'txt':
        with open(filename, 'w') as f:
            f.write(data.to_string(index=False))
    else:
        filename = add_extension(filename, '.csv')
        data.to_csv(filename, index=False)

def save_plot(fig, filename):
    ext = filename.split('.')[-1].lower()
    
    if ext in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf']:
        pio.write_image(fig, filename, format=ext)
    else:
        filename = add_extension(filename, '.html')
        fig.write_html(filename, include_plotlyjs='cdn')

def open_file_dialog(parent):
    file_dialog = QFileDialog()
    return file_dialog.getOpenFileNames(parent, 'Select patient JSON files', '', 'JSON files (*.json)')[0]

def open_save_dialog(parent, title, default_filter):
    file_path, _ = QFileDialog.getSaveFileName(parent, title, "", default_filter)
    return file_path

def create_temp_plot(fig):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
        temp_file.write(fig.to_html(include_plotlyjs='cdn').encode('utf-8'))
        return temp_file.name

def prepare_export_data(df_w_preds, param_dict):
    df_w_preds.sort_values(by='time_bin', inplace=True)
    linAR_r2 = np.unique(df_w_preds[f'lfp_{HEMI_DICT[param_dict['hemisphere']]}_day_r2_{INTERP_MODEL}'].values)
    days = np.unique(df_w_preds['days_since_dbs'].values)
    export_dict = {
        'Days_since_DBS': days.tolist(),
        'R2_values': linAR_r2.tolist()
    }
    lin_ar_df = pd.DataFrame(export_dict)
    lin_ar_df['activation_state'] = (df_w_preds.drop_duplicates('days_since_dbs'))['state_label_str'].values
    return lin_ar_df

def validate_date(date_str):
    try:
        datetime.strptime(date_str, DATE_FORMAT)
        return True
    except ValueError:
        return False

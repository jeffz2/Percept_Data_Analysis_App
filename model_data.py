import pandas as pd
import numpy as np
import model_utils

def model_data(df: pd.DataFrame, window_size=3, causal=True, use_constant=False):
    feature_names = ['OvER']
    hemis = ['left', 'right']
    df_w_preds = df.copy()

    for name in feature_names:
        if window_size == 1:
            daily_groups = df_w_preds.groupby(['pt_id', 'lead_location', pd.Grouper(key='CT_timestamp', freq='D')], group_keys=False)
            ar_features = [f'lfp_{hemis[0]}_z_scored_{name}_lag_1'] + (['constant'] if use_constant else [])
            hemi1_results_df = daily_groups.apply(lambda g: model_utils.predict_series_and_calc_R2_Kfold(g, ar_features, f'lfp_{hemis[0]}_z_scored_{name}'), include_groups=False)
            ar_features = [f'lfp_{hemis[1]}_z_scored_{name}_lag_1'] + (['constant'] if use_constant else [])
            hemi2_results_df = daily_groups.apply(lambda g: model_utils.predict_series_and_calc_R2_Kfold(g, ar_features, f'lfp_{hemis[1]}_z_scored_{name}'), include_groups=False)
        else:
            pt_groups = df_w_preds.groupby(['pt_id', 'lead_location'], group_keys=False)
            ar_features = [f'lfp_{hemis[0]}_z_scored_{name}_lag_1'] + (['constant'] if use_constant else [])
            hemi1_results_df = pt_groups.apply(lambda g: model_utils.apply_sliding_window(g, ar_features, f'lfp_{hemis[0]}_z_scored_{name}', window_size=window_size, causal=causal), include_groups=False)
            ar_features = [f'lfp_{hemis[1]}_z_scored_{name}_lag_1'] + (['constant'] if use_constant else [])
            hemi2_results_df = pt_groups.apply(lambda g: model_utils.apply_sliding_window(g, ar_features, f'lfp_{hemis[1]}_z_scored_{name}', window_size=window_size, causal=causal), include_groups=False)
        df_w_preds = pd.merge(df_w_preds, hemi1_results_df, how='outer', left_index=True, right_index=True)
        df_w_preds = pd.merge(df_w_preds, hemi2_results_df, how='outer', left_index=True, right_index=True)

    return df_w_preds
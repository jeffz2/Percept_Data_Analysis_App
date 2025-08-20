import pandas as pd
import numpy as np
import utils.model_utils as model_utils
import json


def model_data(
    df: pd.DataFrame,
    window_size=3,
    causal=True,
    use_constant=False,
    ark=False,
    max_lag=144,
):
    with open("data/param.json", "r") as f:
        param_dict = json.load(f)
    model = param_dict["model"]
    window_size = int(param_dict["Window size"])
    hemis = ["left", "right"]

    df["constant"] = 1.0
    df_w_preds = df.copy()

    if window_size == 1:
        daily_groups = df_w_preds.groupby(
            ["pt_id", "lead_location", pd.Grouper(key="CT_timestamp", freq="D")],
            group_keys=False,
        )
        ar_features = (
            (
                [f"lfp_{hemis[0]}_z_scored_{model}_lag_1"]
                + (["constant"] if use_constant else [])
            )
            if not ark
            else model_utils.select_sig_lags_kfold()
        )
        hemi1_results_df = daily_groups.apply(
            lambda g: model_utils.predict_series_and_calc_R2_Kfold(
                g, ar_features, f"lfp_{hemis[0]}_z_scored_{model}"
            ),
            include_groups=False,
        )
        ar_features = [f"lfp_{hemis[1]}_z_scored_{model}_lag_1"] + (
            ["constant"] if use_constant else []
        )
        hemi2_results_df = daily_groups.apply(
            lambda g: model_utils.predict_series_and_calc_R2_Kfold(
                g, ar_features, f"lfp_{hemis[1]}_z_scored_{model}"
            ),
            include_groups=False,
        )
    else:
        if ark:

            for hemi in hemis:
                lag_prefix = f"lfp_{hemi}_z_scored_{model}_lag_"
                all_lags = [f"{lag_prefix}{i}" for i in range(1, max_lag + 1)]
                target = f"lfp_{hemi}_z_scored_{model}"

                # Filter for patient/lead group
                pt_groups = df_w_preds.groupby(
                    ["pt_id", "lead_location"], group_keys=False
                )

                # Use a helper to select significant lags for each group
                def process_group(group):
                    try:
                        sig_lags = model_utils.select_significant_lags_kfold(
                            group, all_lags, target
                        )
                        sig_lags = model_utils.drop_sparse_lags(
                            group, all_lags, threshold=0.6
                        )
                        if use_constant:
                            sig_lags.append("constant")
                        return model_utils.apply_sliding_window(
                            group,
                            sig_lags,
                            target,
                            window_size=window_size,
                            causal=causal,
                        )
                    except Exception as e:
                        return pd.DataFrame()

                try:
                    results_df = pt_groups.apply(process_group, include_groups=False)
                except Exception as e:
                    ar_features = [f"lfp_{hemi}_z_scored_{model}_lag_1"] + (
                        ["constant"] if use_constant else []
                    )
                    results_df = pt_groups.apply(
                        lambda g: model_utils.apply_sliding_window(
                            g,
                            ar_features,
                            f"lfp_{hemi}_z_scored_{model}",
                            window_size=window_size,
                            causal=causal,
                        ),
                        include_groups=False,
                    )
                # Merge predictions back
                df_w_preds = pd.merge(
                    df_w_preds,
                    results_df,
                    how="outer",
                    left_index=True,
                    right_index=True,
                )

            return df_w_preds

        else:
            pt_groups = df_w_preds.groupby(["pt_id", "lead_location"], group_keys=False)
            ar_features = [f"lfp_{hemis[0]}_z_scored_{model}_lag_1"] + (
                ["constant"] if use_constant else []
            )
            hemi1_results_df = pt_groups.apply(
                lambda g: model_utils.apply_sliding_window(
                    g,
                    ar_features,
                    f"lfp_{hemis[0]}_z_scored_{model}",
                    window_size=window_size,
                    causal=causal,
                ),
                include_groups=False,
            )
            ar_features = [f"lfp_{hemis[1]}_z_scored_{model}_lag_1"] + (
                ["constant"] if use_constant else []
            )
            hemi2_results_df = pt_groups.apply(
                lambda g: model_utils.apply_sliding_window(
                    g,
                    ar_features,
                    f"lfp_{hemis[1]}_z_scored_{model}",
                    window_size=window_size,
                    causal=causal,
                ),
                include_groups=False,
            )
    df_w_preds = pd.merge(
        df_w_preds, hemi1_results_df, how="outer", left_index=True, right_index=True
    )
    df_w_preds = pd.merge(
        df_w_preds, hemi2_results_df, how="outer", left_index=True, right_index=True
    )

    return df_w_preds

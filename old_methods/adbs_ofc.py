import numpy as np
import pandas as pd
import glob
from pathlib import Path


def clean(raw_df, patient_name):
    if patient_name == "B017":
        raw_df.query(
            'pt_id != "B017" or source_file != "Report_Json_Session_Report_017_01R_20240516T152238.json"',
            inplace=True,
        )

        # B017's data is labeled differently from other OFC patients --> correct it to match the others.
        right_chest_jsons_017 = list(
            Path("Z:/PerceptOCD-48392/017/LFP/R").glob("[!.]*.json")
        )
        right_chest_jsons_017 = [f.name for f in right_chest_jsons_017]
        right_chest_data_017 = raw_df.query(
            'pt_id == "B017" and source_file in @right_chest_jsons_017'
        ).copy()
        right_chest_data_017[
            [
                "lfp_left",
                "stim_left",
                "lfp_right",
                "stim_right",
                "left_lead_location",
                "right_lead_location",
            ]
        ] = right_chest_data_017[
            [
                "lfp_right",
                "stim_right",
                "lfp_left",
                "stim_left",
                "right_lead_location",
                "left_lead_location",
            ]
        ].copy()
        raw_df.drop(labels=right_chest_data_017.index, inplace=True)
        raw_df = pd.concat([raw_df, right_chest_data_017], ignore_index=True)

    # Correct mislabeled data for OFC patients
    pt = [patient_name]
    ofc_pt_right_ipg_data_vcvs = raw_df.query(
        'pt_id in @pt and left_lead_location == "OTHER" and right_lead_location == "VC/VS"'
    ).copy()
    ofc_pt_right_ipg_data_ofc = ofc_pt_right_ipg_data_vcvs.copy()
    ofc_pt_right_ipg_data_vcvs[["lfp_left", "stim_left"]] = np.nan
    ofc_pt_right_ipg_data_vcvs[["right_lead_location", "left_lead_location"]] = "VC/VS"
    ofc_pt_right_ipg_data_ofc[["lfp_right", "stim_right"]] = ofc_pt_right_ipg_data_ofc[
        ["lfp_left", "stim_left"]
    ].copy()
    ofc_pt_right_ipg_data_ofc[["lfp_left", "stim_left"]] = np.nan
    ofc_pt_right_ipg_data_ofc[["right_lead_location", "left_lead_location"]] = "OFC"

    ofc_pt_left_ipg_data_vcvs = raw_df.query(
        'pt_id in @pt and left_lead_location == "VC/VS" and right_lead_location == "OTHER"'
    ).copy()
    ofc_pt_left_ipg_data_ofc = ofc_pt_left_ipg_data_vcvs.copy()
    ofc_pt_left_ipg_data_vcvs[["lfp_right", "stim_right"]] = np.nan
    ofc_pt_left_ipg_data_vcvs[["right_lead_location", "left_lead_location"]] = "VC/VS"
    ofc_pt_left_ipg_data_ofc[["lfp_left", "stim_left"]] = ofc_pt_left_ipg_data_ofc[
        ["lfp_right", "stim_right"]
    ].copy()
    ofc_pt_left_ipg_data_ofc[["lfp_right", "stim_right"]] = np.nan
    ofc_pt_left_ipg_data_ofc[["right_lead_location", "left_lead_location"]] = "OFC"

    # Finish up the OFC data
    raw_df.drop(labels=ofc_pt_right_ipg_data_vcvs.index, inplace=True)
    raw_df.drop(labels=ofc_pt_left_ipg_data_vcvs.index, inplace=True)
    raw_df = pd.concat(
        [
            raw_df,
            ofc_pt_right_ipg_data_vcvs,
            ofc_pt_left_ipg_data_vcvs,
            ofc_pt_right_ipg_data_ofc,
            ofc_pt_left_ipg_data_ofc,
        ],
        ignore_index=True,
    )
    raw_df.dropna(
        subset=["lfp_left", "lfp_right"], how="all", inplace=True, ignore_index=True
    )
    raw_df.sort_values(
        by=["pt_id", "left_lead_location", "timestamp"], inplace=True, ignore_index=True
    )

    return raw_df

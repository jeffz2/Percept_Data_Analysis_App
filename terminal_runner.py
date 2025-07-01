from generate_raw import generate_raw
from process_data import process_data
from model_data import model_data
from plotting_utils import plot_metrics
import json

def main():
    """_summary_
    Used to run the terminal version of the percept_data analysis app.
    Should be used as a toy exploration of the pipeline, change parameters in patient_info.json to see results.
    """

    with open('patient_info.json', 'r') as f:
        patient_dict = json.load(f)

    pt = patient_dict[0] # Process data from first patient in patient_info.json
    # Generate data based on subject name and parameters
    raw_df, param_changes = generate_raw(pt, patient_dict[pt])

    processed_data = process_data(pt, raw_df, patient_dict[pt])

    df_w_preds = model_data(processed_data)

    # Plot the metrics
    fig = plot_metrics(df_w_preds, pt, 0, param_changes)
    
    fig.show()

if __name__ == '__main__':
    main()
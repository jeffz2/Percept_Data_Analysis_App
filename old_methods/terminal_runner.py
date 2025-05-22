import calc_circadian
from utils import read_json
import plotting_utils as plots
import generate_data

def main():
    """_summary_
    Used to run the terminal version of the percept_data analysis app.
    Should be used as a toy exploration of the pipeline, change parameters in param.json to see different results.
    """
    param_dict = read_json('param.json')

    # Generate data based on subject name and parameters
    percept_data, zone_index = generate_data.generate_data(subject_name=param_dict['subject_name'], param=param_dict)

    # Calculate circadian metrics
    percept_data = calc_circadian.calc_circadian(
        percept_data=percept_data, 
        zone_index=zone_index, 
        cosinor_window_left=int(param_dict['cosinor_window_left']), 
        cosinor_window_right=int(param_dict['cosinor_window_right']), 
        include_nonlinear=param_dict['include_nonlinear']
    )

    # Plot the metrics
    fig = plots.plot_metrics(
        percept_data=percept_data, 
        subject=param_dict['subject_name'], 
        hemisphere=param_dict['hemisphere'], 
        pre_DBS_bounds=param_dict['pre_DBS_example_days'], 
        post_DBS_bounds=param_dict['post_DBS_example_days'], 
        zone_index=zone_index
    )
    fig.show()

if __name__ == '__main__':
    main()
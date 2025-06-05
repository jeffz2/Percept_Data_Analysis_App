# Percept Data Analysis App

## Overview

The Percept Desktop App is designed to provide an intuitive user interface for running the percept-data analysis pipeline for OCD patients, as detailed in this [paper](https://www.nature.com/articles/s41591-024-03125-0). The original program was developed using a combination of MATLAB and Python, as seen in the [PerceptDataAnalysis repository](https://github.com/shethlab/PerceptDataAnalysis). This application translates all the code into Python and uses a Python-based library to create the GUI. The source code and instructions for downloading the app can be found on this GitHub repo

## User Manual

### Key Fields

- `Subject_name`: The codename used to track a specific patient (e.g., '009').
- `Initial DBS programming date`: The date the DBS treatment started, entered in the format YYYY-MM-DD.
- `Responder`: Indicates whether the subject has achieved clinical response as noted by YBOCS criteria (e.g., 'yes' or 'no').
- `Responder_date`: If 'yes' was selected for Responder, provide the date when the patient reached clinical response in the format YYYY-MM-DD or the # of days after the initial DBS programming date.

### Features
- **Plot Metrics**: Displays various physiological and linear AR model metrics. More information can be found in the original paper linked above.
- **Download Plot**: The app can download plots as a variety of different file formats using the "Download Plot" button.
- **Export Data**: The raw linear-AR R2 values can be exported variety of different file formats using the “export linAR button”.

### Data Export Guide

Data exporting is a key feature of the Percept Data Analysis App and was designed to make sharing and analyzing your results as seamless as possible.

#### Download Plot

- You can download plots in various formats, including `html`, `png`, `jpg`, `jpeg`, `webp`, `svg`, and `pdf`.
- When saving a plot, a file dialog will pop up, allowing you to choose a target directory and enter a filename. If you don't specify a file extension, the plot will be saved as `html` by default, enabling interactive viewing in your browser.
- For example, if you name your file `plot_right_008`, it will automatically save as `plot_right_008.html`. To save it in a different format, simply add the desired extension, like `plot_right_008.jpg`.
- If you enter an unsupported extension, such as `plot_right_008.bmp`, the app will still save it as `plot_right_008.html` by default

#### Export Linear AR Feature

- Linear AR features can be exported in `csv`, `xlsx`, `json`, `tsv`, or `txt` formats.
- Similar to plot downloads, when you enter a filename in the file dialog without an extension, it will default to `.csv`.
- If you specify an extension, the file will be saved in the corresponding format, unknown file formats will default `.csv`.

### Installation and Demo Video:
- A demo video showcasing the app can be found [here](https://drive.google.com/file/d/1tWAAfF2GR7SGf6W4wstNonslh4T7LCWn/view).


## Developer Guide

This section is intended for developers looking to modify or extend the functionality of the Percept Data Analysis App. The app is divided into two primary components:

1. **Core Analysis Pipeline**
2. **GUI Interface**

### Core Analysis Pipeline

The Core Analysis Pipeline consists of the following key files:

- `generate_raw.py`: Generates a pandas DataFrame from Medtronic Percept json data files, which is used in subsequent analyses.
- `process_data.py`: Processes the data_struct to clean, normalize, and z-score. Different artifact removing techniques are implemented and can be specified in the app settings.
- `model_data.py`: Analyzes the processed_data DataFrame to perform AR(1) predictions and calculate predictability metrics (r2, residual variance, etc.).
- `plotting_utils.py`: Generates a summary plot encapsulating the most critical information from the processed data_struct.
- `utils.py`: Provides various general helper methods utilized by the above files.
- `json_utils.py`: Provides helper methods to process the Medtronic Percept json data files.
- `state_utils.py`: Provides helper methods to determine clinical states from patient dates.
- `model_utils.py`: Provides helper methods to apply the autoregression to the data.

#### User Defined Hyper-Parameters

To run the Core Analysis Pipeline, users must specify certain hyperparameters, divided into `Required` and `Optional` categories.

**Required Parameters:**

- `subject_name`: A codename for tracking a specific patient (e.g., '009').
- `directory`: A string corresponding to the path of the directory with the subject's JSON files.
- `dbs_date`: The start date of DBS treatment, in the format YYYY-MM-DD.
- `response_status`: The response status of the patient if known.
- `response_date`: The response date of the patient, if applicable, in YYYY-MM-DD format or enter the # of days after DBS activation.
- `disinhibited_dates`: The disinhibited dates of the patient, if applicable, in [start date, end date] format with dates in YYYY-MM-DD format, or enter the # of days after DBS activation

Example values for these parameters are provided in `patient_info.json`.

**Optional Parameters:**

- `window_size`: Window size of data, in days, to train the autoregressive model.
- `outlier_fill_method`: Outlier interpolation method used during processing (Naive, Threshold, Overages).

These parameters can be adjusted in the app's settings menu.

#### Running the Core Analysis Pipeline

Typically, the Core Analysis Pipeline is executed in the following order:

1. `generate_raw.py`
2. `process_data.py`
3. `model_data.py`

For a simple example of running this execution pipeline, refer to `terminal_runner.py`. This file provides a basic script to run the data analysis and display the plots generated by `plotting_utils.py`. To modify the hyperparameters, update them in `patient_info.json`.

For specific implementation details, refer to the documentation and comments within these scripts.

### GUI Interface

The GUI Interface is primarily built using two files: `app.py` and `gui_utils.py`.

- `app.py`: This is the main file responsible for generating the GUI, including widgets, textboxes, and other UI elements.
- `gui_utils.py`: A utility file used by `app.py` to perform tasks such as data export, validation, and transformations.

Documentation for the GUI component is minimal, as it is designed to serve as a flexible abstraction layer for the Core Analysis Pipeline. Developers are encouraged to customize the GUI to fit specific needs. The GUI can be replaced or modified, as long as it can interface with the Core Analysis Pipeline and correctly format the user-defined hyperparameters.

### Building/Compiling the App

If you've made changes to the app's source code and want to see those changes reflected, you need to recompile the app.

#### Windows

The primary mode of distribution should be on Windows due to Apple's distribution regulations.

1. Any GUI changes need to be made in `app_win.py`, which contains special code for Windows compatibility.
2. Ensure PyInstaller is installed (included in `requirements.txt`).
3. Modify `build_win.spec`:
   - Update the `datas` field with the local path to the `kaleido` package:
     ```python
     datas=[
         ('your_path', 'kaleido'),
     ],
     ```
4. Compile the app:
   - Run `pyinstaller "build_win.spec"` in the command line/terminal from your source code directory.
5. Find the generated `.exe` file in the `dist` folder.

Optional: Wrap the `.exe` in an installer application (e.g., Inno Setup) for a smoother download experience.

#### macOS

Note: The macOS build should not be used for distribution due to Apple's app distribution licensing. Follow these steps for local use only:

1. Any GUI changes need to be made in `app.py`, which contains macOS-specific code.
2. Ensure PyInstaller is installed (included in `requirements.txt`).
3. Modify `build_macOS.spec`:
   - Update the `datas` field with the local path to the `kaleido` package:
     ```python
     datas=[
         ('your_path', 'kaleido'),
     ],
     ```
4. Compile the app:
   - Run `pyinstaller "build_macOS.spec"` in the terminal from your source code directory.
5. Find the generated `.app` file in the `dist` folder.
6. Move the `.app` file to your Applications folder to run it natively on your Mac.



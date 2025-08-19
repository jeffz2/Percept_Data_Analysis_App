import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QProgressBar, QMessageBox, QCheckBox, QComboBox, QToolBar, QMainWindow,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QButtonGroup
)
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt, QUrl, QTimer, QSize
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
import generate_raw
import process_data
import model_data
import plotting_utils as plots
import gui_utils
import multiprocessing
import os
import json
import pandas as pd
import numpy as np
from opening_windows import OpeningScreen, HelpMenu, SettingsMenu
from patient_menu import PatientMenu

try:
    from ctypes import windll
    myappid = 'Provenza_Labs.Percept_Data_Analysis App.v2.0'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
LOADING_SCREEN_INTERVAL = 100  # in milliseconds
HEMISPHERE = 'left'

def resource_path(relative_path):
    # Works for development and PyInstaller
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def worker_function(patient_dict, result_queue):
    try:
        df_final = pd.DataFrame()
        pt_changes_df = pd.DataFrame()

        with open(resource_path('data/param.json'), 'r') as f:
            param_dict = json.load(f)

        for pt in patient_dict.keys():
            try:
                raw_df, param_changes = generate_raw.generate_raw(pt, patient_dict[pt])

            except TypeError or ValueError as e:
                print(f"Unable to retrieve data for pateint {pt}")
                continue

            processed_data = process_data.process_data(pt, raw_df, patient_dict[pt], ark=param_dict['ark'], max_lag=param_dict['lags'] if param_dict['ark'] else 1)

            df_w_preds = model_data.model_data(processed_data, use_constant=False if not param_dict['ark'] else True, ark=param_dict['ark'], max_lag=param_dict['lags'] if param_dict['ark'] else 1)

            pt_changes_df = pd.concat([pt_changes_df, param_changes], ignore_index=True)

            df_final = pd.concat([df_final, df_w_preds], ignore_index=True)

        result_queue.put((df_final, pt_changes_df))

    except Exception as e:
        print(f"Error in worker_function: {e}")
        result_queue.put(None)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.result_queue = multiprocessing.Queue()
        self.worker_process = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_for_results)

    def initUI(self):
        self.setWindowTitle("Percept Data App")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.stack = QWidget(self)
        self.layout = QVBoxLayout(self.stack)
        self.setLayout(self.layout)

        self.opening_screen = OpeningScreen(self)
        self.layout.addWidget(self.opening_screen)

        self.loading_screen = LoadingScreen(self)
        self.layout.addWidget(self.loading_screen)
        self.loading_screen.hide()

        self.patient_menu = PatientMenu(self)
        self.layout.addWidget(self.patient_menu)
        self.patient_menu.hide()

        self.help_menu = HelpMenu(self)
        self.layout.addWidget(self.help_menu)
        self.help_menu.hide()

        self.settings_menu = SettingsMenu(self)
        self.layout.addWidget(self.settings_menu)
        self.settings_menu.hide()

        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
                color: #f5f5f5;
                font-family: Arial, sans-serif;
            }
            QLabel {
                color: #f5f5f5;
                font-size: 14px;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #3d3d3d;
                color: #f5f5f5;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #1e90ff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1c86ee;
            }
            QComboBox QAbstractItemView {
                background-color: #3d3d3d;
                color: #f5f5f5;
                selection-background-color: #1e90ff;
            }
        """)

    def show_loading_screen(self, patient_dict):
        self.loading_screen.show()
        self.loading_screen.progress_bar.setRange(0, 0)
        self.opening_screen.hide()

        self.patient_dict = patient_dict
        self.worker_process = multiprocessing.Process(
            target=worker_function,
            args=(self.patient_dict, self.result_queue)
        )
        self.worker_process.start()
        self.timer.start(LOADING_SCREEN_INTERVAL)

    def check_for_results(self):
        try:
            if not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                if result is None:
                    self.on_script_finished(None, None)
                else:
                    df_final, pt_changes_df = result
                    self.worker_process.terminate()
                    self.on_script_finished(df_final, pt_changes_df)
        except Exception as e:
            print(f"Error checking results: {e}")

    def on_script_finished(self, df_final=None, pt_changes_df=None):
        self.loading_screen.hide()
        if not df_final.empty:
            self.show_plots(df_final, pt_changes_df)
        else:
            QMessageBox.warning(self, "Error", "Failed to process the data. Please try again.")
            self.show_opening_screen()

    def show_plots(self, df_final, pt_changes_df):
        self.setGeometry(100, 100, 1200, 800)
        self.plots = Plots(self, df_final, pt_changes_df)
        self.layout.addWidget(self.plots)
        self.plots.show()

    def show_opening_screen(self):
        self.opening_screen.show()
        self.loading_screen.hide()
        self.hide_all_menus()

    def show_help_menu(self):
        self.hide_all_menus()
        self.opening_screen.hide()
        self.help_menu.show()

    def show_settings_menu(self):
        self.hide_all_menus()
        self.opening_screen.hide()
        self.settings_menu.show()

    def show_patient_menu(self):
        self.hide_all_menus()
        self.opening_screen.hide()
        self.patient_menu.show()

    def hide_all_menus(self):
        self.help_menu.hide()
        self.settings_menu.hide()
        self.patient_menu.hide()

class LoadingScreen(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.label = QLabel(
            "The application is processing your data.\nPlease wait a moment, this may take a couple of minutes.\nDo not close or restart the application.", 
            self
        )
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-family: 'Arial', sans-serif;
                color: #ffffff;
                padding: 20px;
            }
        """)
        self.layout.addWidget(self.label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #1e90ff;
                width: 20px;
            }
        """)
        self.layout.addWidget(self.progress_bar)

        self.setStyleSheet("""
            background-color: #2d2d2d;
        """)

        self.setLayout(self.layout)

class Plots(QWidget):
    def __init__(self, parent,df_final, pt_changes_df):
        super().__init__(parent)
        self.parent = parent
        with open(resource_path('data/param.json'), 'r') as f:
            self.param_dict = json.load(f)
        self.df_final = df_final
        self.pt_changes_df = pt_changes_df
        self.patients = np.unique(df_final['pt_id'])
        self.curr_pt = self.patients[0]
        with open('data/patient_info.json') as f:
            self.patient_dict = json.load(f)
        self.current_plot = None
        self.web_view = QWebEngineView(self)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.content_layout = QHBoxLayout()

        self.init_json_frame()
        self.init_plot_frame()

        self.layout.addLayout(self.content_layout)
        self.init_bottom_buttons()
        self.setLayout(self.layout)
        self.update_plot(self.curr_pt)

    def init_json_frame(self, index=0):
        self.json_fields_frame = QWidget(self)
        self.json_layout = QVBoxLayout(self.json_fields_frame)
        self.json_text = QTextEdit(self.json_fields_frame)
        self.json_text.setReadOnly(True)
        self.json_text.setStyleSheet("""
            background-color: #4d4d4d;
            color: #f5f5f5;
            border: 1px solid #555;
            border-radius: 5px;
            padding: 10px;
            font-size: 14px;
            font-family: 'Roboto', sans-serif;
        """)
        self.init_patient_selector(index)
        self.json_layout.addWidget(self.json_text)

        self.update_json_fields(self.patients[index])

        self.changes_checkbox = QCheckBox("Show Parameter Changes", self)
        self.changes_checkbox.setChecked(False)
        self.changes_checkbox.stateChanged.connect(self.plot_param_change)
        self.json_layout.addWidget(self.changes_checkbox, alignment=Qt.AlignCenter | Qt.AlignBottom)

        self.export_button = QPushButton("Export LinAR R² feature", self)
        self.export_button.clicked.connect(self.export_data)
        self.json_layout.addWidget(self.export_button, alignment=Qt.AlignCenter | Qt.AlignBottom)

        self.json_fields_frame.setLayout(self.json_layout)
        self.content_layout.addWidget(self.json_fields_frame, 2)

    def plot_param_change(self):
        self.update_plot(self.curr_pt, HEMISPHERE, self.changes_checkbox.isChecked())

    def init_patient_selector(self, index):
        self.patient_selector = QComboBox(self)
        self.patient_selector.addItems(self.patients)
        self.patient_selector.setCurrentIndex(index)
        self.patient_selector.currentIndexChanged.connect(self.patient_change)

    def update_json_fields(self, patient):
        pt_df = self.df_final.query('pt_id == @patient')
        self.json_text.clear()
        self.json_text.append(f"Subject_name: {patient}\n")
        self.json_text.append(f"Initial DBS programming:\n {self.patient_dict[patient]['dbs_date']}\n")
        self.json_text.append(f"Total samples: {len(pt_df.index)}\n")
        self.json_text.append(f"Total days: {len(np.unique(pt_df['days_since_dbs']))}\n")
        if(self.patient_dict[self.curr_pt]['response_status'] == 1):
            self.json_text.append(f"Responder on {self.patient_dict[patient]['response_date']}\n")
        else:
            self.json_text.append(f"Non-responder\n")

    def init_plot_frame(self):
        self.web_view.setFixedSize(900, 650)
        self.configure_web_view()

        self.hemisphere_selector_layout = QHBoxLayout()
        self.hemisphere_selector_layout.addStretch()
        self.init_hemisphere_selector()
        self.hemisphere_selector_layout.addWidget(self.hemisphere_selector)
        self.hemisphere_selector_layout.addSpacing(10)

        self.plot_layout = QVBoxLayout()
        self.plot_layout.addLayout(self.hemisphere_selector_layout)
        self.plot_layout.addWidget(self.web_view)

        self.content_layout.addLayout(self.plot_layout, 8)

    def configure_web_view(self):
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

    def init_hemisphere_selector(self):
        self.hemisphere_selector = QComboBox(self)
        self.hemisphere_selector.addItems(["Left Hemisphere", "Right Hemisphere"])
        self.hemisphere_selector.setCurrentIndex(0)
        self.hemisphere_selector.currentIndexChanged.connect(self.on_hemisphere_change)

    def init_bottom_buttons(self):
        self.button_layout = QHBoxLayout()

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)
        self.button_layout.addWidget(self.back_button, alignment=Qt.AlignLeft | Qt.AlignBottom)

        self.download_button = QPushButton("Download plot", self)
        self.download_button.clicked.connect(self.download_image)
        self.button_layout.addWidget(self.download_button, alignment=Qt.AlignRight | Qt.AlignBottom)

        self.layout.addLayout(self.button_layout)

    def patient_change(self, index):
        patient = self.patients[index]
        self.curr_pt = patient
        self.update_json_fields(self.curr_pt)
        self.update_plot(self.curr_pt, HEMISPHERE, self.changes_checkbox.isChecked())

    def on_hemisphere_change(self, index):
        HEMISPHERE = 'left' if index == 0 else 'right'
        self.update_plot(self.curr_pt, HEMISPHERE, self.changes_checkbox.isChecked())

    def update_plot(self, patient, hemisphere = 'left', show_changes=False):
        fig = plots.plot_metrics(
            df=self.df_final,
            patient=patient,
            hemisphere=hemisphere,
            changes_df=self.pt_changes_df,
            show_changes=show_changes
        )

        self.current_plot = fig

        temp_file_path = gui_utils.create_temp_plot(fig)

        self.web_view.setUrl(QUrl.fromLocalFile(temp_file_path))

    def go_back(self):
        self.hide()
        self.parent.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.parent.show_opening_screen()

    def download_image(self):
        file_path = gui_utils.open_save_dialog(self, "Save Image", "")
        if file_path and self.current_plot:
            gui_utils.save_plot(self.current_plot, file_path)
        else:
            QMessageBox.warning(self, "Error", "No plot is available to save.")

    def export_data(self):
        file_path = gui_utils.open_save_dialog(self, "Save Data", "")
        if file_path:
            data = self.df_final.query('pt_id == @self.curr_pt')
            gui_utils.save_lin_ar_feature(data, file_path, self.param_dict)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    multiprocessing.freeze_support()
    #basedir = os.path.dirname(__file__)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icons/Icon.ico'))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

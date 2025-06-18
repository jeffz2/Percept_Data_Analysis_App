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
from pathlib import Path
import json
import pandas as pd
import numpy as np

try:
    from ctypes import windll
    myappid = 'Provenza_Labs.Percept_Data_Analysis App.v1.0'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
LOADING_SCREEN_INTERVAL = 100  # in milliseconds

def worker_function(patient_dict, result_queue):
    try:
        df_final = pd.DataFrame()
        pt_changes_df = pd.DataFrame()

        for pt in patient_dict.keys():
            raw_df, param_changes = generate_raw.generate_raw(pt, patient_dict[pt])

            processed_data = process_data.process_data(pt, raw_df, patient_dict[pt])

            df_w_preds = model_data.model_data(processed_data)

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
        if not df_final.empty and not pt_changes_df.empty:
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


class OpeningScreen(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.welcome_label = QLabel("Welcome to the Percept Data Analysis App", self)
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setStyleSheet("""
            QLabel {
                font-size: 25px;
                font-family: 'Arial', sans-serif;
                color: #ffffff;
                padding: 10px;
            }
        """)
        self.layout.addWidget(self.welcome_label)

        self.description_label = QLabel(
            'This application helps you process and analyze Medtronic percept data.<br>'
            'Please proceed to start the data processing.<br><br>'
            'Developed by the Provenza Lab', self)
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-family: 'Arial', sans-serif;
                color: #ffffff;
                padding: 20px;
            }
        """)
        self.layout.addWidget(self.description_label)

        self.proceed_button = QPushButton("Start Data Processing", self)
        self.proceed_button.setStyleSheet("""
            QPushButton {
                background-color: #1e90ff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #1c86ee;
            }
        """)
        self.proceed_button.clicked.connect(self.proceed)
        self.layout.addWidget(self.proceed_button, alignment=(Qt.AlignHCenter))

        self.patient_menu_button = QPushButton("Add Patients", self)
        self.patient_menu_button.setStyleSheet("""
            QPushButton {
                background-color: #1e90ff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #1c86ee;
            }
        """)
        
        self.patient_menu_button.clicked.connect(self.parent.show_patient_menu)
        self.layout.addWidget(self.patient_menu_button, alignment=(Qt.AlignHCenter))

        def open_url(url):
            """Opens the given URL in the default web browser."""
            qurl = QUrl(url)
            QDesktopServices.openUrl(qurl)

        toolbar = QToolBar("Main Window Toolbar")
        toolbar.setIconSize(QSize(30, 30))
        self.layout.addWidget(toolbar)

        doc_button = QAction(QIcon("doc_icon.png"), "See GitHub documentation of the app", self)
        doc_button.setStatusTip("See GitHub Documentation of the app")
        doc_button.triggered.connect(lambda: open_url("https://github.com/jeffz2/Percept_Data_Analysis_App/blob/percept_2025_dev/README.md"))
        toolbar.addAction(doc_button)

        help_button = QAction(QIcon("help_icon.png"), "How to use the app", self)
        help_button.setStatusTip("For a step-by-step guide to use the app")
        help_button.triggered.connect(self.parent.show_help_menu)
        toolbar.addAction(help_button)

        settings_button = QAction(QIcon("settings_icon.png"), "Processing settings", self)
        settings_button.setStatusTip("Processing settings")
        settings_button.triggered.connect(self.parent.show_settings_menu)
        toolbar.addAction(settings_button)

    def proceed(self):
        if not os.path.exists("ocd_patient_info.json"):
            return WindowsError
        with open("ocd_patient_info.json", 'r') as f:
            try:
                patient_dict = json.load(f)
            except json.JSONDecodeError:
                QMessageBox.warning(self, "No patient data is stored")
                return
        self.parent.show_loading_screen(patient_dict)

class HelpMenu(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.help_label = QLabel(
            'Write a step-by-step guide to use the app.<br>'
            'Step 1: Add patients with dbs on date, folder containing the data, and responder status.<br><br>'
            'Step 2: Do the data processing and see results', self)
        self.help_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.help_label)

        self.init_bottom_buttons()
        self.setLayout(self.layout)

    def go_back(self):
        self.hide()
        self.window().show_opening_screen()

    def init_bottom_buttons(self):
        self.button_layout = QHBoxLayout()

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)
        self.button_layout.addWidget(self.back_button, alignment=Qt.AlignLeft | Qt.AlignBottom)

        self.layout.addLayout(self.button_layout)

class SettingsMenu(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.field_order = [
            "Window size"
        ]
        # TODO: Implement getting param setttings from json to display as default
        self.fields = {
            "Window size": 3
        }
        
        self.tooltips = self.get_tooltips()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.entries = {}

        self.create_model_type_checkbox()
        self.create_field_entries()

        self.init_bottom_buttons()
        self.setLayout(self.layout)

    def create_field_entries(self):
        for key in self.field_order:
            value = self.fields[key]
            hbox = QHBoxLayout()
            label = QLabel(key, self)
            hbox.addWidget(label)

            if isinstance(value, list):
                self.create_list_field_entries(key, value, hbox)
            else:
                entry = QLineEdit(self)
                entry.setText(str(value))
                entry.setToolTip(self.tooltips[key])
                hbox.addWidget(entry)
                self.entries[key] = entry

            self.layout.addLayout(hbox)

    def create_list_field_entries(self, key, value, layout):
        entry1 = QLineEdit(self)
        entry1.setText(str(value[0]))
        entry1.setToolTip(self.tooltips[key])
        layout.addWidget(entry1)
        self.entries[key] = (entry1)
    
    def create_model_type_checkbox(self):
        dialog = QDialog(self)
        
        self.model_label = QLabel("Model Type ", self)
        self.naive_checkbox = QCheckBox("Naive", self)
        self.threshold_checkbox = QCheckBox("Threshold", self)
        self.overage_checkbox = QCheckBox("Overage", self)

        checkbox_group = QButtonGroup(dialog)
        checkbox_group.setExclusive(True)
        checkbox_group.addButton(self.naive_checkbox)
        checkbox_group.addButton(self.threshold_checkbox)
        checkbox_group.addButton(self.overage_checkbox)

        self.naive_checkbox.setToolTip(self.tooltips["naive"])
        self.threshold_checkbox.setToolTip(self.tooltips["threshold"])
        self.overage_checkbox.setToolTip(self.tooltips["overage"])

        self.overage_checkbox.setChecked(True)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.naive_checkbox)
        model_layout.addSpacing(15)
        model_layout.addWidget(self.threshold_checkbox)
        model_layout.addSpacing(15)
        model_layout.addWidget(self.overage_checkbox)
        model_layout.addStretch()

        self.layout.addLayout(model_layout)

    def get_tooltips(self):
        return {
            "naive": "Naive outlier removal method description",
            "threshold": "Threshold outlier removal method description",
            "overage": "Overage outlier removal method description",
            "Window size": "Window size to train the autoregressive model on"
        }
    
    def validate_fields(self):
        if not self.entries['Window size'].text():
            QMessageBox.warning(self, "Invalid Input", "Window size must be filled in")
            return False
        try:
            tmp = int(self.entries['Window size'].text())
        except Exception or tmp <= 0:
            QMessageBox.warning(self, "Invalid Input", "Window size must be an integer > 0")
            return False
        if not self.naive_checkbox.isChecked() and not self.threshold_checkbox.isChecked() and not self.overage_checkbox.isChecked():
            QMessageBox.warning(self, "Invalid Input", "No overage handling method is checked")
            return False
        
        return True
        
    def go_back(self):
        self.hide()
        self.window().show_opening_screen()

    def set_default_settings(self):
        self.entries["Window size"].setText("3")

        self.naive_checkbox.setChecked(False)
        self.threshold_checkbox.setChecked(False)
        self.overage_checkbox.setChecked(True)
    
    def save_settings(self):
        if not self.validate_fields():
            return
        
        param_dict = {}
        param_dict['hemisphere'] = 'left'
        for key, entry in self.entries.items():
            if key == "Window size":
                param_dict[key] = int(entry.text())
            param_dict[key] = entry.text()

        if self.naive_checkbox.isChecked():
            param_dict['model'] = "naive"
        
        elif self.threshold_checkbox.isChecked():
            param_dict['model'] = "SLOvER+"

        elif self.overage_checkbox.isChecked():
            param_dict['model'] = "OvER"
        
        try:
            with open("param.json", 'w') as f:
                json.dump(param_dict, f, indent=4)
                f.close()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save settings: {e}")
            return

        self.hide()
        self.window().show_opening_screen()

    def init_bottom_buttons(self):
        self.button_layout = QHBoxLayout()

        self.back_button = QPushButton("Back", self)
        self.back_button.clicked.connect(self.go_back)
        self.button_layout.addWidget(self.back_button, alignment=Qt.AlignLeft | Qt.AlignBottom)

        self.default_button = QPushButton("Reset to Default", self)
        self.default_button.clicked.connect(self.set_default_settings)
        self.button_layout.addWidget(self.default_button, alignment=Qt.AlignCenter | Qt.AlignBottom)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_settings)
        self.button_layout.addWidget(self.save_button, alignment=Qt.AlignRight | Qt.AlignBottom)

        self.layout.addLayout(self.button_layout)

class PatientMenu(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.field_order = [
            "Patient ID",
            "directory",
            "dbs_date",
            "response_status",
            "response_date",
            "disinhibited_dates"
        ]
        self.tooltips = self.get_tooltips()
        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout(self)

        self.table_layout = QVBoxLayout()
        self.main_layout.addLayout(self.table_layout)

        self.load_patients_table()
        self.init_bottom_buttons()

        self.setLayout(self.main_layout)

    def load_patients_table(self):
        self.table = QTableWidget(self)

        display_fields = ["Patient ID", "Directory", "Response Status"]
        display_keys = {"Directory": "directory", "Response Status": "response_status"}
        display_response = {0: "Non-responder", 1: "Responder"}
        self.table.setColumnCount(len(display_fields))
        self.table.setHorizontalHeaderLabels(display_fields)

        patients = self.load_patient_data()
        self.table.setRowCount(len(patients))

        for row, patient in enumerate(patients.keys()):
            for col, key in enumerate(display_fields):
                if key == "Patient ID":
                    self.table.setItem(row, col, QTableWidgetItem(patient))
                elif key == "Response Status":
                    response_status = display_response[patients[patient][display_keys[key]]]
                    self.table.setItem(row, col, QTableWidgetItem(response_status))
                else:
                    self.table.setItem(row, col, QTableWidgetItem(patients[patient][display_keys[key]]))

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_layout.addWidget(self.table)


    def load_patient_data(self):
        if not os.path.exists("ocd_patient_info.json"):
            return []
        with open("ocd_patient_info.json", 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def go_back(self):
        self.hide()
        self.window().show_opening_screen()

    def refresh_table(self):
        # Remove existing table from layout
        if hasattr(self, 'table'):
            self.table_layout.removeWidget(self.table)
            self.table.deleteLater()
            self.table = None

        self.load_patients_table()


    def add_patient(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Patient")
        layout = QVBoxLayout(dialog)

        form_entries = {}

        key_labels = {"Patient ID": "Patient ID",
                      "directory": "Directory",
                      "dbs_date": "DBS Date",
                      "response_status": "Response Status",
                      "response_date": "Response Date",
                      "disinhibited_dates": "Disinhibited Dates"}

        for key in self.field_order:
            if key == "response_status":
                hbox = QHBoxLayout()
                label = QLabel(key_labels[key])
                response_checkbox = QCheckBox("Responder")
                non_response_checkbox = QCheckBox("Non-responder")

                checkbox_group = QButtonGroup(dialog)
                checkbox_group.setExclusive(True)
                checkbox_group.addButton(response_checkbox)
                checkbox_group.addButton(non_response_checkbox)

                hbox.addWidget(label)
                hbox.addWidget(response_checkbox)
                hbox.addWidget(non_response_checkbox)
                layout.addLayout(hbox)
            elif key == "response_date":
                response_date_layout = QHBoxLayout()
                response_date_label = QLabel(key_labels[key])
                response_date_entry = QLineEdit()
                response_date_layout.addWidget(response_date_label)
                response_date_layout.addWidget(response_date_entry)
                layout.addLayout(response_date_layout)
                # Hide initially
                response_date_label.hide()
                response_date_entry.hide()
                response_date_entry.setToolTip(self.tooltips[key])
            else:
                hbox = QHBoxLayout()
                label = QLabel(key_labels[key])
                entry = QLineEdit()
                hbox.addWidget(label)
                hbox.addWidget(entry)
                layout.addLayout(hbox)
                form_entries[key] = entry
                entry.setToolTip(self.tooltips[key])
        
        def toggle_response_checkbox():
            if response_checkbox.isChecked():
                response_date_label.show()
                response_date_entry.show()
        
            if non_response_checkbox.isChecked():
                response_date_label.hide()
                response_date_entry.hide()

        def save_and_close():
            pt_dict = {}
            patient = form_entries["Patient ID"].text()
            pt_dict[patient] = {}
            for key in self.field_order:
                if key == "Patient ID":
                    continue
                if key == "response_status":
                    pt_dict[patient][key] = 1 if response_checkbox.isChecked() else 0
                elif key == "response_date":
                    if response_checkbox.isChecked():
                        pt_dict[patient][key] = response_date_entry.text()
                elif key == "directory":
                    pt_dict[patient][key] = form_entries[key].text()[1:-1]
                else:
                    if form_entries[key].text() == "":
                        continue
                    pt_dict[patient][key] = form_entries[key].text()

            if not pt_dict[patient] or "directory" not in pt_dict[patient].keys():
                QMessageBox.warning(dialog, "Validation Error", "Patient ID and Directory are required.")
                return
            
            if not Path(pt_dict[patient]['directory']).is_dir():
                QMessageBox.warning(dialog, "Validation Error", "Path is not a valid directory")
                return

            patients = self.load_patient_data()

            if patient in patients.keys():
                QMessageBox.warning(dialog, "Validation Error", "Patient ID is already in the app database")
                return 
            
            if not gui_utils.validate_date(pt_dict[patient]['dbs_date']):
                QMessageBox.warning(dialog, "Validation Error", "DBS activation date is required in YYYY-MM-DD format.")
                return
            
            if response_checkbox.isChecked() and not gui_utils.validate_date(pt_dict[patient]['response_date']):
                try:
                    pt_dict[patient]['response_date'] = int(pt_dict[patient]['response_date'])
                except Exception:
                    QMessageBox.warning(dialog, "Validation Error", "Response date is required in YYYY-MM-DD format if patient is a responder.")
                    return

            patients.update(pt_dict)

            with open("ocd_patient_info.json", 'w') as f:
                json.dump(patients, f, indent=4)

            dialog.accept()
            self.refresh_table()
            dialog.hide()

        response_checkbox.stateChanged.connect(toggle_response_checkbox)
        non_response_checkbox.stateChanged.connect(toggle_response_checkbox)

        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(save_and_close)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)

        dialog.exec()

    def delete_patient(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Delete Patient")
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Enter Patient ID to delete:"))
        patient_id_entry = QLineEdit()
        layout.addWidget(patient_id_entry)

        def delete_and_close():
            patient_id = patient_id_entry.text().strip()
            if not patient_id:
                QMessageBox.warning(dialog, "Input Error", "Please enter a Patient ID.")
                return

            patients = self.load_patient_data()
            
            if patient_id not in patients.keys():
                QMessageBox.warning(dialog, "Not Found", f"No patient found with ID: {patient_id}")
                return

            del patients[patient_id]
            with open("ocd_patient_info.json", 'w') as f:
                json.dump(patients, f, indent=4)

            dialog.accept()
            self.refresh_table()

        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(delete_and_close)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(delete_button)

        layout.addLayout(button_layout)

        dialog.exec()

    def init_bottom_buttons(self):
        button_layout = QHBoxLayout()

        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.go_back)
        button_layout.addWidget(back_button, alignment=Qt.AlignLeft)

        delete_button = QPushButton("Delete patient", self)
        delete_button.clicked.connect(self.delete_patient)
        button_layout.addWidget(delete_button, alignment=Qt.AlignCenter)

        add_button = QPushButton("Add patient", self)
        add_button.clicked.connect(self.add_patient)
        button_layout.addWidget(add_button, alignment=Qt.AlignRight)

        self.main_layout.addLayout(button_layout)

    def get_tooltips(self):
        return {
            "Patient ID": "Unique patient identifier.",
            "directory": "Directory where patient data is stored wrapped in quotes (Tip: Use CTRL-SHIFT-C on a highlighted folder to copy the path to your clipboard).",
            "dbs_date": "Initial DBS programming date (YYYY-MM-DD format).",
            "response_status": "Responder status (Yes/No).",
            "response_date": "Date the patient became a responder (Enter YYYY-MM-DD format or # of days post-DBS patient achieved response).",
            "disinhibited_dates": "Dates the patient was disinhibited in [start date, end date] format (Enter dates in YYYY-MM-DD format or post-DBS day range patient was disinhibited)."
        }

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
        with open('param.json', 'r') as f:
            self.param_dict = json.load(f)
        self.df_final = df_final
        self.pt_changes_df = pt_changes_df
        self.patients = np.unique(df_final['pt_id'])
        self.curr_pt = self.patients[0]
        self.current_plot = None
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

    def init_json_frame(self):
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
        self.init_patient_selector()
        self.json_layout.addWidget(self.json_text)

        self.populate_json_fields(0)

        self.export_button = QPushButton("Export LinAR R² feature", self)
        self.export_button.clicked.connect(self.export_data)
        self.json_layout.addWidget(self.export_button, alignment=Qt.AlignCenter | Qt.AlignBottom)

        self.json_fields_frame.setLayout(self.json_layout)
        self.content_layout.addWidget(self.json_fields_frame, 2)

    def init_patient_selector(self):
        self.patient_selector = QComboBox(self)
        self.patient_selector.addItems(self.patients)
        self.patient_selector.setCurrentIndex(0)
        self.patient_selector.currentIndexChanged.connect(self.on_patient_change(self.curr_pt))

    def populate_json_fields(self, patient):
        pt_df = self.df_final.query('pt_id == @patient')
        self.json_text.append(f"Subject_name: {patient}\n")
        self.json_text.append(f"Initial_DBS_programming_date: {pt_df.query('days_since_dbs == 0')['timestamp'].head(1)}\n")
        self.json_text.append(f"Total samples: {len(pt_df.index)}\n")
        self.json_text.append(f"Total days: {len(np.unique(pt_df['days_since_dbs']))}\n")
        if(3 in pt_df['state_label']):
            self.json_text.append(f"Responder: {True}\n")
        else:
            self.json_text.append(f"Responder: {False}\n")

    def init_plot_frame(self):
        self.web_view = QWebEngineView(self)
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

    def on_patient_change(self, patient):
        self.update_plot(patient)
        self.populate_json_fields(patient)
        self.curr_pt = patient

    def on_hemisphere_change(self, index):
        self.param_dict['hemisphere'] = index
        self.update_plot(self.curr_pt)

    def update_plot(self, patient):
        fig = plots.plot_metrics(
            df=self.df_final,
            patient=patient,
            hemisphere=self.param_dict['hemisphere'],
            changes_df=self.pt_changes_df
        )

        self.current_plot = fig

        temp_file_path = gui_utils.create_temp_plot(fig)

        self.web_view.setUrl(QUrl.fromLocalFile(temp_file_path))

    def go_back(self):
        self.hide()
        self.parent.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.parent.frame1.show()

    def download_image(self):
        file_path = gui_utils.open_save_dialog(self, "Save Image", "")
        if file_path and self.current_plot:
            gui_utils.save_plot(self.current_plot, file_path)
        else:
            QMessageBox.warning(self, "Error", "No plot is available to save.")

    def export_data(self):
        file_path = gui_utils.open_save_dialog(self, "Save Data", "")
        if file_path:
            lin_ar_df = gui_utils.prepare_export_data(self.percept_data, self.param_dict)
            gui_utils.save_lin_ar_feature(lin_ar_df, file_path)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    multiprocessing.freeze_support()
    #basedir = os.path.dirname(__file__)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('Icon.ico'))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

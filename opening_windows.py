from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QProgressBar, QMessageBox, QCheckBox, QComboBox, QToolBar, QMainWindow,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QButtonGroup
)
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt, QUrl, QTimer, QSize
from PySide6.QtGui import QDesktopServices
import sys
import os
import json
from utils import resource_path

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

        doc_button = QAction(QIcon(resource_path("icons/doc_icon.ico")), "See GitHub documentation of the app", self)
        doc_button.setStatusTip("See GitHub Documentation of the app")
        doc_button.triggered.connect(lambda: open_url("https://github.com/jeffz2/Percept_Data_Analysis_App/blob/percept_2025_dev/README.md"))
        toolbar.addAction(doc_button)

        help_button = QAction(QIcon(resource_path("icons/help_icon.ico")), "How to use the app", self)
        help_button.setStatusTip("For a step-by-step guide to use the app")
        help_button.triggered.connect(self.parent.show_help_menu)
        toolbar.addAction(help_button)

        settings_button = QAction(QIcon(resource_path("icons/settings_icon.ico")), "Processing settings", self)
        settings_button.setStatusTip("Processing settings")
        settings_button.triggered.connect(self.parent.show_settings_menu)
        toolbar.addAction(settings_button)

    def proceed(self):
        if not os.path.exists(resource_path("data/patient_info.json")):
            return WindowsError
        with open(resource_path("data/patient_info.json"), 'r') as f:
            try:
                patient_dict = json.load(f)
            except json.JSONDecodeError:
                QMessageBox.warning(self, "Validation Error", "No patient data is stored")
                return
        if len(patient_dict) == 0:
            QMessageBox.warning(self, "Validation Error", "No patient data is stored")
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
        self.create_delta_checkbox()
        self.create_ark_checkbox()

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
    
    def create_delta_checkbox(self):
        self.delta_label = QLabel("Delta Normalize R²")
        self.delta_checkbox = QCheckBox()
        self.delta_checkbox.setToolTip("Normalize R² value with pre-DBS average. Will revert to original R² values if no pre-DBS data is available.")

        self.delta_checkbox.setChecked(False)

        delta_layout = QHBoxLayout()
        delta_layout.addWidget(self.delta_label)
        delta_layout.addWidget(self.delta_checkbox)

        self.layout.addLayout(delta_layout)

    def create_ark_checkbox(self):
        self.ark_label = QLabel("AR(k) Model")
        self.ark_checkbox = QCheckBox()
        self.ark_checkbox.setToolTip("Use an AR(k) model to predict LFP data. Default model is an AR(1) model.")

        self.lag_label = QLabel("Lags")
        self.lag_entry = QLineEdit(self)
        self.lag_entry.setText('144')

        self.ark_checkbox.setChecked(False)
        self.ark_checkbox.stateChanged.connect(self.toggle_lags)

        ark_layout = QHBoxLayout()
        ark_layout.addWidget(self.ark_label)
        ark_layout.addWidget(self.ark_checkbox)
        ark_layout.addSpacing(15)
        ark_layout.addWidget(self.lag_label)
        ark_layout.addWidget(self.lag_entry)
        self.lag_label.hide()
        self.lag_entry.hide()

        self.layout.addLayout(ark_layout)

    def toggle_lags(self):
        if self.ark_checkbox.isChecked():
            self.lag_label.show()
            self.lag_entry.show()
        else:
            self.lag_label.hide()
            self.lag_entry.hide()

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
        if self.ark_checkbox.isChecked():
            try:
                tmp = int(self.lag_entry.text())
            except Exception or tmp <= 0:
                QMessageBox.warning(self, "Invalid Input", "Lags must an integer greater than 0")
                return
        return True
        
    def go_back(self):
        self.hide()
        self.window().show_opening_screen()

    def set_default_settings(self):
        self.entries["Window size"].setText("3")

        self.naive_checkbox.setChecked(False)
        self.threshold_checkbox.setChecked(False)
        self.overage_checkbox.setChecked(True)

        self.delta_checkbox.setChecked(False)
        self.ark_checkbox.setChecked(False)
    
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
        
        param_dict['delta'] = 1 if self.delta_checkbox.isChecked() else 0

        param_dict['ark'] = 1 if self.ark_checkbox.isChecked() else 0

        param_dict['lags'] = int(self.lag_entry.text()) if self.ark_checkbox.isChecked() else False

        try:
            with open(resource_path("data/param.json"), 'w') as f:
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
#Steps to take before running this Application.
#Before running this program make sure to install the required libraries using pip or you can copy 5th line and paste the line below in your terminal.
# to run this file in Linux copy 4th line
# python3 Data_visualizer.py
# pip install pandas numpy PyQt5 seaborn matplotlib scipy pytz geocoder pvlib queue
#Now Ready to Go.



import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QDoubleSpinBox, QRadioButton,
    QGroupBox, QDateEdit, QSpinBox
)
from PyQt5.QtCore import Qt, QDate
import pandas as pd
import numpy as np
from dateutil import parser
import pytz
import re
import geocoder
from pvlib.location import Location
from pvlib import solarposition, atmosphere
import threading
from queue import Queue

# Regex pattern to match valid data lines
pattern = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{1,2}:\d{1,2})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
)

class SolarProcessorWorker(threading.Thread):
    """
    A worker thread to process a subset of solar data.
    """
    def __init__(self, group_data, latitude, longitude, altitude, timezone_str, is_online_mode, langley_results, result_queue):
        super().__init__()
        self.group_data = group_data
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone_str = timezone_str
        self.is_online_mode = is_online_mode
        self.langley_results = langley_results # Contains slopes, intercepts, and total optical depths
        self.result_queue = result_queue
        self.error = None # To store any errors encountered in the thread

        if self.is_online_mode:
            try:
                self.fixed_location = Location(latitude=self.latitude, longitude=self.longitude,
                                               tz=self.timezone_str, altitude=self.altitude)
            except Exception as e:
                self.error = f"Error initializing PVlib location in thread: {e}"
                self.fixed_location = None

    def _calculate_solar_position_custom(self, datetime_obj, latitude, longitude, altitude, timezone_str):
        """
        Calculates solar zenith angle (degrees and radians) and relative air mass
        using custom formulas, without pvlib. This is used in Offline Mode.
        Moved from main class to worker for encapsulation.
        """
        try:
            tz = pytz.timezone(timezone_str)
            datetime_obj_localized = tz.localize(datetime_obj)

            doy = datetime_obj_localized.timetuple().tm_yday
            gamma = 2 * np.pi / 365 * (doy - 1)

            declination_rad = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) -
                               0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma) -
                               0.002697 * np.cos(3 * gamma) + 0.001480 * np.sin(3 * gamma))

            eot = (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) -
                   0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)) * 229.18

            # Standard meridian for Asia/Kolkata (IST) is 82.5° E
            # For other timezones, this might need to be dynamic or configured.
            standard_meridian = 82.5
            tc = 4 * (longitude - standard_meridian) + eot

            local_time_decimal_hours = datetime_obj_localized.hour + datetime_obj_localized.minute / 60.0 + datetime_obj_localized.second / 3600.0
            lst_decimal_hours = local_time_decimal_hours + tc / 60.0

            hour_angle_deg = 15 * (lst_decimal_hours - 12)

            latitude_rad = np.radians(latitude)
            hour_angle_rad = np.radians(hour_angle_deg)

            cos_zenith = (np.sin(latitude_rad) * np.sin(declination_rad) +
                          np.cos(latitude_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad))

            cos_zenith = np.clip(cos_zenith, -1, 1)

            zenith_rad = np.arccos(cos_zenith)
            zenith_deg = np.degrees(zenith_rad)

            if zenith_deg >= 90:
                relative_airmass = np.inf
            else:
                # Kasten and Young approximation for relative air mass
                z_deg = zenith_deg
                relative_airmass = 1.0 / (np.cos(zenith_rad) + 0.5057 * (96.07995 - z_deg)**-1.6364)
            
            return zenith_deg, zenith_rad, relative_airmass

        except Exception as e:
            self.error = f"Error in custom solar position calculation (worker): {e}"
            return np.nan, np.nan, np.nan

    def run(self):
        """
        Executes the data processing for the assigned group.
        """
        if self.error: # If pvlib init failed, don't proceed
            self.result_queue.put(None) # Indicate failure
            return

        try:
            hour, group = self.group_data

            if len(group) < 5:
                top5 = group.nlargest(len(group), '400nm') # Use 400nm to find top 5 bright readings
            else:
                top5 = group.nlargest(5, '400nm')

            if top5.empty:
                self.result_queue.put(None) # Indicate empty group
                return

            avg_timestamp_ns = top5['Datetime'].values.astype('int64').mean()
            avg_datetime_naive = pd.to_datetime(avg_timestamp_ns)

            zenith, zenith_rad, relative_airmass = np.nan, np.nan, np.nan

            if self.is_online_mode:
                if self.fixed_location: # Check if pvlib location was initialized successfully
                    try:
                        avg_datetime_localized = pd.Timestamp(avg_datetime_naive, tz=self.timezone_str)
                        solpos = self.fixed_location.get_solarposition(pd.DatetimeIndex([avg_datetime_localized]))
                        zenith = solpos.iloc[0]['apparent_zenith']
                        zenith_rad = np.deg2rad(zenith)
                        relative_airmass = atmosphere.get_relative_airmass(zenith)
                    except Exception as e:
                        self.error = f"PVlib calculation failed for hour {hour}: {e}"
                else:
                    self.error = "PVlib location not initialized, skipping calculations."
            else: # Offline mode, use custom calculations
                zenith, zenith_rad, relative_airmass = self._calculate_solar_position_custom(
                    avg_datetime_naive, self.latitude, self.longitude, self.altitude, self.timezone_str
                )

            # Compute natural logs of averages
            avg_400nm_mean = top5['400nm'].mean()
            avg_500nm_mean = top5['500nm'].mean()
            avg_870nm_mean = top5['870nm'].mean()
            avg_1030nm_mean = top5['1030nm'].mean()

            ln_400nm = np.log(avg_400nm_mean) if avg_400nm_mean > 0 else np.nan
            ln_500nm = np.log(avg_500nm_mean) if avg_500nm_mean > 0 else np.nan
            ln_870nm = np.log(avg_870nm_mean) if avg_870nm_mean > 0 else np.nan
            ln_1030nm = np.log(avg_1030nm_mean) if avg_1030nm_mean > 0 else np.nan
            
            # Prepare AOD and new Langley plot related values
            aod_values = {}
            slopes = {}
            intercepts = {}
            exp_intercepts = {}
            total_optical_depths = {} # New dictionary for total optical depth

            wavelengths = {'400nm': 400, '500nm': 500, '870nm': 870, '1030nm': 1030}
            ln_irradiances = {'400nm': ln_400nm, '500nm': ln_500nm, '870nm': ln_870nm, '1030nm': ln_1030nm}

            for wl_str, wl_nm in wavelengths.items():
                # Retrieve Langley plot results for this wavelength
                slope_key = f'slope_{wl_str}'
                intercept_key = f'intercept_{wl_str}'
                
                slope = self.langley_results.get(slope_key, np.nan)
                intercept = self.langley_results.get(intercept_key, np.nan)
                exp_intercept = np.exp(intercept) if not np.isnan(intercept) else np.nan

                slopes[f'Slope_{wl_str}'] = slope
                intercepts[f'Intercept_Y_Axis_{wl_str}'] = intercept # Renamed for clarity
                exp_intercepts[f'Exp_Intercept_{wl_str}'] = exp_intercept

                # Calculate Total Optical Depth as negative of the slope
                total_optical_depths[f'Total_Optical_Depth_{wl_str}'] = -slope if not np.isnan(slope) else np.nan

                # Calculate AOD using the formula: (Intercept - ln(I)) / relative_airmass
                if not np.isnan(ln_irradiances[wl_str]) and not np.isnan(intercept) and \
                   not np.isnan(relative_airmass) and relative_airmass != 0:
                    aod = (intercept - ln_irradiances[wl_str]) / relative_airmass
                    aod_values[f'AOD_{wl_str}'] = aod
                else:
                    aod_values[f'AOD_{wl_str}'] = np.nan

            # Calculate a single "final" AOD value (average of the four AODs)
            valid_aods = [aod_values[f'AOD_{wl_str}'] for wl_str in wavelengths if not np.isnan(aod_values[f'AOD_{wl_str}'])]
            final_aod_avg = np.mean(valid_aods) if valid_aods else np.nan

            # Put the result into the queue
            self.result_queue.put([
                avg_datetime_naive.date(),
                avg_datetime_naive.time(),
                avg_400nm_mean, # Changed to raw mean for easier check
                avg_500nm_mean,
                avg_870nm_mean,
                avg_1030nm_mean,
                ln_400nm,
                ln_500nm,
                ln_870nm,
                ln_1030nm,
                zenith,
                zenith_rad,
                relative_airmass,
                aod_values.get('AOD_400nm', np.nan),
                aod_values.get('AOD_500nm', np.nan),
                aod_values.get('AOD_870nm', np.nan),
                aod_values.get('AOD_1030nm', np.nan),
                final_aod_avg, # New final AOD column
                slopes.get('Slope_400nm', np.nan),
                slopes.get('Slope_500nm', np.nan),
                slopes.get('Slope_870nm', np.nan),
                slopes.get('Slope_1030nm', np.nan),
                intercepts.get('Intercept_Y_Axis_400nm', np.nan), # Updated column name
                intercepts.get('Intercept_Y_Axis_500nm', np.nan), # Updated column name
                intercepts.get('Intercept_Y_Axis_870nm', np.nan), # Updated column name
                intercepts.get('Intercept_Y_Axis_1030nm', np.nan), # Updated column name
                exp_intercepts.get('Exp_Intercept_400nm', np.nan),
                exp_intercepts.get('Exp_Intercept_500nm', np.nan),
                exp_intercepts.get('Exp_Intercept_870nm', np.nan),
                exp_intercepts.get('Exp_Intercept_1030nm', np.nan),
                total_optical_depths.get('Total_Optical_Depth_400nm', np.nan), # New column
                total_optical_depths.get('Total_Optical_Depth_500nm', np.nan), # New column
                total_optical_depths.get('Total_Optical_Depth_870nm', np.nan), # New column
                total_optical_depths.get('Total_Optical_Depth_1030nm', np.nan)  # New column
            ])
        except Exception as e:
            self.error = f"An unexpected error occurred in worker thread: {e}"
            self.result_queue.put(None) # Indicate failure


class SolarDataProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solar Data Processor (PyQt5)")
        self.setGeometry(100, 100, 700, 680) # Increased height to accommodate new section
        self.setFixedSize(self.size())

        self.input_file_path = ""
        self.default_latitude = 23.0258
        self.default_longitude = 72.5873
        self.default_altitude = 55.0
        self.default_timezone = 'Asia/Kolkata'

        self.init_ui()
        self.offline_radio.setChecked(True)
        self.on_mode_changed()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # --- Operation Mode ---
        mode_group_box = QGroupBox("Operation Mode")
        mode_layout = QHBoxLayout()
        self.offline_radio = QRadioButton("Offline Mode (Manual Input, Custom Calculations)")
        self.online_radio = QRadioButton("Online Mode (Auto Location, PVlib Calculations)")
        self.offline_radio.toggled.connect(self.on_mode_changed)
        self.online_radio.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.offline_radio)
        mode_layout.addWidget(self.online_radio)
        mode_group_box.setLayout(mode_layout)
        main_layout.addWidget(mode_group_box)

        # --- File Selection ---
        file_layout = QHBoxLayout()
        self.file_label = QLabel("Input File:")
        self.file_label.setStyleSheet("font-weight: bold;")
        self.file_path_display = QLineEdit()
        self.file_path_display.setReadOnly(True)
        self.file_path_display.setPlaceholderText("No file selected")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        self.browse_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; padding: 8px 15px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_path_display)
        file_layout.addWidget(self.browse_button)
        main_layout.addLayout(file_layout)

        # --- Location Inputs ---
        location_group_box = QGroupBox("Location and Timezone")
        location_layout = QVBoxLayout()

        lat_layout = QHBoxLayout()
        lat_label = QLabel("Latitude:")
        lat_label.setStyleSheet("font-weight: bold;")
        self.latitude_input = QDoubleSpinBox()
        self.latitude_input.setRange(-90.0, 90.0)
        self.latitude_input.setDecimals(4)
        self.latitude_input.setValue(self.default_latitude)
        lat_layout.addWidget(lat_label)
        lat_layout.addWidget(self.latitude_input)
        location_layout.addLayout(lat_layout)

        lon_layout = QHBoxLayout()
        lon_label = QLabel("Longitude:")
        lon_label.setStyleSheet("font-weight: bold;")
        self.longitude_input = QDoubleSpinBox()
        self.longitude_input.setRange(-180.0, 180.0)
        self.longitude_input.setDecimals(4)
        self.longitude_input.setValue(self.default_longitude)
        lon_layout.addWidget(lon_label)
        lon_layout.addWidget(self.longitude_input)
        location_layout.addLayout(lon_layout)

        alt_layout = QHBoxLayout()
        alt_label = QLabel("Altitude (m):")
        alt_label.setStyleSheet("font-weight: bold;")
        self.altitude_input = QDoubleSpinBox()
        self.altitude_input.setRange(-1000.0, 10000.0)
        self.altitude_input.setDecimals(2)
        self.altitude_input.setValue(self.default_altitude)
        alt_layout.addWidget(alt_label)
        alt_layout.addWidget(self.altitude_input)
        location_layout.addLayout(alt_layout)

        tz_layout = QHBoxLayout()
        tz_label = QLabel("Timezone (e.g., Asia/Kolkata):")
        tz_label.setStyleSheet("font-weight: bold;")
        self.timezone_input = QLineEdit()
        self.timezone_input.setText(self.default_timezone)
        tz_layout.addWidget(tz_label)
        tz_layout.addWidget(self.timezone_input)
        location_layout.addLayout(tz_layout)
        
        location_group_box.setLayout(location_layout)
        main_layout.addWidget(location_group_box)

        # --- New button for Optical Depth calculation ---
        self.calculate_od_button = QPushButton("Calculate Total Optical Depth")
        self.calculate_od_button.clicked.connect(self.calculate_optical_depth)
        self.calculate_od_button.setStyleSheet(
            "QPushButton { background-color: #FFA500; color: white; font-size: 12px; font-weight: bold; border-radius: 8px; padding: 10px 20px; }"
            "QPushButton:hover { background-color: #FF8C00; }"
        )
        main_layout.addWidget(self.calculate_od_button, alignment=Qt.AlignCenter)

        # --- Langley Plot Results Display ---
        langley_results_group_box = QGroupBox("Langley Plot Derived Total Optical Depth (from entire file)")
        langley_results_layout = QVBoxLayout()
        
        # Dictionary to hold the QLineEdit widgets for easy access
        self.total_optical_depth_displays = {}
        wavelengths = ['400nm', '500nm', '870nm', '1030nm']
        for wl in wavelengths:
            h_layout = QHBoxLayout()
            # label = QLabel(f"Total Optical Depth ({wl}):")
            label = QLabel(f"OD ({wl}):")

            label.setStyleSheet("font-weight: bold;")
            display = QLineEdit("N/A")
            display.setReadOnly(True)
            display.setStyleSheet("background-color: #e0e0e0;")
            self.total_optical_depth_displays[wl] = display
            h_layout.addWidget(label)
            h_layout.addWidget(display)
            langley_results_layout.addLayout(h_layout)

        langley_results_group_box.setLayout(langley_results_layout)
        main_layout.addWidget(langley_results_group_box)
        
        # --- Process Data Button ---
        self.process_button = QPushButton("Process Hourly Data (Calculate AODs)")
        self.process_button.clicked.connect(self.process_data)
        self.process_button.setStyleSheet(
            "QPushButton { background-color: #008CBA; color: white; font-size: 14px; font-weight: bold; border-radius: 8px; padding: 12px 25px; }"
            "QPushButton:hover { background-color: #007bb5; }"
        )
        main_layout.addWidget(self.process_button, alignment=Qt.AlignCenter)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-style: italic; color: blue;")
        main_layout.addWidget(self.status_label)


        self.setLayout(main_layout)

    def on_mode_changed(self):
        is_offline = self.offline_radio.isChecked()

        self.latitude_input.setReadOnly(not is_offline)
        self.longitude_input.setReadOnly(not is_offline)
        self.timezone_input.setReadOnly(not is_offline)

        read_only_style = "background-color: #f0f0f0;"
        editable_style = ""
        self.latitude_input.setStyleSheet(read_only_style if not is_offline else editable_style)
        self.longitude_input.setStyleSheet(read_only_style if not is_offline else editable_style)
        self.timezone_input.setStyleSheet(read_only_style if not is_offline else editable_style)

        if not is_offline:
            self.status_label.setText("Attempting to get current location...")
            self.status_label.setStyleSheet("font-style: italic; color: blue;")
            QApplication.processEvents()
            self.get_current_location_online()
        else:
            self.latitude_input.setValue(self.default_latitude)
            self.longitude_input.setValue(self.default_longitude)
            self.altitude_input.setValue(self.default_altitude)
            self.timezone_input.setText(self.default_timezone)
            self.status_label.setText("")
            self.status_label.setStyleSheet("font-style: italic; color: blue;")
        
        # Clear Langley results display when mode changes
        for wl in self.total_optical_depth_displays:
            self.total_optical_depth_displays[wl].setText("N/A")


    def get_current_location_online(self):
        try:
            # Use current time to get a more accurate location if possible
            current_time = pd.Timestamp.now(tz='UTC')
            g = geocoder.ip('me')
            if g.ok and g.latlng:
                self.latitude_input.setValue(g.latlng[0])
                self.longitude_input.setValue(g.latlng[1])
                # Altitude is often not reliably returned by IP geocoders.
                # Keeping it at 0.0 or a sensible default for online mode is common.
                self.altitude_input.setValue(0.0) 

                if g.raw and 'timezone' in g.raw:
                    self.timezone_input.setText(g.raw['timezone'])
                else:
                    self.timezone_input.setText('UTC')
                    QMessageBox.warning(self, "Location Info", "Could not determine precise timezone from geocoder. Defaulting to UTC. Please verify.")

                self.status_label.setText("Location found. Verify details if necessary.")
                self.status_label.setStyleSheet("font-style: italic; color: green;")
            else:
                QMessageBox.warning(self, "Location Error", "Could not determine current location automatically. Please check your internet connection or switch to Offline Mode.")
                self.status_label.setText("Failed to get location online.")
                self.status_label.setStyleSheet("font-style: italic; color: orange;")
                self.offline_radio.setChecked(True) # Fallback to offline mode
        except Exception as e:
            QMessageBox.critical(self, "Geocoder Error", f"An error occurred while getting location: {e}\nPlease check your internet connection or switch to Offline Mode.")
            self.status_label.setText(f"Error getting location: {e}")
            self.status_label.setStyleSheet("font-style: italic; color: red;")
            self.offline_radio.setChecked(True) # Fallback to offline mode

    def browse_file(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Text files (*.txt);;All files (*.*)")
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.input_file_path = selected_files[0]
                self.file_path_display.setText(os.path.basename(self.input_file_path))
                # Clear previous Langley results when a new file is selected
                for wl in self.total_optical_depth_displays:
                    self.total_optical_depth_displays[wl].setText("N/A")


    def _get_common_params(self):
        """Helper to get common parameters from UI."""
        try:
            latitude = self.latitude_input.value()
            longitude = self.longitude_input.value()
            altitude = self.altitude_input.value()
            timezone_str = self.timezone_input.text()
            pytz.timezone(timezone_str) # Validate timezone string
            return latitude, longitude, altitude, timezone_str
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Please enter valid numeric values for Latitude, Longitude, and Altitude.")
            self.status_label.setText("Invalid numeric input for location parameters.")
            self.status_label.setStyleSheet("font-style: italic; color: red;")
            return None, None, None, None
        except pytz.UnknownTimeZoneError:
            QMessageBox.critical(self, "Invalid Timezone", "The entered timezone is not recognized. Please use a valid IANA timezone string (e.g., 'America/New_York', 'Europe/London', 'Asia/Kolkata').")
            self.status_label.setText("Invalid timezone string.")
            self.status_label.setStyleSheet("font-style: italic; color: red;")
            return None, None, None, None

    def calculate_optical_depth(self):
        """
        Calculates and displays the Total Optical Depth using the Langley plot method.
        This function can be called independently.
        """
        if not self.input_file_path:
            QMessageBox.critical(self, "Error", "Please select an input file first.")
            self.status_label.setText("Please select an input file.")
            self.status_label.setStyleSheet("font-style: italic; color: red;")
            return

        latitude, longitude, altitude, timezone_str = self._get_common_params()
        if latitude is None: return

        self.status_label.setText("Calculating Total Optical Depth...")
        self.status_label.setStyleSheet("font-style: italic; color: blue;")
        QApplication.processEvents()

        data = []
        try:
            with open(self.input_file_path, 'r') as file:
                for line in file:
                    match = pattern.match(line.strip())
                    if match:
                        date_str, time_str, v1, v2, v3, v4 = match.groups()
                        dt_str = f"{date_str} {time_str}"
                        dt_obj = parser.parse(dt_str, fuzzy=True)
                        data.append([dt_obj, float(v1), float(v2), float(v3), float(v4)])
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Input file '{self.input_file_path}' not found.")
            self.status_label.setText(f"Error: Input file '{self.input_file_path}' not found.")
            self.status_label.setStyleSheet("font-style: italic; color: red;")
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while reading or parsing the input file: {e}")
            self.status_label.setText(f"Error during file read/parse: {e}")
            self.status_label.setStyleSheet("font-style: italic; color: red;")
            return

        if not data:
            QMessageBox.warning(self, "No Data", "No valid data found in the input file for Langley plot.")
            self.status_label.setText("No valid data found for Langley plot.")
            self.status_label.setStyleSheet("font-style: italic; color: orange;")
            return

        df = pd.DataFrame(data, columns=['Datetime', '400nm', '500nm', '870nm', '1030nm'])
        
        langley_results = self._perform_langley_calculation(df, latitude, longitude, altitude, timezone_str)
        
        self._update_langley_display(langley_results)

        if any(not np.isnan(langley_results.get(f'total_optical_depth_{wl}', np.nan)) for wl in ['400nm', '500nm', '870nm', '1030nm']):
            self.status_label.setText("Total Optical Depth calculated and displayed above.")
            self.status_label.setStyleSheet("font-style: italic; color: green;")
            QMessageBox.information(self, "Calculation Complete", "Total Optical Depth values have been calculated and updated in the display.")
        else:
            self.status_label.setText("Failed to calculate Total Optical Depth for all wavelengths. Check data and parameters.")
            self.status_label.setStyleSheet("font-style: italic; color: orange;")
            QMessageBox.warning(self, "Calculation Failed", "Could not calculate Total Optical Depth for all wavelengths. Please ensure your data is suitable for Langley plot analysis (e.g., sufficient range of air masses).")


    def _perform_langley_calculation(self, df, latitude, longitude, altitude, timezone_str):
        """
        Encapsulates the Langley plot calculation logic.
        Returns a dictionary of slopes, intercepts, and total optical depths.
        """
        langley_results = {}
        wavelength_cols = ['400nm', '500nm', '870nm', '1030nm']
        
        temp_df = df.copy()
        temp_df['Datetime_localized'] = pd.to_datetime(temp_df['Datetime']).dt.tz_localize(timezone_str)
        
        is_online_mode = self.online_radio.isChecked()
        
        if is_online_mode:
            try:
                fixed_location_for_langley = Location(latitude=latitude, longitude=longitude,
                                                       tz=timezone_str, altitude=altitude)
                solpos_all = fixed_location_for_langley.get_solarposition(temp_df['Datetime_localized'])
                temp_df['relative_airmass'] = atmosphere.get_relative_airmass(solpos_all['apparent_zenith'])
            except Exception as e:
                QMessageBox.critical(self, "PVlib Error", f"Error during Langley plot airmass calculation: {e}")
                for wl_col in wavelength_cols:
                    langley_results[f'slope_{wl_col}'] = np.nan
                    langley_results[f'intercept_{wl_col}'] = np.nan
                    langley_results[f'total_optical_depth_{wl_col}'] = np.nan
                return langley_results
        else:
            airmass_list = []
            dummy_worker = SolarProcessorWorker(
                (None, None), latitude, longitude, altitude, timezone_str, False, {}, Queue()
            )
            for _, row in temp_df.iterrows():
                zenith_deg, zenith_rad, relative_airmass = dummy_worker._calculate_solar_position_custom(
                    row['Datetime'], latitude, longitude, altitude, timezone_str
                )
                airmass_list.append(relative_airmass)
            temp_df['relative_airmass'] = airmass_list

        # Filter for valid air mass values (e.g., m > 0 and m < 5 for good linearity)
        # Also ensure values are not NaN before plotting
        langley_df = temp_df[(temp_df['relative_airmass'] > 0) & (temp_df['relative_airmass'] < 5)].copy()

        if langley_df.empty:
            QMessageBox.warning(self, "Langley Plot Warning", "Not enough suitable data points (air mass 0-5) for Langley plot. Io values and related parameters will be NaN.")
            for wl_col in wavelength_cols:
                langley_results[f'slope_{wl_col}'] = np.nan
                langley_results[f'intercept_{wl_col}'] = np.nan
                langley_results[f'total_optical_depth_{wl_col}'] = np.nan
        else:
            for wl_col in wavelength_cols:
                langley_df[f'ln_{wl_col}'] = np.log(langley_df[wl_col].replace(0, np.nan))

                valid_data = langley_df[['relative_airmass', f'ln_{wl_col}']].dropna()

                if len(valid_data) >= 2: # Need at least 2 points for a line
                    try:
                        coefficients = np.polyfit(valid_data['relative_airmass'], valid_data[f'ln_{wl_col}'], 1)
                        langley_results[f'slope_{wl_col}'] = coefficients[0]
                        langley_results[f'intercept_{wl_col}'] = coefficients[1]
                        # Total optical depth is the negative of the slope
                        langley_results[f'total_optical_depth_{wl_col}'] = -coefficients[0]
                    except Exception as e:
                        QMessageBox.warning(self, "Langley Plot Error", f"Could not perform Langley regression for {wl_col}: {e}. Parameters will be NaN.")
                        langley_results[f'slope_{wl_col}'] = np.nan
                        langley_results[f'intercept_{wl_col}'] = np.nan
                        langley_results[f'total_optical_depth_{wl_col}'] = np.nan
                else:
                    QMessageBox.warning(self, "Langley Plot Warning", f"Not enough valid data points for Langley plot for {wl_col} (need at least 2). Parameters will be NaN.")
                    langley_results[f'slope_{wl_col}'] = np.nan
                    langley_results[f'intercept_{wl_col}'] = np.nan
                    langley_results[f'total_optical_depth_{wl_col}'] = np.nan
        return langley_results

    def process_data(self):
        """
        Processes the entire input file to calculate hourly AODs.
        """
        if not self.input_file_path:
            QMessageBox.critical(self, "Error", "Please select an input file.")
            self.status_label.setText("Please select an input file.")
            self.status_label.setStyleSheet("font-style: italic; color: red;")
            return

        latitude, longitude, altitude, timezone_str = self._get_common_params()
        if latitude is None: return

        self.status_label.setText("Processing hourly data (this includes Langley calculations)...")
        self.status_label.setStyleSheet("font-style: italic; color: blue;")
        QApplication.processEvents()

        data = []
        try:
            with open(self.input_file_path, 'r') as file:
                for line in file:
                    match = pattern.match(line.strip())
                    if match:
                        date_str, time_str, v1, v2, v3, v4 = match.groups()
                        dt_str = f"{date_str} {time_str}"
                        dt_obj = parser.parse(dt_str, fuzzy=True)
                        data.append([dt_obj, float(v1), float(v2), float(v3), float(v4)])
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Input file '{self.input_file_path}' not found.")
            self.status_label.setText(f"Error: Input file '{self.input_file_path}' not found.")
            self.status_label.setStyleSheet("font-style: italic; color: red;")
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while reading or parsing the input file: {e}")
            self.status_label.setText(f"Error during file read/parse: {e}")
            self.status_label.setStyleSheet("font-style: italic; color: red;")
            return

        if not data:
            QMessageBox.warning(self, "No Data", "No valid data found in the input file.")
            self.status_label.setText("No valid data found.")
            self.status_label.setStyleSheet("font-style: italic; color: orange;")
            return

        df = pd.DataFrame(data, columns=['Datetime', '400nm', '500nm', '870nm', '1030nm'])
        df['HourGroup'] = df['Datetime'].dt.floor('H')

        # Calculate Langley Plot parameters (Slopes and Intercepts) first
        langley_results = self._perform_langley_calculation(df, latitude, longitude, altitude, timezone_str)
        # Update the GUI with Langley results
        self._update_langley_display(langley_results)


        hourly_averages = []
        result_queue = Queue()
        threads = []

        for hour, group in df.groupby('HourGroup'):
            worker = SolarProcessorWorker(
                (hour, group), latitude, longitude, altitude, timezone_str, self.online_radio.isChecked(), langley_results, result_queue
            )
            threads.append(worker)
            worker.start()

        for worker in threads:
            worker.join()
            if worker.error:
                self.status_label.setText(f"Error in worker thread: {worker.error}")
                self.status_label.setStyleSheet("font-style: italic; color: red;")

        while not result_queue.empty():
            result = result_queue.get()
            if result is not None:
                hourly_averages.append(result)

        if not hourly_averages:
            QMessageBox.warning(self, "No Results", "No hourly averages could be calculated from the data.")
            self.status_label.setText("No hourly averages calculated.")
            self.status_label.setStyleSheet("font-style: italic; color: orange;")
            return

        result_df = pd.DataFrame(hourly_averages, columns=[
            'Date', 'Avg_Top5_Time', 'Avg_400nm_Meas', 'Avg_500nm_Meas', 'Avg_870nm_Meas', 'Avg_1030nm_Meas', # Raw measurements
            'ln_400nm', 'ln_500nm', 'ln_870nm', 'ln_1030nm',
            'Solar_Zenith_Angle_Deg', 'Solar_Zenith_Angle_Rad', 'Relative_Air_Mass',
            'AOD_400nm', 'AOD_500nm', 'AOD_870nm', 'AOD_1030nm',
            'Final_AOD_Avg',
            'Slope_400nm', 'Slope_500nm', 'Slope_870nm', 'Slope_1030nm',
            'Intercept_Y_Axis_400nm', 'Intercept_Y_Axis_500nm', 'Intercept_Y_Axis_870nm', 'Intercept_Y_Axis_1030nm', # Updated column names
            'Exp_Intercept_400nm', 'Exp_Intercept_500nm', 'Exp_Intercept_870nm', 'Exp_Intercept_1030nm',
            'Total_Optical_Depth_400nm', 'Total_Optical_Depth_500nm', 'Total_Optical_Depth_870nm', 'Total_Optical_Depth_1030nm' # New columns
        ])

        result_df['Date'] = result_df['Date'].apply(lambda d: d.strftime('%Y-%m-%d'))
        result_df['Avg_Top5_Time'] = result_df['Avg_Top5_Time'].apply(lambda t: t.strftime('%H:%M:%S'))
        # result_df = result_df.sort_values(by='Date')

        output_file = '11top_hourly_top5_avg_ln_with_zenith_airmass_aod_langley.xlsx' # Renamed output file
        try:
            result_df.to_excel(output_file, index=False)
            self.status_label.setText(f"✅ Results saved to '{output_file}' (AOD based on (Intercept - ln(I)) / m)")
            self.status_label.setStyleSheet("font-style: italic; color: green;")
            QMessageBox.information(self, "Success", f"Hourly processing complete. Results saved to '{output_file}'\n\n"
                                                      f"Note on AOD:\n- AOD is calculated as (Intercept - ln(I)) / Relative Air Mass.\n"
                                                      f"- 'Final_AOD_Avg' is the average of AODs at the four wavelengths.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving output to Excel file: {e}")
            self.status_label.setText(f"Error saving output: {e}")
            self.status_label.setStyleSheet("font-style: italic; color: red;")

    def _update_langley_display(self, langley_results):
        """Updates the Total Optical Depth display fields in the GUI."""
        wavelengths = ['400nm', '500nm', '870nm', '1030nm']
        for wl in wavelengths:
            tod_value = langley_results.get(f'total_optical_depth_{wl}', np.nan)
            if not np.isnan(tod_value):
                self.total_optical_depth_displays[wl].setText(f"{tod_value:.4f}")
            else:
                self.total_optical_depth_displays[wl].setText("N/A")


# Main execution block
if __name__ == "__main__":
    app = QApplication(sys.argv)    
    ex = SolarDataProcessorApp()
    ex.show()
    sys.exit(app.exec_())
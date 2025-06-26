import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import curve_fit, minimize
from scipy.special import expit
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QDoubleSpinBox, QFileDialog, QCheckBox,
    QGridLayout, QGroupBox, QComboBox, QStatusBar, QMessageBox, QSplitter,
    QFormLayout, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt
from sklearn.metrics import r2_score, mean_squared_error

def double_logistic(t, baseline, amplitude, start, end, growth_rate, senescence_rate):
    return baseline + (amplitude * expit(growth_rate * (t - start))) * (1 - expit(senescence_rate * (t - end)))

class GroupingDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Grouping")
        self.setFixedSize(450, 320)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.start_combo = QComboBox()
        self.start_combo.addItems(columns)
        form.addRow("Start Column:", self.start_combo)
        self.end_combo = QComboBox()
        self.end_combo.addItems(columns)
        self.end_combo.setCurrentIndex(-1)
        form.addRow("End Column (optional):", self.end_combo)
        self.label_combo = QComboBox()
        self.label_combo.addItems(columns)
        form.addRow("Group Label Column:", self.label_combo)
        self.color_combo = QComboBox()
        self.color_combo.addItems(columns)
        self.color_combo.setCurrentIndex(-1)
        form.addRow("Color Column (optional):", self.color_combo)
        layout.addLayout(form)
        help_label = QLabel("Tip: For single-day events, leave End Column blank")
        help_label.setStyleSheet("font-size: 10px; color: #666;")
        layout.addWidget(help_label)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

class CurveFitApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CurveFit Pro")
        self.resize(1400, 900)
        self.df = None
        self.groups = None
        self.x = None
        self.y = None
        self.param_names = ['Baseline', 'Amplitude', 'Start', 'End', 'Growth_Rate', 'Senescence_Rate']
        # Dynamic bounds - will be set based on data
        self.bounds = ([-1, 0, 0, 0, 0.01, 0.01], [1, 10, 500, 500, 1, 1])
        self.slider_bounds = ([-1, 0, 0, 0, 0.01, 0.01], [1, 1, 365, 365, 1, 1])
        self.init_params = [0.2, 0.6, 50, 200, 0.1, 0.1]
        self.params = self.init_params.copy()
        self.grouping_config = {}
        self.current_y_col = ""
        self.x_col_name = ""
        self.build_ui()
        self.statusBar().showMessage("Ready")
        self.current_file = ""
        self.groups_file = ""

    def build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        file_group = QGroupBox("Data Input")
        file_layout = QVBoxLayout(file_group)
        self.load_data_btn = QPushButton("Load Data File")
        self.load_data_btn.clicked.connect(self.load_data_file)
        file_layout.addWidget(self.load_data_btn)
        self.load_groups_btn = QPushButton("Load Grouping Data")
        self.load_groups_btn.clicked.connect(self.load_groups_file)
        file_layout.addWidget(self.load_groups_btn)
        self.clear_grouping_btn = QPushButton("Clear Grouping Data")
        self.clear_grouping_btn.clicked.connect(self.clear_grouping_data)
        self.clear_grouping_btn.setEnabled(False)
        file_layout.addWidget(self.clear_grouping_btn)
        self.x_col_combo = QComboBox()
        self.y_col_combo = QComboBox()
        col_layout = QGridLayout()
        col_layout.addWidget(QLabel("X Column:"), 0, 0)
        col_layout.addWidget(self.x_col_combo, 0, 1)
        col_layout.addWidget(QLabel("Y Column:"), 1, 0)
        col_layout.addWidget(self.y_col_combo, 1, 1)
        file_layout.addLayout(col_layout)
        left_layout.addWidget(file_group)
        param_group = QGroupBox("Parameters Control")
        param_layout = QVBoxLayout(param_group)
        self.sliders = []
        self.spin_boxes = []
        self.locks = []
        for i, name in enumerate(self.param_names):
            param_box = QGroupBox(name)
            box_layout = QHBoxLayout(param_box)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(500)
            slider.setMinimumWidth(150)
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setMinimum(self.bounds[0][i])
            spin.setMaximum(self.bounds[1][i])
            spin.setValue(self.params[i])
            lock = QCheckBox("Lock")
            self.sliders.append(slider)
            self.spin_boxes.append(spin)
            self.locks.append(lock)
            box_layout.addWidget(slider)
            box_layout.addWidget(spin)
            box_layout.addWidget(lock)
            param_layout.addWidget(param_box)
        left_layout.addWidget(param_group)
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        self.optimize_btn = QPushButton("Optimize Fit")
        self.optimize_btn.clicked.connect(self.optimize_fit)
        self.save_btn = QPushButton("Download Graph + Excel")
        self.save_btn.clicked.connect(self.export_outputs)
        self.show_key_points_cb = QCheckBox("Show SOS/EOS/Peak")
        self.show_key_points_cb.setChecked(True)
        action_layout.addWidget(self.optimize_btn)
        action_layout.addWidget(self.save_btn)
        action_layout.addWidget(self.show_key_points_cb)
        left_layout.addWidget(action_group)
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.r2_label = QLabel("R²: ---")
        self.rmse_label = QLabel("RMSE: ---")
        self.params_label = QLabel("Parameters: ---")
        stats_layout.addWidget(self.r2_label)
        stats_layout.addWidget(self.rmse_label)
        stats_layout.addWidget(self.params_label)
        left_layout.addWidget(stats_group)
        left_layout.addStretch()
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])
        self.update_ui_state(False)
        for slider in self.sliders:
            slider.valueChanged.connect(self.sync_from_sliders)
        for spin in self.spin_boxes:
            spin.valueChanged.connect(self.sync_from_spinbox)
        self.x_col_combo.currentTextChanged.connect(self.update_data_columns)
        self.y_col_combo.currentTextChanged.connect(self.update_data_columns)
        self.show_key_points_cb.stateChanged.connect(self.update_plot)

    def update_ui_state(self, data_loaded):
        self.optimize_btn.setEnabled(data_loaded)
        self.save_btn.setEnabled(data_loaded)
        for slider in self.sliders:
            slider.setEnabled(data_loaded)
        for spin in self.spin_boxes:
            spin.setEnabled(data_loaded)
        for lock in self.locks:
            lock.setEnabled(data_loaded)
        self.x_col_combo.setEnabled(data_loaded)
        self.y_col_combo.setEnabled(data_loaded)

    def load_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", 
            "Data Files (*.xlsx *.xls *.csv);;All Files (*)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path)
            else:
                self.df = pd.read_csv(file_path)
            self.current_file = file_path
            self.statusBar().showMessage(f"Loaded: {file_path}")
            self.x_col_combo.clear()
            self.y_col_combo.clear()
            self.x_col_combo.addItems(self.df.columns)
            self.y_col_combo.addItems(self.df.columns)
            for col in ['DAS', 'DOY', 'Days', 'days', 'Time']:
                if col in self.df.columns:
                    self.x_col_combo.setCurrentText(col)
                    break
            for col in ['GCC', 'NDVI', 'Value', 'Index']:
                if col in self.df.columns:
                    self.y_col_combo.setCurrentText(col)
                    break
            self.update_ui_state(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
            self.statusBar().showMessage("Error loading file")

    def load_groups_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Grouping Data File", "", 
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if not file_path:
            return
        try:
            try:
                groups_df = pd.read_excel(file_path)
            except:
                groups_df = pd.read_csv(file_path)
            self.groups_file = file_path
            dialog = GroupingDialog(groups_df.columns.tolist(), self)
            if dialog.exec_() == QDialog.Accepted:
                start_col = dialog.start_combo.currentText()
                end_col = dialog.end_combo.currentText()
                label_col = dialog.label_combo.currentText()
                color_col = dialog.color_combo.currentText() if dialog.color_combo.currentText() else None
                if start_col not in groups_df.columns or label_col not in groups_df.columns:
                    QMessageBox.warning(self, "Error", "Start and Group Label columns are required in grouping data")
                    return
                self.grouping_config = {
                    'start_col': start_col,
                    'end_col': end_col if end_col else None,
                    'label_col': label_col,
                    'color_col': color_col
                }
                self.groups = groups_df
                self.clear_grouping_btn.setEnabled(True)
                self.statusBar().showMessage(f"Loaded grouping data: {file_path}")
                self.update_plot()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load grouping data:\n{str(e)}")

    def clear_grouping_data(self):
        self.groups = None
        self.grouping_config = {}
        self.clear_grouping_btn.setEnabled(False)
        self.statusBar().showMessage("Grouping data cleared")
        self.update_plot()

    def sync_from_sliders(self):
        """Update spin boxes from slider values using slider bounds"""
        for i, slider in enumerate(self.sliders):
            val = self.slider_bounds[0][i] + (self.slider_bounds[1][i] - self.slider_bounds[0][i]) * slider.value() / 1000
            self.spin_boxes[i].blockSignals(True)
            self.spin_boxes[i].setValue(val)
            self.spin_boxes[i].blockSignals(False)
            self.params[i] = val
        
        # Validate SOS/EOS relationship
        self.validate_and_fix_parameters()
        self.update_plot()

    def sync_from_spinbox(self):
        """Update sliders from spin box values, allowing spin box to exceed slider bounds"""
        for i, spin in enumerate(self.spin_boxes):
            self.params[i] = spin.value()
            val = spin.value()
            slider_min = self.slider_bounds[0][i]
            slider_max = self.slider_bounds[1][i]
            if val <= slider_min:
                slider_pos = 0
            elif val >= slider_max:
                slider_pos = 1000
            else:
                slider_pos = int((val - slider_min) / (slider_max - slider_min) * 1000)
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(slider_pos)
            self.sliders[i].blockSignals(False)
        
        # Validate SOS/EOS relationship
        self.validate_and_fix_parameters()
        self.update_plot()

    def validate_and_fix_parameters(self):
        """Ensure SOS < EOS and other logical constraints"""
        if self.x is None or len(self.x) == 0:
            return
            
        # Get current parameters
        baseline, amplitude, start, end, growth_rate, senescence_rate = self.params
        
        # Find approximate peak time
        x_range = np.linspace(min(self.x), max(self.x), 1000)
        y_pred = double_logistic(x_range, *self.params)
        peak_time = x_range[np.argmax(y_pred)]
        
        # Fix SOS/EOS relationship
        if start >= end:
            # If SOS >= EOS, adjust EOS to be after peak
            self.params[3] = max(peak_time + 10, start + 20)
            # Update the spinbox
            self.spin_boxes[3].blockSignals(True)
            self.spin_boxes[3].setValue(self.params[3])
            self.spin_boxes[3].blockSignals(False)

    def estimate_smart_parameters(self):
        """Estimate biologically meaningful initial parameters"""
        if self.x is None or self.y is None or len(self.x) == 0:
            return self.init_params.copy()
        
        try:
            # Sort data by x values
            sorted_indices = np.argsort(self.x)
            x_sorted = self.x[sorted_indices]
            y_sorted = self.y[sorted_indices]
            
            # Baseline: minimum value (winter baseline)
            baseline = np.percentile(y_sorted, 5)
            
            # Amplitude: difference between max and baseline
            max_val = np.percentile(y_sorted, 95)
            amplitude = max_val - baseline
            
            # Find approximate season boundaries
            # Smooth the data to find general trend
            window_size = max(5, len(y_sorted) // 20)
            if len(y_sorted) >= window_size:
                y_smooth = np.convolve(y_sorted, np.ones(window_size)/window_size, mode='valid')
                x_smooth = x_sorted[window_size//2:len(x_sorted)-window_size//2+1]
            else:
                y_smooth = y_sorted
                x_smooth = x_sorted
            
            # Find peak
            peak_idx = np.argmax(y_smooth)
            peak_time = x_smooth[peak_idx]
            
            # SOS: Find where curve starts rising (threshold method)
            threshold_rise = baseline + 0.2 * amplitude
            rise_indices = np.where(y_smooth >= threshold_rise)[0]
            if len(rise_indices) > 0:
                sos = x_smooth[rise_indices[0]]
            else:
                sos = np.percentile(x_sorted, 15)
            
            # EOS: Find where curve starts declining significantly
            threshold_decline = baseline + 0.8 * amplitude
            decline_indices = np.where((y_smooth <= threshold_decline) & (x_smooth > peak_time))[0]
            if len(decline_indices) > 0:
                eos = x_smooth[decline_indices[0]]
            else:
                eos = np.percentile(x_sorted, 85)
            
            # Ensure logical ordering: SOS < Peak < EOS
            if sos >= peak_time:
                sos = peak_time - (max(self.x) - min(self.x)) * 0.1
            if eos <= peak_time:
                eos = peak_time + (max(self.x) - min(self.x)) * 0.1
            
            # Growth and senescence rates (moderate values)
            growth_rate = 0.1
            senescence_rate = 0.05
            
            return [baseline, amplitude, sos, eos, growth_rate, senescence_rate]
            
        except Exception as e:
            print(f"Error in parameter estimation: {e}")
            return self.init_params.copy()

    def update_plot(self):
        """Redraw the plot with current data and parameters"""
        self.ax.clear()
        if self.x is not None and self.y is not None and len(self.x) > 0:
            try:
                # Plot observed data
                self.ax.scatter(self.x, self.y, color='#1f77b4', s=60, alpha=0.8, 
                               edgecolor='k', label='Observed')
                
                # Plot fitted curve
                t_fit = np.linspace(min(self.x), max(self.x), 500)
                y_fit = double_logistic(t_fit, *self.params)
                self.ax.plot(t_fit, y_fit, 'r-', linewidth=2, label='Fitted Curve')
                
                # Plot key points if enabled
                if self.show_key_points_cb.isChecked():
                    x_range = max(self.x) - min(self.x)
                    y_range = max(self.y) - min(self.y)
                    x_min, x_max = min(self.x), max(self.x)
                    y_min, y_max = min(self.y), max(self.y)
                    
                    # SOS (Start of Season)
                    sos = self.params[2]
                    if x_min <= sos <= x_max:
                        y_sos = double_logistic(sos, *self.params)
                        self.ax.plot(sos, y_sos, 'ro', markersize=8)
                        self.ax.annotate('SOS', xy=(sos, y_sos), 
                                        xytext=(sos - 0.05*x_range, y_sos + 0.05*y_range),
                                        arrowprops=dict(arrowstyle="->", color='black'),
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.9))
                    
                    # EOS (End of Season)
                    eos = self.params[3]
                    if x_min <= eos <= x_max:
                        y_eos = double_logistic(eos, *self.params)
                        self.ax.plot(eos, y_eos, 'ro', markersize=8)
                        self.ax.annotate('EOS', xy=(eos, y_eos), 
                                        xytext=(eos + 0.02*x_range, y_eos + 0.05*y_range),
                                        arrowprops=dict(arrowstyle="->", color='black'),
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.9))
                    
                    # Peak (maximum value in fitted curve)
                    peak_idx = np.argmax(y_fit)
                    t_peak = t_fit[peak_idx]
                    y_peak = y_fit[peak_idx]
                    
                    self.ax.plot(t_peak, y_peak, 'ro', markersize=8)
                    self.ax.annotate('Peak', xy=(t_peak, y_peak), 
                                    xytext=(t_peak, y_peak + 0.08*y_range),
                                    arrowprops=dict(arrowstyle="->", color='black'),
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.9))
                
                # Plot grouping data if available
                if self.groups is not None and self.grouping_config:
                    start_col = self.grouping_config['start_col']
                    end_col = self.grouping_config['end_col']
                    label_col = self.grouping_config['label_col']
                    color_col = self.grouping_config['color_col']
                    if color_col and color_col in self.groups.columns:
                        unique_colors = self.groups[color_col].nunique()
                        colors = plt.cm.tab10(np.linspace(0, 1, unique_colors))
                        color_map = {val: colors[i] for i, val in enumerate(self.groups[color_col].unique())}
                    else:
                        colors = plt.cm.tab10(np.linspace(0, 1, len(self.groups)))
                        color_map = {}
                    for idx, group in self.groups.iterrows():
                        start = group[start_col]
                        end = group[end_col] if end_col and end_col in group else start
                        color = color_map[group[color_col]] if color_col and color_col in group else colors[idx % len(colors)]
                        if start != end:
                            self.ax.axvspan(start, end, alpha=0.15, color=color, zorder=0)
                        else:
                            self.ax.axvline(start, color=color, linestyle='--', alpha=0.7)
                        if label_col in group:
                            label = group[label_col]
                            mid_point = (start + end) / 2
                            self.ax.text(mid_point, max(self.y)*0.95, label, 
                                        color=color, ha='center', fontsize=8)
                
                # Calculate and display statistics
                y_pred = double_logistic(self.x, *self.params)
                r2 = r2_score(self.y, y_pred)
                rmse = np.sqrt(mean_squared_error(self.y, y_pred))
                
                # Update statistics labels
                self.r2_label.setText(f"R²: {r2:.4f}")
                self.rmse_label.setText(f"RMSE: {rmse:.4f}")
                params_str = "<br>".join([f"<b>{name}</b>: {val:.4f}" for name, val in zip(self.param_names, self.params)])
                self.params_label.setText(f"<b>Parameters:</b><br>{params_str}")
                self.params_label.setTextFormat(Qt.RichText)
                
                # Set plot labels with actual column names
                self.ax.set_xlabel(self.x_col_name)
                y_label = self.y_col_combo.currentText()
                self.ax.set_ylabel(y_label if y_label else "Value")
                
                self.ax.set_title("Double Logistic Curve Fit")
                self.ax.legend()
                self.ax.grid(True, linestyle='--', alpha=0.3)
            
            except Exception as e:
                self.statusBar().showMessage(f"Plot error: {str(e)}")
        self.canvas.draw()

    def update_data_columns(self):
        if self.df is None:
            return
        x_col = self.x_col_combo.currentText()
        y_col = self.y_col_combo.currentText()
        if not x_col or not y_col:
            return
        try:
            df_clean = self.df[[x_col, y_col]].dropna()
            self.x = df_clean[x_col].values
            self.y = df_clean[y_col].values
            self.current_y_col = y_col
            self.x_col_name = x_col
            
            # Update bounds based on actual data range
            x_min, x_max = np.min(self.x), np.max(self.x)
            y_min, y_max = np.min(self.y), np.max(self.y)
            
            # Smart parameter estimation
            self.params = self.estimate_smart_parameters()
            
            # Update parameter bounds to be more appropriate for the data
            self.bounds = (
                [y_min - 0.1, 0, x_min, x_min, 0.01, 0.01], 
                [y_max + 0.1, (y_max-y_min)*3, x_max, x_max, 1, 1]
            )
            
            # Update spin box bounds and values
            for i, spin in enumerate(self.spin_boxes):
                spin.setMinimum(self.bounds[0][i])
                spin.setMaximum(self.bounds[1][i])
                spin.setValue(self.params[i])
            
            self.update_plot()
            self.statusBar().showMessage(f"Using columns: X={x_col}, Y={y_col}")
            
        except Exception as e:
            self.statusBar().showMessage(f"Column error: {str(e)}")

    def optimize_fit(self):
        """Optimize parameters using constrained optimization"""
        if self.x is None or self.y is None or len(self.x) == 0:
            return
            
        try:
            # Get current parameters as initial guess
            p0 = self.params.copy()
            
            # Determine which parameters are locked
            fixed_mask = [lock.isChecked() for lock in self.locks]
            
            # Create bounds for variable parameters
            lower_bounds = []
            upper_bounds = []
            variable_p0 = []
            param_indices = []
            
            for i in range(len(self.param_names)):
                if not fixed_mask[i]:
                    lower_bounds.append(self.bounds[0][i])
                    upper_bounds.append(self.bounds[1][i])
                    variable_p0.append(p0[i])
                    param_indices.append(i)
            
            if len(variable_p0) == 0:
                self.statusBar().showMessage("All parameters locked - nothing to optimize")
                return
            
            # Define objective function with constraints
            def objective(params_var):
                full_params = p0.copy()
                for i, idx in enumerate(param_indices):
                    full_params[idx] = params_var[i]
                
                # Add penalty for SOS >= EOS
                baseline, amplitude, start, end, growth_rate, senescence_rate = full_params
                penalty = 0
                if start >= end:
                    penalty += 1000 * (start - end + 1)**2
                
                try:
                    y_pred = double_logistic(self.x, *full_params)
                    mse = np.mean((self.y - y_pred)**2)
                    return mse + penalty
                except:
                    return 1e10
            
            # Define constraints
            constraints = []
            
            # SOS < EOS constraint
            start_idx = param_indices.index(2) if 2 in param_indices else None
            end_idx = param_indices.index(3) if 3 in param_indices else None
            
            if start_idx is not None and end_idx is not None:
                def sos_eos_constraint(params_var):
                    return params_var[end_idx] - params_var[start_idx] - 5  # EOS must be at least 5 days after SOS
                constraints.append({'type': 'ineq', 'fun': sos_eos_constraint})
            
            # Perform optimization
            result = minimize(
                objective,
                variable_p0,
                method='SLSQP',
                bounds=list(zip(lower_bounds, upper_bounds)),
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                # Update parameters
                for i, idx in enumerate(param_indices):
                    self.spin_boxes[idx].setValue(result.x[i])
                self.statusBar().showMessage("Optimization successful with constraints")
            else:
                # Fallback to curve_fit without constraints
                self.optimize_fit_fallback()
                
        except Exception as e:
            self.statusBar().showMessage(f"Optimization error: {str(e)}")
            QMessageBox.warning(self, "Optimization Error", f"Failed to optimize fit:\n{str(e)}")

    def optimize_fit_fallback(self):
        """Fallback optimization without constraints"""
        try:
            p0 = self.params.copy()
            fixed_mask = [lock.isChecked() for lock in self.locks]
            lower_bounds = []
            upper_bounds = []
            variable_p0 = []
            
            for i in range(len(self.param_names)):
                if not fixed_mask[i]:
                    lower_bounds.append(self.bounds[0][i])
                    upper_bounds.append(self.bounds[1][i])
                    variable_p0.append(p0[i])
            
            def wrapped_func(t, *variable_params):
                full_params = p0.copy()
                var_idx = 0
                for i in range(len(p0)):
                    if not fixed_mask[i]:
                        full_params[i] = variable_params[var_idx]
                        var_idx += 1
                return double_logistic(t, *full_params)
            
            popt, _ = curve_fit(
                wrapped_func, 
                self.x, 
                self.y, 
                p0=variable_p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=5000
            )
            
            # Update only variable parameters
            var_idx = 0
            for i in range(len(self.param_names)):
                if not fixed_mask[i]:
                    self.spin_boxes[i].setValue(popt[var_idx])
                    var_idx += 1
            
            # Validate the result
            self.validate_and_fix_parameters()
            self.statusBar().showMessage("Optimization completed (fallback method)")
            
        except Exception as e:
            self.statusBar().showMessage(f"Fallback optimization failed: {str(e)}")

    def export_outputs(self):
        if self.x is None or self.y is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Outputs", "curvefit_outputs", "Excel Files (*.xlsx)"
        )
        if not path:
            return
        try:
            x_col = self.x_col_combo.currentText()
            y_col = self.y_col_combo.currentText()
            y_fit = double_logistic(self.x, *self.params)
            out_df = pd.DataFrame({
                x_col: self.x, 
                y_col: self.y, 
                'Fitted': y_fit
            })
            param_df = pd.DataFrame({
                'Parameter': self.param_names,
                'Value': self.params,
                'Bounds_Lower': self.bounds[0],
                'Bounds_Upper': self.bounds[1]
            })
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                out_df.to_excel(writer, index=False, sheet_name='Fitted_Curve')
                param_df.to_excel(writer, index=False, sheet_name='Fitted_Params')
                if self.groups is not None:
                    self.groups.to_excel(writer, index=False, sheet_name='Grouping_Data')
            plot_path = path.replace(".xlsx", ".png")
            self.canvas.figure.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.statusBar().showMessage(f"Saved outputs to {path}")
            QMessageBox.information(self, "Success", f"Outputs saved successfully:\n{path}\n{plot_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Save error: {str(e)}")
            QMessageBox.critical(self, "Save Error", f"Failed to save outputs:\n{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CurveFitApp()
    window.show()
    sys.exit(app.exec_())

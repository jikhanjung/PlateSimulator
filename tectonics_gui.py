import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QSpinBox, QHBoxLayout)
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from plate_tectonics_sim import PlateTectonicsSimulator  # Previous simulator class


class TectonicsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.simulator = PlateTectonicsSimulator(grid_size=100, num_plates=5)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Plate Tectonics Simulator')
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = QHBoxLayout()
        
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.toggle_simulation)
        control_panel.addWidget(self.start_button)
        
        speed_label = QLabel('Speed:')
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 1000)
        self.speed_spin.setValue(100)
        control_panel.addWidget(speed_label)
        control_panel.addWidget(self.speed_spin)
        
        plates_label = QLabel('Plates:')
        self.plates_spin = QSpinBox()
        self.plates_spin.setRange(2, 20)
        self.plates_spin.setValue(5)
        self.plates_spin.valueChanged.connect(self.reset_simulation)
        control_panel.addWidget(plates_label)
        control_panel.addWidget(self.plates_spin)
        
        reset_button = QPushButton('Reset')
        reset_button.clicked.connect(self.reset_simulation)
        control_panel.addWidget(reset_button)
        
        layout.addLayout(control_panel)
        
        # Visualization layout
        vis_layout = QHBoxLayout()
        
        self.elevation_plot = pg.PlotWidget()
        self.elevation_plot.setTitle('Elevation')
        self.elevation_img = pg.ImageItem()
        self.elevation_plot.addItem(self.elevation_img)
        vis_layout.addWidget(self.elevation_plot)
        
        self.temp_plot = pg.PlotWidget()
        self.temp_plot.setTitle('Temperature')
        self.temp_img = pg.ImageItem()
        self.temp_plot.addItem(self.temp_img)
        vis_layout.addWidget(self.temp_plot)
        
        self.plate_plot = pg.PlotWidget()
        self.plate_plot.setTitle('Plate IDs')
        self.plate_img = pg.ImageItem()
        self.plate_plot.addItem(self.plate_img)
        vis_layout.addWidget(self.plate_plot)
        
        layout.addLayout(vis_layout)
        
        # Color maps
        self.elevation_cmap = pg.ColorMap(pos=np.linspace(0, 1, 3),
                                        color=[(0, 0, 255), (0, 255, 0), (139, 69, 19)])
        self.temp_cmap = pg.ColorMap(pos=np.linspace(0, 1, 3),
                                   color=[(0, 0, 255), (255, 255, 0), (255, 0, 0)])
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.simulation_running = False
        
        self.update_plots()
        
    def toggle_simulation(self):
        if self.simulation_running:
            self.timer.stop()
            self.start_button.setText('Start')
        else:
            self.timer.start(1000 // self.speed_spin.value())
            self.start_button.setText('Stop')
        self.simulation_running = not self.simulation_running
        
    def reset_simulation(self):
        self.simulator = PlateTectonicsSimulator(
            grid_size=100, 
            num_plates=self.plates_spin.value()
        )
        self.update_plots()
        
    def update_simulation(self):
        self.simulator.step()
        self.update_plots()
        
    def safe_normalize(self, array):
        """Safely normalize array to [0,1] range, handling edge cases"""
        min_val = np.nanmin(array)
        max_val = np.nanmax(array)
        
        if np.isnan(min_val) or np.isnan(max_val):
            return np.zeros_like(array)
            
        range_val = max_val - min_val
        if range_val == 0:
            return np.zeros_like(array)
            
        return (array - min_val) / range_val
        
    def update_plots(self):
        # Update elevation plot with safe normalization
        elevation_norm = self.safe_normalize(self.simulator.elevation)
        self.elevation_img.setImage(elevation_norm)
        self.elevation_img.setLookupTable(self.elevation_cmap.getLookupTable())
        
        # Update temperature plot with safe normalization
        temp_norm = self.safe_normalize(self.simulator.temperature)
        self.temp_img.setImage(temp_norm)
        self.temp_img.setLookupTable(self.temp_cmap.getLookupTable())
        
        # Update plate ID plot
        self.plate_img.setImage(self.simulator.plate_ids)

def main():
    app = QApplication(sys.argv)
    gui = TectonicsGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
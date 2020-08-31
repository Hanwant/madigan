import os
import ast
import json
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout
import pyqtgraph as pg
from .dash_synth_base import Ui_MainWindow
from .utils import make_dark_palette
from madigan.environments.synth import test_env
from madigan.utils import load_json, save_json



class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, data=None, parent=None, exp_name=''):
        super().__init__(parent)
        self.setupUi(self)
        self.data = data
        self.colours = {'eq': (218, 112, 214), 'returns': (255, 228, 181)}

        # CONTROL SIGNALS/SLOTS #########################################
        # Config from file
        self.config_path = Path('/home/hemu/madigan/madigan/environments')/'test.json'
        self.FilenameLabel.setText('/'.join(self.config_path.parts[-1:]))
        self.LoadConfigButton.clicked.connect(self.load_config)
        self.SaveConfigButton.clicked.connect(self.save_config)

        with open(self.config_path, 'r') as f:
            self.exp_config = json.load(f)
        self.ParamsEdit.setText(str(self.exp_config))
        self.ParamsEdit.textChanged.connect(self.update_config)

        # Run exp based on config
        self.RunCommand.clicked.connect(self.run_exp)

        # PLOTS ########################################################
        self.plot_layout = QtGui.QGridLayout()
        self.plot = pg.GraphicsLayoutWidget(show=True, title=f'Run: {exp_name}')
        self.subplots = {}
        self.subplots['prices'] = self.plot.addPlot(title='Prices', bottom='time', left='price')
        self.subplots['eq'] = self.plot.addPlot(title='Equity', bottom='time', left='Denomination currency')
        self.subplots['returns'] = self.plot.addPlot(title='Returns', bottom='time', left='returns (proportion)')
        self.subplots['prices'].showGrid(1, 1)
        self.subplots['eq'].showGrid(1, 1)
        self.subplots['eq'].setLabels()
        self.subplots['returns'].showGrid(1, 1)
        self.eq_line = self.subplots['eq'].plot(y=[])
        self.returns_line = self.subplots['returns'].plot(y=[])
        self.price_plots = {}

        self.current_pos_line = pg.InfiniteLine(movable=True)
        self.subplots['eq'].addItem(self.current_pos_line)
        self.current_pos_line.sigPositionChanged.connect(self.update_portfolio)

        self.plot_layout.addWidget(self.plot)
        self.PlotsWidget.setLayout(self.plot_layout)
        # self.update_data(data)
        # PORTFOLIO TABLES ###################################################
        self.PositionsTable.setColumnCount(2)
        self.CashTable.setColumnCount(1)
        self.CashTable.setHorizontalHeaderLabels(['$'])
        # self.PositionsTable.setHorizontalHeaderLabels(['asset', 'pos'])

        if data is not None:
            self.update_data(data)

    def clear_data(self):
        self.eq_line.setData(y=[])
        self.returns_line.setData(y=[])
        for asset, plot in self.price_plots.items():
            plot.setData(y=[])

    def add_data(self, data):
        # self.subplots['eq'].plot(y=data['eq'], pen=self.colours['eq'])
        # self.subplots['returns'].plot(y=data['returns'], pen=self.colours['returns'])
        self.eq_line.setData(y=data['eq'], pen=self.colours['eq'])
        self.returns_line.setData(y=data['returns'], pen=self.colours['returns'])
        for i, asset in enumerate(data['assets']):
            if asset not in self.price_plots:
                self.price_plots[asset] = self.subplots['prices'].plot(y=data['prices'][:, i], pen=(i, data['prices'].shape[1]))
            else:
                self.price_plots[asset].setData(y=data['prices'][:, i], pen=(i, data['prices'].shape[1]))
        self.data = data
        self.current_pos_line.setValue(len(self.data['positions'])-1)
        self.update_portfolio()

    def update_data(self, data):
        self.clear_data()
        self.add_data(data)

    def update_portfolio(self):
        self.PositionsTable.setRowCount(len(self.data['assets']))
        current_timepoint = int(self.current_pos_line.value())
        try:
            cash = QtGui.QTableWidgetItem(str(self.data['cash'][current_timepoint]))
            margin = QtGui.QTableWidgetItem(str(self.data['margin'][current_timepoint]))
            eq = QtGui.QTableWidgetItem(str(self.data['eq'][current_timepoint]))
            self.CashTable.setItem(0, 0, cash)
            self.CashTable.setItem(1, 0, margin)
            self.CashTable.setItem(2, 0, eq)
        except IndexError:
            pass
        for i, asset in enumerate(self.data['assets']):
            self.PositionsTable.setItem(i, 0, QtGui.QTableWidgetItem(asset))
            # if 0 < current_timepoint <= len(self.positions)-1:
            try:
                pos = self.data['positions'][current_timepoint][i]
                self.PositionsTable.setItem(i, 1, QtGui.QTableWidgetItem(str(pos)))
            except IndexError:
                continue

    def run_exp(self):
        data = test_env(self.exp_config, agent=None, eps=1.)
        self.update_data(data)

    def update_config(self):
        self.exp_config = ast.literal_eval(self.ParamsEdit.toPlainText())
        print('saved config')
        # print(self.exp_config)

    def load_config(self):
        self.config_path = self.get_file(load=True)
        self.exp_config = load_json(self.config_path)
        self.ParamsEdit.setText(str(self.exp_config))

    def save_config(self):
        self.config_path = self.get_file(save=True)
        save_json(self.exp_config, self.config_path, write_mode='w')

    def get_file(self, load=False, save=False):
        assert load != save, "specify either load=True or save=True"
        if load:
            fname = QtGui.QFileDialog.getOpenFileName(self, 'Choose config file', '/home/hemu/madigan/')[0]
        if save:
            fname = QtGui.QFileDialog.getSaveFileName(self, 'Choose config file', '/home/hemu/madigan/')[0]
        return fname



def run_dash(data):

    app = QApplication([])

    app.setStyle('Fusion')
    palette = make_dark_palette()
    app.setPalette(palette)
    # main = Main(data)
    # main = QtWidgets.QMainWindow()
    # ui = Ui_MainWindow()
    # ui.setupUi(main)
    main = MainWindow(data=data, exp_name='test')
    main.show()

    app.exec_()


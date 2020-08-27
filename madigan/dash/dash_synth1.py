from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout
import pyqtgraph as pg
from .dash_synth_base import Ui_MainWindow
from .utils import make_dark_palette
from madigan.environments.synth import test_env



class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, data=None, parent=None, exp_name=''):
        super().__init__(parent)
        self.setupUi(self)
        self.data = data
        self.colours = {'eq': (218, 112, 214), 'returns': (255, 228, 181)}

        # CONTROL SIGNAL/SLOTS #########################################
        self.RunCommand.clicked.connect(self.run_exp)



        # PLOTS ########################################################
        self.plot_layout = QtGui.QGridLayout()
        self.plot = pg.GraphicsLayoutWidget(show=True, title=f'Run: {exp_name}')
        self.subplots = {}
        self.subplots['prices'] = self.plot.addPlot(title='Prices')
        self.subplots['eq'] = self.plot.addPlot(title='Equity', )
        self.subplots['returns'] = self.plot.addPlot(title='Returns', )
        self.subplots['prices'].showGrid(1, 1)
        self.subplots['eq'].showGrid(1, 1)
        self.subplots['returns'].showGrid(1, 1)

        self.plot_layout.addWidget(self.plot)
        self.PlotsWidget.setLayout(self.plot_layout)
        # self.update_data(data)

    def clear_data(self):
        self.subplots['eq'].clear()
        self.subplots['returns'].clear()
        self.subplots['prices'].clear()

    def add_data(self, data):
        self.subplots['eq'].plot(y=data['eq'], pen=self.colours['eq'])
        self.subplots['returns'].plot(y=data['returns'], pen=self.colours['returns'])
        for asset in range(data['prices'].shape[1]):
            self.subplots['prices'].plot(y=data['prices'][:, asset], pen=(asset, data['prices'].shape[1]))

    def update_data(self, data):
        self.clear_data()
        self.add_data(data)
    def run_exp(self):
        data = test_env()
        self.update_data(data)

    def update_params(self):
        pass

    def load_params(self):
        pass

    def save_params(self):
        pass


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


from functools import partial
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtWidgets import QCheckBox, QGroupBox, QGridLayout
from PyQt5.QtWidgets import QMessageBox
import pyqtgraph as pg
import numpy as np
from .utils import make_dark_palette

class Plot(pg.GraphicsLayout):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.plot1 = self.addPlot(0, 0, 1, 1)
        self.plot1.plot(data['eq'], pen='b')

        self.plot_options = QGroupBox()
        self.option_layout = QVBoxLayout()
        self.options = {i:QCheckBox(option) for i, option in enumerate(['eq', 'returns'])}
        for option in self.options.values():
            option.setChecked(True)
            self.option_layout.addWidget(option)
        self.plot_options.setLayout(self.option_layout)

class Main(QWidget):
    def __init__(self, data, exp_name='', env=None):
        super(Main, self).__init__()
        # layout = QHBoxLayout()
        self.data = data
        # self.colours = {'eq': (255, 255, 255), 'returns': (0, 255, 0)}
        self.colours = {'eq': 'lightskyblue', 'returns': 'orchid'}
        self.colours = {'eq': (218, 112, 214), 'returns': (255, 228, 181)}
        # self.asset_colours = cycle([''])

        layout = QGridLayout()

        self.plot = pg.GraphicsLayoutWidget(show=True, title=f'Run: {exp_name}')
        self.plot.setWindowTitle(f'Run: {exp_name}')
        self.subplots = {}
        self.subplots['prices'] = self.plot.addPlot(title='prices')
        for asset in range(data['prices'].shape[1]):
            self.subplots['prices'].plot(y=data['prices'][:, asset], pen=(asset, data['prices'].shape[1]))
        self.subplots['eq'] = self.plot.addPlot(title='Equity', )
        self.subplots['eq'].plot(y=data['eq'], pen=self.colours['eq'])
        self.subplots['returns'] = self.plot.addPlot(title='Returns', )
        self.subplots['returns'].plot(y=data['returns'], pen=self.colours['returns'])

        self.option_layout = QVBoxLayout()
        self.options = {option: QCheckBox(option) for i, option in enumerate(['eq', 'returns'])}
        self.option_layout = QVBoxLayout()
        for option in self.options.values():
            option.setChecked(True)
            self.option_layout.addWidget(option)
            option.stateChanged.connect(self.update_visibility)
        layout.addWidget(self.plot)
        layout.addLayout(self.option_layout, 0, 1)

        self.setLayout(layout)
        self.show()

    def update_visibility(self):
        for option, checkbox in self.options.items():
            self.subplots[option].clear()
            if checkbox.isChecked():
                print(f'plotting {option}')
                self.subplots[option].plot(self.data[option], pen=self.colours[option])
            else:
                print(f'removing {option}')


def run_dash(data):

    app = QApplication([])

    app.setStyle('Fusion')
    palette = make_dark_palette()
    app.setPalette(palette)
    main = Main(data)

    app.exec_()






if __name__ == "__main__":
    # run_dash(data)
    pass




import os
import ast
import json
import yaml
import logging
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout
import pyqtgraph as pg
from .training_base import Ui_MainWindow
from .utils import make_dark_palette
from ..run.test import test
from ..run.train import Trainer
from ..utils.config import make_config, Config, load_config, save_config

def deref_dict(dic):
    d = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            v = deref_dict(v)
        d[k] = v
    return d

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, data=None, parent=None, exp_name=''):
        super().__init__(parent)
        self.setupUi(self)
        # VARIABLES ##########################################################
        self.training_data = None
        self.test_data = None
        self.colours = {'equity': (218, 112, 214), 'returns': (255, 228, 181),
                        'loss': (242, 242, 242), 'G_t': (0, 255, 255), 'Q_t': (255, 86, 0),
                        'rewards': (242, 242, 242)}
        self.trainer = None
        self.compSource = "Local"

        # SERVER COMMUINICATION ###############################################
        default_server = {'name': 'local',
                          'address': 'self',
                          'port': None,
                          'pid': str(os.getpid()),
                          'status': 'Live',
                          'keyPath': None,
                          'pass': None,
                          }
        self.servers = [default_server]
        self.ServerInfo.setColumnCount(4)
        self.ServerInfo.setRowCount(len(self.servers))
        # self.ServerInfo.setData()
        self.LocalRadio.toggled.connect(self.compSourceToggle)
        self.ServerRadio.toggled.connect(self.compSourceToggle)
        self.serverInfoCols = ['name', 'pid', 'location', 'status']
        self.ServerInfo.setHorizontalHeaderLabels(self.serverInfoCols)
        for row, server in enumerate(self.servers):
            for col, name in enumerate(self.serverInfoCols):
                if name == 'location':
                    item = QtGui.QTableWidgetItem(f"{server['address']}:{server['port']}")
                else:
                    item = QtGui.QTableWidgetItem(server[name])
                self.ServerInfo.setItem(row, col, item)
        self.update_server_status()


        # CONTROL SIGNALS/SLOTS #########################################
        # CONFIG
        self.config_path = Path('/home/hemu/madigan/madigan/environments')/'test.json'
        self.FilenameLabel.setText('/'.join(self.config_path.parts[-1:]))
        self.LoadConfigButton.clicked.connect(self.load_config)
        self.SaveConfigButton.clicked.connect(self.save_config)

        self.exp_config = load_config(self.config_path)
        self.exp_config = make_config(env_type="Synth", nsteps=100_000, agent_type="DQN",
                                      double_dqn=True, min_tf=64)
        # import ipdb; ipdb.set_trace()
        self.ParamsEdit.setText(str(yaml.safe_dump(deref_dict(self.exp_config))))
        self.ParamsEdit.textChanged.connect(self.update_config)

        # Run exp based on config
        self.TestCommand.clicked.connect(lambda: self.run_job('test'))
        self.TrainCommand.clicked.connect(lambda: self.run_job('train'))

        # EXPERIMENT ##########################################################
        # self.TrainLog.addItem('Train')
        # self.TestLog.addItem('Test')

        # PLOTS ########################################################
        # 2 rows of 3 columns
        # row 1: prices, equity, returns
        # row 2: loss, G_t, Q_t
        # Training #####################################################
        self.train_plot_layout = QtGui.QGridLayout()
        self.train_plot = pg.GraphicsLayoutWidget(show=True, title=f'Run: {exp_name}')
        self.subplots = {}
        self.subplots['loss'] = self.train_plot.addPlot(title='Loss', bottom='step', left='Loss')
        self.subplots['values'] = self.train_plot.addPlot(title='Values', bottom='step', left='Value')
        self.subplots['rewards'] = self.train_plot.addPlot(title='Rewards', bottom='step', left='rewards')
        self.subplots['loss'].showGrid(1, 1)
        self.subplots['values'].showGrid(1, 1)
        # self.subplots['values'].setLabels()
        self.subplots['rewards'].showGrid(1, 1)
        self.loss_line = self.subplots['loss'].plot(y=[])
        self.subplots['values'].addLegend()
        self.Gt_line = self.subplots['values'].plot(y=[], name='Gt')
        self.Qt_line = self.subplots['values'].plot(y=[], name='Qt')
        self.rewards_line = self.subplots['rewards'].plot(y=[])

        self.train_plot_layout.addWidget(self.train_plot)
        self.PlotsWidgetTrain.setLayout(self.train_plot_layout)
        #  Testing #####################################################
        self.test_plot_layout = QtGui.QGridLayout()
        self.test_plot = pg.GraphicsLayoutWidget(show=True, title=f'Run: {exp_name}')
        self.subplots = {}
        self.subplots['prices'] = self.test_plot.addPlot(title='Prices', bottom='time', left='price')
        self.subplots['equity'] = self.test_plot.addPlot(title='Equity', bottom='time', left='Denomination currency')
        self.subplots['returns'] = self.test_plot.addPlot(title='Returns', bottom='time', left='returns (proportion)')
        self.subplots['prices'].showGrid(1, 1)
        self.subplots['equity'].showGrid(1, 1)
        self.subplots['equity'].setLabels()
        self.subplots['returns'].showGrid(1, 1)
        self.eq_line = self.subplots['equity'].plot(y=[])
        self.returns_line = self.subplots['returns'].plot(y=[])

        self.current_pos_line = pg.InfiniteLine(movable=True)
        self.subplots['equity'].addItem(self.current_pos_line)
        self.current_pos_line.sigPositionChanged.connect(self.update_portfolio)

        self.price_plots = {}

        self.test_plot_layout.addWidget(self.test_plot)
        self.PlotsWidgetTest.setLayout(self.test_plot_layout)
        # PORTFOLIO TABLES ###################################################
        self.PositionsTable.setColumnCount(2)
        self.CashTable.setColumnCount(1)
        self.CashTable.setHorizontalHeaderLabels(['$'])
        # self.PositionsTable.setHorizontalHeaderLabels(['asset', 'pos'])

    def clear_training_data(self):
        self.loss_line.setData(y=[])
        self.Gt_line.setData(y=[])
        self.Qt_line.setData(y=[])
        self.rewards_line.setData(y=[])

    def clear_test_data(self):
        self.eq_line.setData(y=[])
        self.returns_line.setData(y=[])
        for asset, plot in self.price_plots.items():
            plot.setData(y=[])

    def add_training_data(self, data):
        self.training_data = data
        if len(data) == 0:
            logging.warning("train data is empty")
            data = {k: [] for k in ('loss', 'G_t', 'Q_t', 'rewards')}
        self.loss_line.setData(y=data['loss'], pen=self.colours['loss'])
        # G_t and Q_t are np/torch arrays - .mean().item() works on both
        G_t = [dat.mean().item() for dat in data['G_t']]
        Q_t = [dat.mean().item() for dat in data['Q_t']]
        rewards = [dat.mean().item() for dat in data['rewards']]
        self.Gt_line.setData(y=G_t, pen=self.colours['G_t'])
        self.Qt_line.setData(y=Q_t, pen=self.colours['Q_t'])
        self.rewards_line.setData(y=rewards, pen=self.colours['rewards'])

    def add_test_data(self, data):
        self.test_data = data
        if len(data) == 0:
            logging.warning("test data is empty")
            data = {k: [] for k in ('equity', 'returns', 'prices', 'positions')}
            data['assets'] = 0
        self.eq_line.setData(y=data['equity'], pen=self.colours['equity'])
        self.returns_line.setData(y=data['returns'], pen=self.colours['returns'])
        for i, asset in enumerate(data['assets']):
            if asset not in self.price_plots:
                self.price_plots[asset] = self.subplots['prices'].plot(y=data['prices'][:, i], pen=(i, data['prices'].shape[1]))
            else:
                self.price_plots[asset].setData(y=data['prices'][:, i], pen=(i, data['prices'].shape[1]))
        self.current_pos_line.setValue(len(data['positions'])-1)
        self.update_portfolio()

    def update_training_data(self, data):
        self.clear_training_data()
        self.add_training_data(data)

    def update_test_data(self, data):
        self.clear_test_data()
        self.add_test_data(data)

    def update_server_status(self):
        for row in range(self.ServerInfo.rowCount()):
            status = self.ServerInfo.item(row, 3).text()
            if status == 'Live':
                for col in range(self.ServerInfo.columnCount()):
                    self.ServerInfo.item(row, col).setBackground(QColor('#0a290a'))
            elif status == "Not Responding":
                for col in range(self.ServerInfo.columnCount()):
                    self.ServerInfo.item(row, col).setBackground(QColor('#999999'))

    # def update_data(self, data):
    #     self.clear_data()
    #     self.add_data(data)

    def update_portfolio(self):
        self.PositionsTable.setRowCount(len(self.test_data['assets']))
        current_timepoint = int(self.current_pos_line.value())
        try:
            cash = QtGui.QTableWidgetItem(str(self.test_data['cash'][current_timepoint]))
            margin = QtGui.QTableWidgetItem(str(self.test_data['margin'][current_timepoint]))
            eq = QtGui.QTableWidgetItem(str(self.test_data['equity'][current_timepoint]))
            self.CashTable.setItem(0, 0, cash)
            self.CashTable.setItem(1, 0, margin)
            self.CashTable.setItem(2, 0, eq)
        except IndexError:
            pass
        for i, asset in enumerate(self.test_data['assets']):
            self.PositionsTable.setItem(i, 0, QtGui.QTableWidgetItem(asset))
            # if 0 < current_timepoint <= len(self.positions)-1:
            try:
                pos = self.test_data['positions'][current_timepoint][i]
                self.PositionsTable.setItem(i, 1, QtGui.QTableWidgetItem(str(pos)))
            except IndexError:
                import traceback
                traceback.print_exc()
                continue

    def run_job(self, action):
        assert action in ('train', 'test')
        try:
            if self.compSource == "Local":
                if self.trainer is None:
                    print('Initializing environment and agent')
                    self.trainer = Trainer.from_config(self.exp_config)
                if action == 'test':
                    test_metrics = test(self.trainer.agent, self.trainer.env, nsteps=1000)
                    self.update_test_data(test_metrics)
                elif action == 'train':
                    train_metrics, test_metrics = self.trainer.train()
                    self.update_training_data(train_metrics)
            elif self.compSource == "Server":
                # if action == 'test':
                    # self.socket.send_pyobj({'signal': 'job', 'action': 'test',
                    #                         'config': self.exp_config})
                # elif action == 'train':
                    # self.socket.send_pyobj({'signal': 'job', 'action': 'train',
                    #                         'config': self.exp_config})
                raise NotImplementedError
        except KeyboardInterrupt:
            pass
        except Exception as E:
            import traceback; traceback.print_exc()
        finally:
            print('destructing env and agent')
            self.trainer = None


    def compSourceToggle(self):
        if self.LocalRadio.isChecked():
            self.compSource = "Local"
        elif self.ServerRadio.isChecked():
            self.compSource = "Server"

    def update_config(self):
        # self.exp_config = Config(ast.literal_eval(self.ParamsEdit.toPlainText()))
        self.exp_config = Config(yaml.safe_load(self.ParamsEdit.toPlainText()))
        print('saved config')
        # print(self.exp_config)

    def load_config(self):
        self.config_path = self.get_file(load=True)
        self.exp_config = Config(load_config(self.config_path))
        self.ParamsEdit.setText(str(yaml.safe_dump(deref_dict(self.exp_config))))

    def save_config(self):
        self.config_path = self.get_file(save=True)
        save_config(self.exp_config, self.config_path, write_mode='w')

    def get_file(self, load=False, save=False):
        assert load != save, "specify either load=True or save=True"
        if load:
            fname = QtGui.QFileDialog.getOpenFileName(self, 'Choose config file', '/home/hemu/madigan/')[0]
        if save:
            fname = QtGui.QFileDialog.getSaveFileName(self, 'Choose config file', '/home/hemu/madigan/')[0]
        return fname

def run_dash():

    app = QApplication([])

    app.setStyle('Fusion')
    palette = make_dark_palette()
    app.setPalette(palette)
    main = MainWindow(exp_name='test')
    main.show()

    app.exec_()


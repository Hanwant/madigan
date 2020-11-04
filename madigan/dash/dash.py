import os
import ast
import json
import yaml
import logging
from pathlib import Path
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout
import pyqtgraph as pg
from .dash_ui import Ui_MainWindow
from .utils import make_dark_palette
from .plots import make_train_plots, make_test_episode_plots, make_test_history_plots
from ..run.test import test
from ..run.trainer import Trainer
from ..utils.config import make_config, Config, load_config, save_config


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # VARIABLES ##########################################################
        self.exp_config = make_config(experiment_id="TestDash",
                                      # overwrite_exp=True,
                                      data_source_type="OU",
                                      agent_type="DQN",
                                      assets=["ou1"])
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
        self.config_path = Path('/media/hemu/Data/Markets/farm/TestDash')/'test.json'
        self.FilenameLabel.setText('/'.join(self.config_path.parts[-1:]))
        self.LoadConfigButton.clicked.connect(self.load_config)
        self.SaveConfigButton.clicked.connect(self.save_config)

        # self.exp_config = load_config(self.config_path)
        # save_config(self.exp_config, self.config_path)
        self.ParamsEdit.setText(str(yaml.safe_dump(self.exp_config.to_dict())))
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
        self.train_plots = make_train_plots(self.exp_config.agent_type)
        self.PlotsWidgetTrain.setLayout(self.train_plots)
        #  Testing #####################################################
        self.test_episode_plots = make_test_episode_plots(self.exp_config.agent_type)
        self.PlotsWidgetTestEpisodes.setLayout(self.test_episode_plots)
        #  Test History #####################################################
        self.test_history_plots = make_test_history_plots(self.exp_config.agent_type)
        self.PlotsWidgetTest.setLayout(self.test_history_plots)


    def update_server_status(self):
        for row in range(self.ServerInfo.rowCount()):
            status = self.ServerInfo.item(row, 3).text()
            if status == 'Live':
                for col in range(self.ServerInfo.columnCount()):
                    self.ServerInfo.item(row, col).setBackground(QColor('#0a290a'))
            elif status == "Not Responding":
                for col in range(self.ServerInfo.columnCount()):
                    self.ServerInfo.item(row, col).setBackground(QColor('#999999'))

    def run_job(self, action):
        assert action in ('train', 'test')
        try:
            if self.compSource == "Local":
                if self.trainer is None:
                    print('Initializing environment and agent')
                    self.trainer = Trainer.from_config(self.exp_config)
                if action == 'test':
                    eps=self.exp_config.agent_config.greedy_eps_testing
                    test_metric = test(self.trainer.agent, self.trainer.env,
                                       nsteps=1000, eps=eps)
                    self.update_full_episode_data(test_metric)
                    self.update_episode_data(test_metric)
                elif action == 'train':
                    train_metrics, test_metrics = self.trainer.train()
                    self.update_training_data(train_metrics)
                    self.update_test_data(test_metrics)
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
            del self.trainer
            self.trainer=None

    def set_datapath(self, path):
        self.datapath = path
        self.train_plots.set_datapath(path)
        self.test_episode_plots.set_datapath(path)
        # self.test_history_plots.set_datapath(path)

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
        self.config_path = Path(self.get_file(load=True))
        self.exp_config = Config(load_config(self.config_path))
        self.ParamsEdit.setText(str(yaml.safe_dump(self.exp_config.to_dict())))
        self.set_datapath(Path(self.exp_config.experiment_path)/'logs')
        print()
        self.FilenameLabel.setText('/'.join(self.config_path.parts[-2:]))

    def save_config(self):
        self.config_path = self.get_file(save=True)
        save_config(self.exp_config, self.config_path, write_mode='w')
        self.load_config()

    def get_file(self, load=False, save=False):
        assert load != save, "specify either load=True or save=True"
        if load:
            fname = QtGui.QFileDialog.getOpenFileName(self, 'Choose config file',
                                                      '/home/hemu/madigan/')[0]
        if save:
            fname = QtGui.QFileDialog.getSaveFileName(self, 'Choose config file',
                                                      '/home/hemu/madigan/')[0]
        return fname

    def __del__(self):
        """ Save local settings to pkl here """
        super().__del__()

def run_dash():

    app = QApplication([])

    app.setStyle('Fusion')
    palette = make_dark_palette()
    app.setPalette(palette)
    main = MainWindow()
    main.show()

    app.exec_()


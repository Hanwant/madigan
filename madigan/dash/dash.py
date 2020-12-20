import os
import ast
import json
import yaml
import pickle
import logging
from pathlib import Path
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout
import pyqtgraph as pg
from .dash_ui import Ui_MainWindow
from .trainer_thread import TrainerWorker
from .utils import make_dark_palette, delete_layout
from .plots import make_train_plots, make_test_episode_plots, make_test_history_plots
# from ..run.test import test
from ..run.trainer import Trainer
from ..utils.config import make_config, Config, load_config, save_config


class MainWindow( Ui_MainWindow, QtWidgets.QMainWindow):
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
        self.cache_filepath = Path(os.getcwd())/'.cache.dash'

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
        self.LocalRadio.toggled.connect(self.setCompSource)
        self.ServerRadio.toggled.connect(self.setCompSource)
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

        # HOUSE KEEPING
        self.config_path = None
        self.experiments_path = None
        self.load_user_cache()  # loads saved config_path and experiments_path

        # Experiment CONTROL SIGNALS/SLOTS #########################################
        # MENU BAR
        self.actionSet_Experiments_Folder.triggered.connect(
            self.choose_experiments_path)
        # CONFIG TAB
        # self.FilenameLabel.setText('/'.join(self.config_path.parts[-1:]))
        self.LoadConfigButton.clicked.connect(self.load_experiment)
        self.SaveConfigButton.clicked.connect(self.save_config)

        # self.exp_config = load_config(self.config_path)
        # save_config(self.exp_config, self.config_path)
        self.ParamsEdit.setText(str(yaml.safe_dump(self.exp_config.to_dict())))
        self.ParamsEdit.textChanged.connect(self.update_config)
        # EXP TAB
        self.ExperimentsList.currentRowChanged.connect(
            lambda: self.load_experiment(
                self.experiments_path/
                self.ExperimentsList.item(
                    self.ExperimentsList.currentRow()).text()/
                'config.yaml')
            )
        if self.experiments_path is None:
            self.choose_experiments_path()
        else:
            self.load_experiments_list()

        # Run exp based on config
        self.TestCommand.clicked.connect(lambda: self.run_job('test'))
        self.TrainCommand.clicked.connect(lambda: self.run_job('train'))
        self.StopCommand.clicked.connect(self.stop_job)
        # Branching into new experiments
        self.BranchButton.clicked.connect(self.new_branch)
        self.BranchCheckpointButton.clicked.connect(
            self.new_branch_checkpoint)

        self.plots = {'train': None, 'test_episodes': None,
                      'test_history': None}  # custom plots
        self.make_plots()
        self.worker = None
        self.threadpool = QtCore.QThreadPool()
        self.centralwidget.destroyed.connect(self.save_user_cache)
        if self.config_path is not None:
            print('loading')
            self.load_experiment(self.config_path)

    def connect_to_server(self, host, port):
        pass

    def choose_experiments_path(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Choose Folder containing experiments',
            os.getcwd())
        self.experiments_path = Path(path) if path != '' else None
        self.load_experiments_list()

    def new_branch(self):
        if self.worker is None:
            response = QtWidgets.QMessageBox.information(
                self, 'Branching semantics',
                'No worker exists to perform branching, create idle worker?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if response == QtWidgets.QMessageBox.Yes:
                self.make_worker('idle')
                self.new_branch()
        else:
            branch, ok = QtWidgets.QInputDialog.getText(
                self, 'Pick Name for New Branch', 'Branch Name: ',
                QtWidgets.QLineEdit.Normal, self.exp_config.experiment_id)
            if ok:
                self.worker.trainer.branch_experiment(branch)
                self.load_experiment(
                    Path(self.worker.trainer.config.basepath) /
                    self.worker.trainer.config.experiment_id /
                    'config.yaml')

    def new_branch_checkpoint(self):
        if self.worker is None:
            response = QtWidgets.QMessageBox.information(
                self, 'Branching Semantics',
                'No worker exists to perform branching, create idle worker?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if response == QtWidgets.QMessageBox.Yes:
                self.make_worker('idle')
                self.new_branch_checkpoint()
        else:
            checkpoints = [check[1].name for check in self.checkpoints]
            checkpoint, ok = QtWidgets.QInputDialog.getItem(
                self, 'Pick checkpoint to branch from', 'Checkpoint: ',
                checkpoints, 0, False)
            if ok:
                self.worker.trainer.branch_from_checkpoint(checkpoint)
                self.load_experiment(
                    Path(self.worker.trainer.config.basepath) /
                    self.worker.trainer.config.experiment_id /
                    'config.yaml')

    def load_experiments_list(self):
        experiments = [p.name for p in Path(self.experiments_path).iterdir()
                       if p.is_dir()]
        experiments = sorted(experiments,
                             key=lambda x:
                             (self.experiments_path/x).stat().st_mtime,
                             reverse=True)
        self.ExperimentsList.clear()
        for exp in experiments:
            self.ExperimentsList.addItem(exp)

    def load_checkpoints_list(self):
        assert self.exp_config is not None, "can't load checkpoints without config"
        model_path = Path(self.exp_config.basepath)/ \
            self.exp_config.experiment_id/'models'
        self.checkpoints = [(int(f.stem.split('_')[1]), f)
                            for f in model_path.iterdir()
                            if len(f.stem.split('_')) > 1]
        self.checkpoints.sort(reverse=True)
        self.CheckpointsList.clear()
        for check in self.checkpoints:
            self.CheckpointsList.addItem(check[1].name)

    def make_plots(self):
        self.remove_plots()  # remove existing plots if they exist
        self.plots['train'] = make_train_plots(self.exp_config.agent_type)
        self.plots['test_episode'] = \
            make_test_episode_plots(self.exp_config.agent_type)
        self.plots['test_history'] = \
            make_test_history_plots(self.exp_config.agent_type)
        self.PlotsWidgetTrain.setLayout(self.plots['train'])
        self.PlotsWidgetTestEpisodes.setLayout(self.plots['test_episode'])
        self.PlotsWidgetTest.setLayout(self.plots['test_history'])

    def remove_plots(self):
        for plot in self.plots.values():
            if plot is not None:
                delete_layout(plot)
                QtCore.QObjectCleanupHandler().add(plot)

    def update_server_status(self):
        for row in range(self.ServerInfo.rowCount()):
            status = self.ServerInfo.item(row, 3).text()
            if status == 'Live':
                for col in range(self.ServerInfo.columnCount()):
                    self.ServerInfo.item(row, col).setBackground(QColor('#0a290a'))
            elif status == "Not Responding":
                for col in range(self.ServerInfo.columnCount()):
                    self.ServerInfo.item(row, col).setBackground(QColor('#999999'))

    def make_worker(self, action: str):
        if action not in TrainerWorker.action_types:
            raise ValueError("aciton must be either 'test' or 'train")
        self.worker = TrainerWorker(self.exp_config, action)
        self.worker.signals.test_episode_metrics.connect(
            self.plots['test_episode'].set_data)
        self.worker.signals.test_history_metrics.connect(
            self.plots['test_history'].set_data)
        self.worker.signals.train_metrics.connect(self.plots['train'].set_data)
        return self.worker

    def remove_worker(self):
        if self.worker is not None:
            taken = self.threadpool.tryTake(self.worker)
            if taken:
                self.worker = None
            return taken
        return True

    def run_job(self, action):
        assert action in TrainerWorker.action_types
        try:
            if self.compSource == "Local":
                worker = self.make_worker(action)
                self.threadpool.start(worker)
            elif self.compSource == "Server":
                # if action == 'test':
                    # self.socket.send_pyobj({'signal': 'job', 'action': 'test',
                    #                         'config': self.exp_config})
                # elif action == 'train':
                    # self.socket.send_pyobj({'signal': 'job', 'action': 'train',
                    #                         'config': self.exp_config})
                raise NotImplementedError(
                    f"comp source {self.compSource} has not been impl")
        except KeyboardInterrupt:
            pass
        except Exception as E:
            import traceback; traceback.print_exc()
        finally:
            pass

    def stop_job(self):
        if self.compSource == "Local":
            if self.worker is not None:
                print('setting terminate early to true')
                self.worker.trainer.terminate_early = True

    def set_datapath(self, path):
        self.datapath = path
        for plot in self.plots.values():
            if plot is not None:
                plot.set_datapath(path)

    def setCompSource(self):
        if self.LocalRadio.isChecked():
            self.compSource = "Local"
        elif self.ServerRadio.isChecked():
            self.compSource = "Server"

    def update_config(self):
        # self.exp_config = Config(ast.literal_eval(self.ParamsEdit.toPlainText()))
        self.exp_config = Config(yaml.safe_load(self.ParamsEdit.toPlainText()))
        # print(self.exp_config)

    def load_experiment(self, config_path=None):
        config_path = config_path or Path(self.get_file(load=True))
        try:
            if config_path is not None and config_path != "":
                self.config_path = config_path
                old_agent_type = self.exp_config.agent_type
                self.exp_config = Config(load_config(self.config_path))
                self.ParamsEdit.setText(
                    str(yaml.safe_dump(self.exp_config.to_dict())))
                path = Path(self.exp_config.basepath)/ \
                    self.exp_config.experiment_id/'logs'
                if self.exp_config.agent_type != old_agent_type:
                    self.make_plots()
                self.FilenameLabel.setText('/'.join(self.config_path.parts[-2:]))
                # set new datapath and let plots load new data
                self.set_datapath(path)
            self.load_checkpoints_list()
            if not self.remove_worker():
                print('worker could not be removed - might still be running')
        except Exception as E:
            self.log(E)

    def log(self, message):
        """
        Sends message to log box
        """
        print(message)

    def save_config(self):
        config_path = self.get_file(save=True)
        if config_path != "":
            self.config_path = config_path
            save_config(self.exp_config, self.config_path, write_mode='w')
            self.load_config()

    def get_file(self, load=False, save=False):
        assert load != save, "specify either load=True or save=True"
        if load:
            # fname = QtGui.QFileDialog.getOpenFileName(self, 'Choose config file',
            #                                           Path(os.getcwd()))[0]
            fname = QtGui.QFileDialog.getOpenFileName(self, 'Choose config file',
                                                      os.getcwd())[0]
        if save:
            # fname = QtGui.QFileDialog.getSaveFileName(self, 'Choose config file',
            #                                           Path(os.getcwd()))[0]
            fname = QtGui.QFileDialog.getSaveFileName(self, 'Choose config file',
                                                      os.getcwd())[0]
        return fname

    def save_user_cache(self):
        if self.config_path is not None:
            settings = {'config_path': self.config_path,
                        'experiments_path': self.experiments_path}
            with open(self.cache_filepath, 'wb') as f:
                pickle.dump(settings, f)
                print('saving user cache')

    def load_user_cache(self):
        if self.cache_filepath.is_file():
            print('loading cache')
            with open(self.cache_filepath, 'rb') as f:
                settings = pickle.load(f)
            self.config_path = settings['config_path']
            self.experiments_path = settings['experiments_path']
            # print('setting config_path: ', self.config_path)

    def closeEvent(self, event):
        """ Save local settings to pkl here """
        self.save_user_cache()
        super(QtWidgets.QMainWindow, self).closeEvent(event)

def run_dash():

    app = QApplication([])

    app.setStyle('Fusion')
    palette = make_dark_palette()
    app.setPalette(palette)
    main = MainWindow()
    main.show()

    app.exec_()


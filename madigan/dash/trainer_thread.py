# from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QRunnable, pyqtSlot, QObject
from PyQt5.QtCore import pyqtSignal
import pandas as pd

from madigan.run.trainer import Trainer

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    test_history_metrics = pyqtSignal(pd.DataFrame)
    test_episode_metrics = pyqtSignal(pd.DataFrame)
    train_metrics = pyqtSignal(pd.DataFrame)
    progress = pyqtSignal(int)

class TrainerWorker(QRunnable):
    action_types = ('test', 'train', 'idle')
    def __init__(self, config, action: str, train_steps: int = None,
                 test_steps: int = None):
        super().__init__()
        self.trainer = Trainer.from_config(config)
        self.action = action
        self.train_steps = train_steps
        self.test_steps = test_steps
        self.signals = WorkerSignals()
        if action not in self.action_types:
            raise NotImplementedError(
                f"action must be one of: {self.action_types}")
    # def update_config(self, config):
    #     self.trainer.config = config

    @pyqtSlot()
    def run(self):
        """
        Runs a task.
        task = options are 'test' or 'train'
        nsteps = number of steps for testing or training.
                 If None, the default value is taken from config
        """
        print('running_task in thread: ', QThread.currentThreadId())
        print(self.action)
        try:
            if self.action == "test":
                # return self.test(self.test_steps)
                test_metrics = self.trainer.test(self.test_steps)
                self.signals.test_episode_metrics.emit(test_metrics)
            elif self.action == "train":
                # return self.train(self.train_steps)
                train_metrics, test_metrics = \
                    self.trainer.train(self.train_steps)
                test_episode = self.trainer.load_latest_test_run()
                self.signals.test_history_metrics.emit(test_metrics)
                self.signals.test_episode_metrics.emit(test_episode)
                self.signals.train_metrics.emit(train_metrics)
            elif self.action == "idle":
                pass
            else:
                raise NotImplementedError("only 'test' or 'train' accepted as tasks")
        except Exception as E:
            import traceback; traceback.print_exc()
        finally:
            self.signals.finished.emit()

    def save_test_metrics(self, metrics_df: pd.DataFrame):
        """
        Only test metrics which are interleaved with training
        get automatically saved. This function allows saving
        of metrics for test episodes prompted manually, via gui
        """
        self.trainer.save_logs({}, metrics_df)

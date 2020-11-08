# from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QRunnable, pyqtSlot, QObject
from PyQt5.QtCore import pyqtSignal
import pandas as pd

from madigan.run.trainer import Trainer

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    test_metrics = pyqtSignal(pd.DataFrame)
    train_metrics = pyqtSignal(pd.DataFrame)
    progress = pyqtSignal(int)

class TrainerWorker(QRunnable):
    def __init__(self, config, action: str, train_steps: int = None,
                 test_steps: int = None):
        super().__init__()
        self.trainer = Trainer.from_config(config)
        self.action = action
        self.train_steps = train_steps
        self.test_steps = test_steps
        self.signals = WorkerSignals()

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
                self.signals.test_metrics.emit(test_metrics)
            elif self.action == "train":
                # return self.train(self.train_steps)
                train_metrics, test_metrics = \
                    self.trainer.train(self.train_steps)
                self.signals.test_metrics.emit(test_metrics)
                self.signals.train_metrics.emit(train_metrics)
            else:
                raise NotImplementedError("only 'test' or 'train' accepted as tasks")
        except Exception as E:
            import traceback; traceback.print_exc()
        finally:
            self.signals.finished.emit()


    # def train(self, nsteps: int = None) -> pd.DataFrame:
    #     """
    #     Performs training and returns results
    #     """
    #     nsteps = nsteps or self.trainer.train_steps
    #     metrics_df = self.trainer.train(nsteps)
    #     return metrics_df

    # def test(self, nsteps: int = None) -> pd.DataFrame:
    #     """
    #     Performs testing and returns results
    #     """
    #     nsteps = nsteps or self.trainer.test_steps
    #     metrics_df = self.trainer.test(nsteps)
    #     return metrics_df

    def save_test_metrics(self, metrics_df: pd.DataFrame):
        self.trainer.save_logs({}, metrics_df)

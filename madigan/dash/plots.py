from pathlib import Path
from itertools import product
import traceback

import numpy as np
import pandas as pd
import h5py

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtGui import QTableWidget, QTableWidgetItem, QGridLayout
from PyQt5.QtGui import QListWidget, QLabel
import pyqtgraph as pg
import matplotlib.pyplot as plt

from madigan.utils.plotting_pub import make_figs, save_fig


###############################################################################
############# Factory Methods #################################################
###############################################################################
def make_train_plots(agent_type, title=None, **kw):
    if agent_type in ("DQN", "DQNCURL", "DQNReverser", "DQNController",
                      "DQNRecurrent", "IQN", "IQNCURL", "IQNReverser",
                      "IQNController", "DQNAE", "DQNMixedActions",
                      "IQNMixedActions"):
        return TrainPlotsDQN(title=title, **kw)
    if agent_type in ("DDPG", "DDPGDiscretized"):
        return TrainDDPG(title=title, **kw)
    if agent_type in ("SACDiscrete", "SACD"):
        return TrainPlotsSACD(title=title, **kw)
    raise NotImplementedError(
        f"Train plot widget for agent_type: {agent_type}"+\
        " has not been implemented")


def make_test_episode_plots(agent_type, title=None, **kw):
    if agent_type in ("DQN", "DQNCURL", "DQNReverser", "DQNController",
                      "DQNRecurrent", "IQN", "IQNCURL", "IQNReverser",
                      "IQNController", "DQNAE", "DQNMixedActions",
                      "IQNMixedActions"):
        return TestEpisodePlotsDQN(title=title, **kw)
    if agent_type in ("DDPG", "DDPGDiscretized"):
        return TestEpisodeDDPG(title=title, **kw)
    if agent_type in ("SACDiscrete", "SACD"):
        return TestEpisodePlotsSACD(title=title, **kw)
    raise NotImplementedError(
        f"Test episode plot widget for agent_type: {agent_type}"+\
        " has not been implemented")


def make_test_history_plots(agent_type, title=None, **kw):
    if agent_type in ("DQN", "DQNCURL", "DQNReverser", "DQNController",
                      "DQNRecurrent", "IQN", "IQNCURL", "IQNReverser",
                      "IQNController", "DQNAE", "DQNMixedActions",
                      "IQNMixedActions"):
        return TestHistoryPlotsDQN(title=title, **kw)
    if agent_type in ("DDPG", "DDPGDiscretized"):
        return TestHistoryDDPG(title=title, **kw)
    if agent_type in ("SACDiscrete", "SACD"):
        return TestHistoryPlotsSACD(title=title, **kw)
    raise NotImplementedError(
        f"Test History plot widget for agent_type: {agent_type}"+\
        " has not been implemented")


############# Training Plots  ######################################################
####################################################################################


class TrainPlots(QGridLayout):
    """ Base class for train plots """
    def __init__(self, title=None):
        super().__init__()
        self.datapath = None
        self.data = None
        self.idxs = None
        self.graphs = pg.GraphicsLayoutWidget(show=True, title=title)
        self.addWidget(self.graphs)
        self.colours = {
            'reward': (255, 228, 181),
            'reward_mean': (255, 0, 0),
            'loss': (242, 242, 242)
        }
        self.plots = {}
        self.lines = {}
        # self.plots['loss'] = self.graphs.addPlot(title='Loss',
        #                                          bottom='step',
        #                                          left='Loss')
        self.plots['loss'] = pg.PlotItem(title='loss',
                                         bottom='step',
                                         left='loss')
        self.graphs.addItem(self.plots['loss'])
        self.plots['loss'].showGrid(1, 1)
        self.plots['loss'].addLegend()
        self.plots['running_reward'] = self.graphs.addPlot(
            title='Episode Rewards', bottom='step', left='reward')
        self.plots['running_reward'].showGrid(1, 1)
        self.plots['running_reward'].addLegend()

        self.lines['loss'] = self.plots['loss'].plot(y=[])
        self.lines['running_reward'] = self.plots['running_reward'].plot(
            y=[], name='rewards')
        self.reward_mean_window = 50000
        self.lines['running_reward_mean'] = self.plots['running_reward'].plot(
            y=[], name=f'rewards_{self.reward_mean_window}_window_average')
        self.plots['running_reward'].setLabels()
        self.link_x_axes()

    def link_x_axes(self):
        for name, plot in self.plots.items():
            if name != "loss":
                plot.setXLink(self.plots['loss'])

    def clear_plots(self):
        for line in self.lines.values():
            line.setData(y=[])

    def process_data(self, data):
        """ Converts data to ndarrays """
        self.data = dict(data.items())
        for label, arr in self.data.items():
            self.data[label] = np.array(arr)

    def set_data(self, data):
        self.process_data(data)
        if len(data) == 0:
            print("train data is empty")
            data = {k: [] for k in self.lines.keys()}
        if len(self.data['loss']) > 100_000:
            self.idxs = np.sort(
                np.random.choice(len(self.data['loss']),
                                 100_000,
                                 replace=False))
        else:
            self.idxs = np.arange(len(self.data['loss']))
        # size = len(data['loss'])
        # if size > 500000:
        #    sparse_idx = np.random.choice(size / 10_000_000)
        self.lines['loss'].setData(x=self.idxs,
                                   y=self.data['loss'][self.idxs],
                                   pen=self.colours['loss'])
        rewards = self.data['running_reward'][self.idxs]
        rewards_mean = pd.Series(
            self.data['running_reward'][self.idxs]).ewm(20000).mean()
        rewards_mean = np.nan_to_num(rewards_mean.values)
        self.lines['running_reward'].setData(x=self.idxs,
                                             y=rewards,
                                             pen=self.colours['reward'])
        self.lines['running_reward_mean'].setData(
            x=self.idxs,
            y=rewards_mean,
            pen=pg.mkPen({
                'color': self.colours['reward_mean'],
                'width': 2
            }))

    def set_datapath(self, path):
        self.datapath = Path(path)
        self.load_from_hdf()

    def load_from_hdf(self, path=None):
        path = path or self.datapath / 'train.hdf5'
        if path is not None:
            data = pd.read_hdf(path, key='train')
            self.set_data(data)

    def export_plots(self, export_path):
        figs = make_figs(self.data)
        for label, fig in figs.items():
            ax = fig.axes[0]
            xrange_label = self.plots['loss'].viewRange()[0][0]
            if label in self.plots.keys():
                x_range, y_range = self.plots[label].viewRange()
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
            save_fig(fig, export_path / f'{label}_{int(xrange_label)}.pdf')
        plt.close('all')


class TrainPlotsDQN(TrainPlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({'Gt': (0, 255, 255), 'Qt': (255, 86, 0)})
        self.plots['values'] = self.graphs.addPlot(title='Values',
                                                   bottom='step',
                                                   left='Value')
        self.plots['values'].showGrid(1, 1)
        self.plots['values'].addLegend()
        self.lines['Gt'] = self.plots['values'].plot(y=[], name='Gt')
        self.lines['Qt'] = self.plots['values'].plot(y=[], name='Qt')
        self.plots['values'].setLabels()

        self.link_x_axes()

    def set_data(self, data):
        super().set_data(data)
        if data is None or len(data) == 0:
            return
        self.lines['Gt'].setData(x=self.idxs,
                                 y=self.data['Gt'][self.idxs],
                                 pen=self.colours['Gt'])
        self.lines['Qt'].setData(x=self.idxs,
                                 y=self.data['Qt'][self.idxs],
                                 pen=self.colours['Qt'])


class TrainDDPG(TrainPlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({'Gt': (0, 255, 255), 'Qt': (255, 86, 0)})
        self.colours.update({
            'loss_critic': (255, 0, 0),
            'loss_actor': (0, 255, 0)
        })
        self.plots['values'] = self.graphs.addPlot(title='Values',
                                                   bottom='step',
                                                   left='Value')
        self.plots['values'].showGrid(1, 1)
        self.plots['values'].addLegend()
        self.lines['Gt'] = self.plots['values'].plot(y=[], name='Gt')
        self.lines['Qt'] = self.plots['values'].plot(y=[], name='Qt')
        self.plots['values'].setLabels()

        del self.lines['loss']
        self.lines['loss_critic'] = self.plots['loss'].plot(y=[],
                                                            name='loss_critic')
        self.lines['loss_actor'] = self.plots['loss'].plot(y=[],
                                                           name='loss_actor')
        self.plots['loss'].setLabels()

        self.link_x_axes()

    def set_data(self, data):
        self.data = data
        if len(data) == 0:
            print("train data is empty")
            data = {k: [] for k in self.lines.keys()}

        self.lines['loss_critic'].setData(
            x=self.idxs,
            y=self.data['loss_critic'][self.idxs],
            pen=self.colours['loss_critic'][self.idxs])
        self.lines['loss_actor'].setData(x=self.idxs,
                                         y=self.data['loss_actor'][self.idxs],
                                         pen=self.colours['loss_actor'])
        # rewards = data['running_reward'][self.idxs]
        # rewards_mean = pd.Series(data['running_reward']).ewm(20000).mean()
        # rewards_mean = np.nan_to_num(rewards_mean.values)
        # self.lines['running_reward'].setData(y=rewards,
        #                                      pen=self.colours['reward'])
        # self.lines['running_reward_mean'].setData(
        #     y=rewards_mean,
        #     pen=pg.mkPen({
        #         'color': self.colours['reward_mean'],
        #         'width': 2
        #     }))
        self.lines['Gt'].setData(x=self.idxs,
                                 y=self.data['Gt'][self.idxs],
                                 pen=self.colours['Gt'])
        self.lines['Qt'].setData(x=self.idxs,
                                 y=self.data['Qt'][self.idxs],
                                 pen=self.colours['Qt'])


class TrainPlotsSACD(TrainPlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({
            'Gt': (0, 255, 255),
            'Qt1': (255, 86, 0),
            'Qt2': (86, 255, 0),
            'entropy': (255, 0, 0),
            'entropy_temp': (0, 255, 0)
        })
        self.colours.update({
            'loss_critic1': (255, 0, 0),
            'loss_critic2': (0, 0, 255),
            'loss_actor': (0, 255, 0),
            'loss_entropy': (255, 255, 0)
        })
        self.plots['values'] = self.graphs.addPlot(title='Values',
                                                   bottom='step',
                                                   left='Value')
        self.plots['entropy'] = self.graphs.addPlot(title='Entropy',
                                                    bottom='step')
        self.lines['entropy'] = self.plots['entropy'].plot(y=[],
                                                           name='entropy')
        self.lines['entropy_temp'] = self.plots['entropy'].plot(
            y=[], name='entropy_temp')
        self.plots['entropy'].setLabels()

        self.plots['values'].showGrid(1, 1)
        self.plots['values'].addLegend()
        self.lines['Gt'] = self.plots['values'].plot(y=[], name='Gt')
        self.lines['Qt1'] = self.plots['values'].plot(y=[], name='Qt1')
        self.lines['Qt2'] = self.plots['values'].plot(y=[], name='Qt2')
        self.plots['values'].setLabels()

        del self.lines['loss']
        self.lines['loss_critic1'] = self.plots['loss'].plot(
            y=[], name='loss_critic1')
        self.lines['loss_critic2'] = self.plots['loss'].plot(
            y=[], name='loss_critic2')
        self.lines['loss_actor'] = self.plots['loss'].plot(y=[],
                                                           name='loss_actor')
        self.lines['loss_entropy'] = self.plots['loss'].plot(
            y=[], name='loss_entropy')
        self.plots['loss'].setLabels()

        self.link_x_axes()

    def set_data(self, data):
        print("AC ")
        self.data = data
        if len(data) == 0:
            print("train data is empty")
            data = {k: [] for k in self.lines.keys()}

        self.lines['loss_critic1'].setData(y=data['loss_critic1'],
                                           pen=self.colours['loss_critic1'])
        self.lines['loss_critic2'].setData(y=data['loss_critic2'],
                                           pen=self.colours['loss_critic2'])
        self.lines['loss_actor'].setData(y=data['loss_actor'],
                                         pen=self.colours['loss_actor'])
        self.lines['loss_entropy'].setData(y=data['loss_entropy'],
                                           pen=self.colours['loss_entropy'])
        rewards = data['running_reward']
        rewards_mean = pd.Series(data['running_reward']).ewm(20000).mean()
        rewards_mean = np.nan_to_num(rewards_mean.values)
        self.lines['running_reward'].setData(y=rewards,
                                             pen=self.colours['reward'])
        self.lines['running_reward_mean'].setData(
            y=rewards_mean,
            pen=pg.mkPen({
                'color': self.colours['reward_mean'],
                'width': 2
            }))
        self.lines['Gt'].setData(y=data['Gt'], pen=self.colours['Gt'])
        self.lines['Qt1'].setData(y=data['Qt1'], pen=self.colours['Qt1'])
        self.lines['Qt2'].setData(y=data['Qt2'], pen=self.colours['Qt2'])

        self.lines['entropy'].setData(y=data['entropy'],
                                      pen=self.colours['entropy'])
        self.lines['entropy_temp'].setData(y=data['entropy_temp'],
                                           pen=self.colours['entropy_temp'])


####################################################################################
############# Test Episode Plots  ##################################################
####################################################################################


class TestEpisodePlots(QGridLayout):
    def __init__(self, title=None, **kw):
        super().__init__()
        self.graphs = pg.GraphicsLayoutWidget(show=True, title=title)
        self.datapath = None
        self.episodes = []
        self.data = None
        self.assets = None
        self.colours = {
            'equity': (218, 112, 214),
            'reward': (242, 242, 242),
            'cash': (0, 255, 255),
            'margin': (255, 86, 0)
        }
        self.heatmap_colors = [(0, 0, 85), (85, 0, 0)]
        cmap = pg.ColorMap(pos=np.linspace(-1., 1., 2),
                           color=self.heatmap_colors)
        self.plots = {}
        self.lines = {}
        self.plots['prices'] = self.graphs.addPlot(title='Prices',
                                                   bottom='time',
                                                   left='prices',
                                                   colspan=2)
        self.plots['equity'] = self.graphs.addPlot(
            title='Equity',
            bottom='time',
            left='Denomination currency',
            row=0,
            col=2,
            colspan=2)
        self.plots['reward'] = self.graphs.addPlot(title='Rewards',
                                                   bottom='time',
                                                   left='reward',
                                                   row=0,
                                                   col=4,
                                                   colspan=2)
        self.lines['ledgerNormed'] = pg.ImageItem(np.empty(shape=(1, 1, 1)))
        self.lines['ledgerNormed'].setLookupTable(cmap.getLookupTable())
        self.plots['ledgerNormed'] = self.graphs.addPlot(title="Ledger",
                                                         colspan=2)
        self.plots['ledgerNormed'].addItem(self.lines['ledgerNormed'])
        self.plots['ledgerNormed'].showAxis('left', False)
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.lines['ledgerNormed'])
        self.graphs.addItem(hist, row=0, col=8, colspan=1)
        self.plots['equity'].setLabels()
        self.lines['equity'] = self.plots['equity'].plot(y=[])
        self.lines['reward'] = self.plots['reward'].plot(y=[])
        self.lines['prices'] = {}
        self.current_pos_line = pg.InfiniteLine(movable=True)
        self.plots['equity'].addItem(self.current_pos_line)
        self.plots['cash'] = self.graphs.addPlot(title="cash",
                                                 row=1,
                                                 col=0,
                                                 colspan=2)
        self.plots['availableMargin'] = self.graphs.addPlot(
            title="available margin", row=1, col=2, colspan=2)
        self.plots['transaction'] = self.graphs.addPlot(title="transaction",
                                                        row=1,
                                                        col=4,
                                                        colspan=2)
        self.lines['cash'] = self.plots['cash'].plot(y=[])
        self.lines['availableMargin'] = self.plots['availableMargin'].plot(
            y=[])
        self.lines['transaction'] = {}

        self.plots['prices'].showGrid(1, 1)
        self.plots['equity'].showGrid(1, 1)
        self.plots['reward'].showGrid(1, 1)
        self.plots['prices'].showGrid(1, 1)
        self.plots['cash'].showGrid(1, 1)
        self.plots['availableMargin'].showGrid(1, 1)
        self.plots['transaction'].showGrid(1, 1)
        limits = np.finfo('float64')
        # self.plots['transaction'].setYRange(limits.min, limits.max)
        self.plots['ledgerNormed'].showGrid(1, 1)

        self.episode_table = QListWidget()
        self.episode_table.setWindowTitle('Episode Name')
        self.episode_table.setStyleSheet("background-color:rgb(99, 102, 49) ")
        self.accounting_table = QTableWidget(5, 1)
        self.accounting_table.setVerticalHeaderLabels([
            'Equity', 'Balance', 'Cash', 'Available Margin', 'Used Margin',
            'Net PnL'
        ])
        self.accounting_table.setHorizontalHeaderLabels(['Accounting'])
        # self.accounting_table.horizontalHeaderItem(0).setTextAlignment(
        #     QtCore.Qt.AlignJustify)  # doesn't work
        self.accounting_table.setStyleSheet(
            "background-color:rgb(44, 107, 42) ")
        self.positions_table = QTableWidget(0, 2)
        self.positions_table.setHorizontalHeaderLabels(
            ['Ledger', 'Ledger Normed'])
        self.positions_table.setStyleSheet("background-color:rgb(23, 46, 67) ")

        self.asset_picker = QListWidget()
        self.asset_picker.setWindowTitle('Episode Name')
        self.asset_picker.setStyleSheet("background-color:rgb(99, 102, 49) ")
        self.asset_picker.setSelectionMode(QListWidget.MultiSelection)
        self.asset_picker.itemSelectionChanged.connect(
            self._set_data_with_variable_assets)
        self.asset_picker.setStyleSheet("background-color:rgb(23, 46, 67) ")

        self.tear_sheet = pg.TableWidget(editable=False, sortable=False)

        # self.episode_table.horizontalHeader().setStretchLastSection(True)
        self.accounting_table.horizontalHeader().setStretchLastSection(True)
        self.positions_table.horizontalHeader().setStretchLastSection(True)

        self.episode_table_label = QLabel("Test Episodes")
        self.asset_picker_label = QLabel("Assets")

        self.tables = QtGui.QVBoxLayout()
        self.tables.addWidget(self.episode_table_label)
        self.tables.addWidget(self.episode_table)
        self.tables.addWidget(self.accounting_table)
        self.tables.addWidget(self.positions_table)
        self.tables.addWidget(self.asset_picker_label)
        self.tables.addWidget(self.asset_picker)

        self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(self.graphs)
        self.tables_widget = QtGui.QWidget()
        self.tables_widget.setLayout(self.tables)
        self.splitter.addWidget(self.tables_widget)
        self.splitter.addWidget(self.tear_sheet)
        self.addWidget(self.splitter, 0, 0, -1, -1)

        for i in range(8):
            self.setColumnStretch(i, 4)

        self.current_pos_line.sigPositionChanged.connect(
            self.update_accounting_table)
        self.current_pos_line.sigPositionChanged.connect(
            self.update_positions_table)
        # self.episode_table.cellDoubleClicked.connect(
        #     lambda: self.load_from_hdf(self.datapath/self.episode_table.currentItem().text())
        # )
        self.episode_table.currentRowChanged.connect(
            lambda: self.load_from_hdf(self.datapath / self.episode_table.
                                       currentItem().text()))

        self.link_x_axes()
        # self.unlink_x_axes()

    def log_adjust(self):
        for metric, plot in self.plots.items():
            if metric in self.data.keys():
                max_ele = np.nanmax(self.data[metric])
                min_ele = np.nanmin(self.data[metric])
                if max_ele - min_ele > 1e10:
                    plot.setLogMode(False, True)
                else:
                    plot.setLogMode(False, False)

    def link_x_axes(self):
        for name, plot in self.plots.items():
            if name != "equity":
                plot.setXLink(self.plots['equity'])

    def unlink_x_axes(self):
        for plot in self.plots.values():
            plot.setXLink(plot)

    def set_datapath(self, path):
        self.datapath = Path(path)
        self.load_episode_list()
        self.load_from_hdf()

    def clear_plots(self):
        """ Clear data for all plots """
        for _, line in self.lines.items():
            if isinstance(line, dict):
                for _, sub_line in line.items():
                    sub_line.clear()
            else:
                line.clear()

    def process_data(self, data):
        """ Converts data to ndarrays """
        self.data = dict(data.items())
        if isinstance(data['prices'], (pd.Series, pd.DataFrame)):
            self.data['prices'] = np.array(data['prices'].tolist())
            self.data['transaction'] =\
                np.array(data['transaction'].tolist()) # assuming this is also pd
            self.data['ledgerNormed'] = np.array(data['ledgerNormed'].tolist())
        else:
            self.data['prices'] = np.array(data['prices'])
            self.data['transaction'] = np.array(data['transaction'])
            self.data['ledgerNormed'] = np.array(data['ledgerNormed'])
            assert len(self.data['prices'].shape) == \
                len(self.data['transaction'].shape) ==\
                len(self.data['ledgerNormed'].shape) == 2,\
                "number of assets dims in prices, transaction and ledger" + \
                "must match"
        # self.data = {k: np.nan_to_num(v, 0.) for k, v in self.data.items()}

    def _set_data_with_variable_assets(self):
        """ For plots with multiple lines corresponding to assets """
        idxs = [idx.row() for idx in self.asset_picker.selectedIndexes()]
        for i, asset in enumerate(self.assets):
            if i not in self.lines['prices']:  # initialize plots
                self.lines['prices'][i] = self.plots['prices'].plot(
                    y=[], pen=(i, self.data['prices'].shape[1]))
                self.lines['transaction'][i] =\
                    self.plots['transaction'].plot(
                        y=[], pen=(i, self.data['transaction'].shape[1]))
            if i in idxs:
                self.lines['prices'][i].setData(y=self.data['prices'][:, i])
                self.lines['transaction'][i].setData(
                    y=self.data['transaction'][:, i])
            else:
                self.lines['prices'][i].clear()
                self.lines['transaction'][i].clear()

    def _set_assets(self, assets):
        """
        Call after self.process_data(data) (i.e in self._set_data) so that
        n_assets can be ectracted and compared from self.data['prices']
        """
        n_assets = self.data['prices'].shape[1]
        if assets is None:
            assets = [str(asset) for asset in range(n_assets)]
        else:
            if isinstance(assets, np.ndarray):
                assets = assets.tolist()
            if len(assets) != n_assets:
                raise ValueError(
                    "assets provided by dataset attribute is not"
                    "the same length as assets indicated by  data")
        if self.assets != assets:
            self.asset_picker.clear()
            self.assets = assets
            # self.asset_picker.addItems(self.assets)
            for i, asset in enumerate(self.assets):
                self.asset_picker.addItem(asset)
                self.asset_picker.itemAt(i, 0).setSelected(1)

    def _set_data(self, data, assets=None):
        if len(data) == 0:
            print('test episode data is empty')
            data = {k: [] for k in self.lines.keys()}

        self._set_assets(assets)
        self._set_data_with_variable_assets()

        self.lines['equity'].setData(y=data['equity'],
                                     pen=self.colours['equity'])
        self.lines['reward'].setData(y=data['reward'],
                                     pen=self.colours['reward'])
        self.lines['cash'].setData(y=data['cash'], pen=self.colours['cash'])
        self.lines['availableMargin'].setData(y=data['availableMargin'],
                                              pen=self.colours['cash'])
        self.lines['ledgerNormed'].setImage(self.data['ledgerNormed'],
                                            axes={
                                                'x': 0,
                                                'y': 1
                                            })
        self.current_pos_line.setValue(len(data['equity']) - 1)
        self.current_pos_line.setBounds((0, len(data['equity']) - 1))
        self.update_accounting_table()
        self.positions_table.setRowCount(self.data['ledgerNormed'].shape[1])
        self.update_positions_table()

    def set_data(self, data, assets=None):
        self.process_data(data)
        self.clear_plots()
        self._set_data(data, assets)
        # self.pick_assets()
        self.log_adjust()

    def load_episode_list(self, path=None):
        path = path or self.datapath
        if path is not None:
            episodes = filter(lambda x: "episode" in str(x.name),
                              path.iterdir())
            self.episodes = sorted(
                episodes,
                key=lambda x: int(str(x.name).split('_')[3]),
                reverse=True)
            # self.episode_table.setRowCount(len(self.episodes))
            self.episode_table.clear()
            self.episode_table.addItems([ep.name for ep in self.episodes])
            # for i, episode in enumerate(self.episodes):
            #     val = QTableWidgetItem(str(episode.name))
            #     self.episode_table.setItem(i, 0, val)

    def load_from_hdf(self, path=None):
        if path is None:
            if len(self.episodes) > 0:
                latest_episode = self.episodes[-1]
            else:
                return
        else:
            latest_episode = path
        data = pd.read_hdf(latest_episode, key='full_run')
        assets = None
        with h5py.File(latest_episode, 'r') as f:
            if 'asset_names' in f.attrs.keys():
                assets = f.attrs['asset_names']
        self.load_tearsheet(path)
        self.set_data(data, assets=assets)

    def load_tearsheet(self, path):
        if path is not None:
            ep_name = path.stem
            env_steps = int(ep_name.split('_')[3])
            data = pd.read_hdf(path.parent / 'test.hdf5', key='run_history')
            run_summary = data[data['env_steps'] == env_steps]
            if len(run_summary) == 0:
                run_summary = data[data['env_steps'] == env_steps - 1]
                if len(run_summary) == 0:
                    run_summary = data[data['env_steps'] == env_steps + 1]
            # print('len run summ: ', len(run_summary))
            if len(run_summary) > 0:
                run_summary = run_summary.to_dict()
                self.tear_sheet.setData(run_summary)

    def update_accounting_table(self):
        current_timepoint = int(self.current_pos_line.value())
        try:
            for i, metric in enumerate([
                    'equity', 'balance', 'cash', 'availableMargin',
                    'usedMargin', 'pnl'
            ]):
                metric = QTableWidgetItem(
                    str(self.data[metric][current_timepoint]))
                self.accounting_table.setItem(i, 0, metric)
        except IndexError:
            print("IndexError")

    def update_positions_table(self):
        current_timepoint = int(self.current_pos_line.value())
        try:
            n_assets = len(self.data['ledger'][0])
            for asset in range(n_assets):
                for i, metric in enumerate(['ledger', 'ledgerNormed']):
                    val = self.data[metric][current_timepoint][asset]
                    val = QTableWidgetItem(f"{val: 0.3f}")
                    self.positions_table.setItem(asset, i, val)
        except IndexError:
            import traceback
            traceback.print_exc()

    def export_plots(self, export_path):
        figs = make_figs(self.data, assets=self.assets)
        ep_name = Path(self.episode_table.currentItem().text()).stem
        if ep_name is None:
            raise IndexError("Episode from table must be selected for export")
        xrange_label = self.plots['prices'].viewRange()[0][0]
        for label, fig in figs.items():
            ax = fig.axes[0]
            if label in self.plots.keys():
                x_range, y_range = self.plots[label].viewRange()
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
            save_fig(
                fig,
                export_path / f'{ep_name}_{int(xrange_label)}/{label}.pdf')
        plt.close('all')


class TestEpisodePlotsDQN(TestEpisodePlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({'qvals': (255, 86, 0)})
        self.lines['qvals'] = pg.ImageItem(np.empty(shape=(1, 1, 1)))
        self.plots['qvals'] = self.graphs.addPlot(title="qvals", row=1, col=6)
        self.plots['qvals'].addItem(self.lines['qvals'])
        self.plots['qvals'].showGrid(1, 1)
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.lines['qvals'])
        self.graphs.addItem(hist, row=1, col=8, colspan=1)
        # self.lines['qvals'].autoLevels()

        self.link_x_axes()

    def process_data(self, data):
        super().process_data(data)
        if isinstance(data['qvals'], (pd.Series, pd.DataFrame)):
            self.data['qvals'] = np.array(data['qvals'].tolist())
        else:
            self.data['qvals'] = np.array(data['qvals'])

    def _set_data(self, data, assets=None):
        super()._set_data(data, assets)
        if len(data) == 0:
            return
        qvals = self.data['qvals']
        if len(qvals.shape) == 4:
            qvals = qvals.squeeze(1)
            self.data['qvals'] = qvals
        assert len(qvals.shape) == 3
        qvals = qvals.reshape(qvals.shape[0], -1)
        self.lines['qvals'].setImage(qvals, axes={'x': 0, 'y': 1})
        cmap = pg.ColorMap(pos=np.linspace(np.nanmin(qvals), np.nanmax(qvals),
                                           2),
                           color=self.heatmap_colors)
        self.lines['qvals'].setLookupTable(cmap.getLookupTable())


class TestEpisodeDDPG(TestEpisodePlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({'qvals': (255, 86, 0)})
        # self.plots['qvals'].addItem(self.lines['qvals'])
        # self.plots['qvals'] = self.graphs.addViewBox(row=1, col=4)
        self.plots['qvals'] = self.graphs.addPlot(row=1, col=6, colspan=2)
        # self.lines['qvals'] = pg.ImageItem(parent=self.graphs,
        #                                    view=self.plots['qvals'].getViewBox())
        # self.plots['qvals'].addWidget(self.lines['qvals'])
        # self.lines['qvals'] = pg.ImageView(self.graphs)
        self.lines['qvals'] = pg.ImageItem(np.empty(shape=(1, 1)),
                                           axes={
                                               'x': 0,
                                               'y': 1
                                           })
        self.plots['qvals'].addItem(self.lines['qvals'])
        self.plots['qvals'].setTitle('qvals')
        self.plots['qvals'].showAxis('left', False)
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.lines['qvals'])
        self.graphs.addItem(hist, row=1, col=8, colspan=1)
        # self.lines['qvals'].showGrid(1, 1)
        self.lines['qvals'].setImage(np.empty(shape=(1, 1)),
                                     axes={
                                         'x': 0,
                                         'y': 1
                                     })
        # self.plots['qvals'].ledgerNormedshow()
        # self.plots['qvals'].hoverEvent = self.update_qvals_title
        # self.plots['qvals'].addItem(self.lines['qvals'])
        self.plots['action'] = self.graphs.addPlot(
            title="model output - portfolio weights", row=2, col=0, colspan=2)
        self.lines['action'] = pg.ImageItem(np.empty(shape=(1, 1, 1)))
        self.plots['action'].addItem(self.lines['action'], colspan=2)
        self.plots['action'].showGrid(1, 1)
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.lines['action'])
        self.graphs.addItem(hist, row=2, col=2, colspan=1)
        self.transaction_table = pg.TableWidget()
        self.transaction_table.setVerticalHeaderLabels(['Model Outputs'])
        # self.graphs.addItem(self.actions_table, row=2, col=3, colspan=1)
        self.tables.addWidget(self.transaction_table)
        # self.addWidget(self.transaction_table, 3, 9, 1, 1)
        self.link_x_axes()
        self.current_pos_line.sigPositionChanged.connect(
            self.update_transaction_table)

    def update_transaction_table(self):
        current_timepoint = int(self.current_pos_line.value())
        try:
            actions = self.data['action'][current_timepoint]
            transaction = self.data['transaction'][current_timepoint]
            transaction = np.concatenate([[0], transaction], axis=0)
            data = np.stack([actions, transaction], axis=1)
            # data = np.array([actions, transaction],
            #                 dtype=[('model_output', float),
            #                        ('transaction', float)])
            self.transaction_table.setData(data)
            self.transaction_table.setHorizontalHeaderLabels(
                ['Model Outputs', 'Transaction'])
            self.transaction_table.resizeColumnsToContents()
        except IndexError:
            import traceback
            traceback.print_exc()

    # def update_qvals_title(self, event):
    #     if event.isExit():
    #         # self.plots['ledgerNormed'].setTitle("")
    #         return
    #     pos = event.pos()
    #     j, i = pos.y(), pos.x()
    #     i = int(np.clip(i, 0, self.data['qvals'].shape[0]-1))
    #     j = int(np.clip(j, 0, self.data['qvals'].shape[1]-1))
    #     val = self.data['qvals'][i, j]
    #     ppos = self.lines['qvals'].mapToParent(pos)
    #     x, y = ppos.x(), ppos.y()
    #     self.plots['qvals'].setTitle(f"qval ({i}, {j}) ({x}, {y}): {val}")

    def process_data(self, data):
        super().process_data(data)
        if isinstance(data['qvals'], (pd.DataFrame, pd.Series)):
            self.data['qvals'] = np.array(data['qvals'].tolist())
        else:
            self.data['qvals'] = np.array(data['qvals'])
        assert len(self.data['qvals'].shape) == 2
        if isinstance(data['action'], (pd.DataFrame, pd.Series)):
            self.data['action'] = np.array(data['action'].tolist())[:, 0, :]
        else:
            self.data['action'] = np.array(data['action'])
        assert len(self.data['action'].shape) == 2

    def _set_data(self, data, assets=None):
        super()._set_data(data, assets)
        if len(data) == 0:
            return
        self.lines['qvals'].setImage(self.data['qvals'], axes={'x': 0, 'y': 1})
        # cmap = pg.ColorMap(pos=np.linspace(np.nanmin(qvals), np.nanmax(qvals), 2),
        #                    color=self.heatmap_colors)
        # self.lines['qvals'].setLookupTable(cmap.getLookupTable())
        self.lines['action'].setImage(self.data['action'],
                                      axes={
                                          'x': 0,
                                          'y': 1
                                      })
        # cmap = pg.ColorMap(pos=np.linspace(np.nanmin(self.data['action']),
        #                                    np.nanmax(self.data['action']), 2),
        #                    color=self.heatmap_colors)
        # self.lines['action'].setLookupTable(cmap.getLookupTable())


class TestEpisodePlotsSACD(TestEpisodePlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({'qvals1': (255, 86, 0), 'qvals2': (86, 255, 0)})
        self.plots['qvals'] = self.graphs.addPlot(row=1, col=6, colspan=2)
        self.lines['qvals1'] = pg.ImageItem(np.empty(shape=(1, 1)),
                                            axes={
                                                'x': 0,
                                                'y': 1
                                            })
        self.lines['qvals2'] = pg.ImageItem(np.empty(shape=(1, 1)),
                                            axes={
                                                'x': 0,
                                                'y': 1
                                            })
        self.plots['qvals'].addItem(self.lines['qvals1'])
        self.plots['qvals'].addItem(self.lines['qvals2'])
        self.plots['qvals'].setTitle('qvals')
        self.plots['qvals'].showAxis('left', False)
        hist1 = pg.HistogramLUTItem()
        hist1.setImageItem(self.lines['qvals1'])
        self.graphs.addItem(hist1, row=1, col=8, colspan=1)
        hist2 = pg.HistogramLUTItem()
        hist2.setImageItem(self.lines['qvals2'])
        self.graphs.addItem(hist2, row=1, col=9, colspan=1)
        # self.lines['qvals'].showGrid(1, 1)
        self.lines['qvals1'].setImage(np.empty(shape=(1, 1)),
                                      axes={
                                          'x': 0,
                                          'y': 1
                                      })
        self.lines['qvals2'].setImage(np.empty(shape=(1, 1)),
                                      axes={
                                          'x': 0,
                                          'y': 1
                                      })
        self.plots['action'] = self.graphs.addPlot(
            title="actor greedy actions", row=2, col=0, colspan=2)
        self.lines['action'] = self.plots['action'].plot(y=[])
        self.plots['action'].showGrid(1, 1)
        self.plots['action_probs'] = self.graphs.addPlot(
            title="actor prob distribution", row=2, col=2, colspan=2)
        self.lines['action_probs'] = pg.ImageItem(np.empty(shape=(1, 1, 1)))
        self.plots['action_probs'].addItem(self.lines['action_probs'],
                                           colspan=2)
        self.plots['action_probs'].showGrid(1, 1)
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.lines['action_probs'])
        self.graphs.addItem(hist, row=2, col=4, colspan=1)
        self.transaction_table = pg.TableWidget()
        self.transaction_table.setVerticalHeaderLabels(['Model Outputs'])
        # self.graphs.addItem(self.actions_table, row=2, col=3, colspan=1)
        self.addWidget(self.transaction_table, 3, 9, 1, 1)
        self.link_x_axes()
        self.current_pos_line.sigPositionChanged.connect(
            self.update_transaction_table)

    def update_transaction_table(self):
        current_timepoint = int(self.current_pos_line.value())
        try:
            actions = self.data['action'][current_timepoint]
            transaction = self.data['transaction'][current_timepoint]
            transaction = np.concatenate([[0], transaction], axis=0)
            data = np.stack([actions, transaction], axis=1)
            # data = np.array([actions, transaction],
            #                 dtype=[('model_output', float),
            #                        ('transaction', float)])
            self.transaction_table.setData(data)
            self.transaction_table.setHorizontalHeaderLabels(
                ['Model Outputs', 'Transaction'])
            self.transaction_table.resizeColumnsToContents()
        except IndexError:
            import traceback
            traceback.print_exc()

    def process_data(self, data):
        super().process_data(data)
        for qval_key in ('qvals1', 'qvals2'):
            if isinstance(data[qval_key], (pd.DataFrame, pd.Series)):
                self.data[qval_key] = np.array(data[qval_key].tolist())
            else:
                self.data[qval_key] = np.array(data[qval_key])
        assert len(self.data['qvals1'].shape) == 2
        if isinstance(data['action_probs'], (pd.DataFrame, pd.Series)):
            self.data['action_probs'] = np.array(data['action_probs'].tolist())
        else:
            self.data['action_probs'] = np.array(data['action_probs'])
        assert len(self.data['action_probs'].shape) == 3
        # self.data['action'] = self.data['action'].cpu().numpy()

    def _set_data(self, data, assets=None):
        super()._set_data(data, assets)
        if len(data) == 0:
            return
        self.lines['qvals1'].setImage(self.data['qvals1'],
                                      axes={
                                          'x': 0,
                                          'y': 1
                                      })
        self.lines['qvals2'].setImage(self.data['qvals2'],
                                      axes={
                                          'x': 0,
                                          'y': 1
                                      })
        # cmap = pg.ColorMap(pos=np.linspace(np.nanmin(qvals), np.nanmax(qvals), 2),
        #                    color=self.heatmap_colors)
        # self.lines['qvals'].setLookupTable(cmap.getLookupTable())
        self.lines['action'].setData(self.data['action'])
        self.lines['action_probs'].setImage(self.data['action_probs'],
                                            axes={
                                                'x': 0,
                                                'y': 1
                                            })
        # cmap = pg.ColorMap(pos=np.linspace(np.nanmin(self.data['action']),
        #                                    np.nanmax(self.data['action']), 2),
        #                    color=self.heatmap_colors)
        # self.lines['action'].setLookupTable(cmap.getLookupTable())


###############################################################################
############# Test History Plots  #############################################
###############################################################################


class TestHistoryPlots(QGridLayout):
    def __init__(self, title=None):
        super().__init__()

        self.timeframes = None
        self.assets = None
        self.data = None
        self.colours = {
            'mean_equity': (0, 255, 0),
            'final_equity': (255, 0, 0),
            'mean_reward': (242, 242, 242),
            'cash': (0, 255, 255),
            'margin': (255, 86, 0),
            'max_drawdown': (255, 86, 0)
        }

        self.graphs = pg.GraphicsLayoutWidget(show=True, title=title)
        # self.graphs_widget = pg.GraphicsLayoutWidget()
        # self.graphs = self.graphs_widget.addLayout()
        self.plots = {}

        self.plots['equity'] = self.graphs.addPlot(
            title='Equity',
            bottom='training steps',
            left='Denomination currency',
            row=0,
            col=0,
            colspan=1)
        self.plots['reward'] = self.graphs.addPlot(title='Rewards',
                                                   bottom='training steps',
                                                   left='rewards',
                                                   row=0,
                                                   col=1,
                                                   colspan=1)
        self.plots['returns'] = self.graphs.addPlot(title='Log Returns',
                                                    bottom='training steps',
                                                    left='log returns',
                                                    row=0,
                                                    col=2,
                                                    colspan=1)
        self.plots['sharpe'] = self.graphs.addPlot(title='Sharpe Ratios',
                                                   bottom='training steps',
                                                   left='sharpe ratio',
                                                   row=0,
                                                   col=3,
                                                   colspan=1)
        self.plots['sortino'] = self.graphs.addPlot(title='Sortino Ratios',
                                                    bottom='training steps',
                                                    left='sortino ratio',
                                                    row=1,
                                                    col=0,
                                                    colspan=1)
        self.plots['max_drawdown'] = self.graphs.addPlot(
            title='Max Drawdown',
            bottom='training steps',
            left='peak / valley ratio',
            row=1,
            col=1,
            colspan=1)
        self.plots['time_spent'] = self.graphs.addPlot(
            title='Time Spent in Positions',
            bottom='training steps',
            left='peak / valley ratio',
            row=1,
            col=2,
            colspan=1)

        self.plots['equity'].showGrid(1, 1)
        self.plots['returns'].showGrid(1, 1)
        self.plots['reward'].showGrid(1, 1)
        self.plots['time_spent'].showGrid(1, 1)
        self.plots['max_drawdown'].showGrid(1, 1)
        self.plots['sharpe'].showGrid(1, 1)
        self.plots['sortino'].showGrid(1, 1)

        self.plots['equity'].addLegend()
        self.plots['returns'].addLegend()
        self.plots['sharpe'].addLegend()
        self.plots['sortino'].addLegend()
        self.plots['max_drawdown'].addLegend()
        self.plots['time_spent'].addLegend()

        self.plots['equity'].setLabels()
        self.plots['returns'].setLabels()
        self.plots['sharpe'].setLabels()
        self.plots['sortino'].setLabels()

        self.lines = {}
        self.lines['mean_equity'] = self.plots['equity'].plot(
            y=[], name='mean_equity')
        self.lines['final_equity'] = self.plots['equity'].plot(
            y=[], name='final_equity')
        self.lines['mean_reward'] = self.plots['reward'].plot(y=[])
        # Following lines have timeframes as keys
        self.lines['returns'] = {}
        self.lines['sharpe'] = {}
        self.lines['sortino'] = {}
        self.lines['time_spent'] = {}
        self.lines['max_drawdown'] = self.plots['max_drawdown'].plot(
            y=[], name='drawdown')

        self.assets_table_label = QLabel("Assets")
        self.assets_table = QListWidget()
        self.assets_table.setSelectionMode(QListWidget.MultiSelection)

        self.timeframes_table_label = QLabel("Aggregation TimeFrames")
        self.timeframes_table = QListWidget()
        self.timeframes_table.setSelectionMode(QListWidget.MultiSelection)

        self.assets_table.itemSelectionChanged.connect(
            self._set_data_for_variable_assets)
        self.timeframes_table.itemSelectionChanged.connect(
            self._set_data_for_variable_timeframes)

        self.tables = QtGui.QVBoxLayout()
        self.tables.addWidget(self.assets_table_label)
        self.tables.addWidget(self.assets_table)
        self.tables.addWidget(self.timeframes_table_label)
        self.tables.addWidget(self.timeframes_table)
        self.tables_widget = QtGui.QWidget()
        self.tables_widget.setLayout(self.tables)

        self.addWidget(self.graphs, 0, 0, -1, 8)
        self.addWidget(self.tables_widget, 0, 8, -1, 1)

        for i in range(8):
            self.setColumnStretch(i, 4)
        self.setColumnStretch(8, 1)

        self.link_x_axes()

    def link_x_axes(self):
        for name, plot in self.plots.items():
            plot.setXLink(self.plots['equity'])

    def clear_plots(self):
        for _, line in self.lines.items():
            if isinstance(line, dict):
                for _, _line in line.items():
                    _line.clear()
            else:
                line.clear()

    def update_timeframes_table(self, data):
        sharpes = [col for col in data.keys() if 'sharpe' in col]
        self.timeframes = [col[21:] for col in sharpes]
        self.timeframes_table.clear()
        for i, tf in enumerate(self.timeframes):
            self.timeframes_table.addItem(tf)
            self.timeframes_table.itemAt(i, 0).setSelected(1)

    def update_assets_table(self, data):
        times_spent = [col for col in data.keys() if 'time_spent' in col]
        self.assets = [col[18:] for col in times_spent]
        self.assets_table.clear()
        for i, asset in enumerate(self.assets):
            self.assets_table.addItem(asset)
            self.assets_table.itemAt(i, 0).setSelected(1)

    def _set_data_for_variable_assets(self):
        idxs = [idx.row() for idx in self.assets_table.selectedIndexes()]
        x = self.data['training_steps']
        if self.assets is None:
            return
        for i, asset in enumerate(self.assets):
            if asset not in self.lines['time_spent']:
                self.lines['time_spent'][asset] = self.plots[
                    'time_spent'].plot(y=[],
                                       pen=(i, len(self.assets)),
                                       name=asset)
            if i in idxs:
                self.lines['time_spent'][asset].setData(
                    x=x, y=self.data[f'time_spent_in_pos_{asset}'])
            else:
                self.lines['time_spent'][asset].clear()

    def _set_data_for_variable_timeframes(self):
        idxs = [idx.row() for idx in self.timeframes_table.selectedIndexes()]
        x = self.data['training_steps']
        if self.timeframes is None:
            return
        for i, tf in enumerate(self.timeframes):
            if tf not in self.lines['returns']:  # assume not in others either
                self.lines['returns'][tf] = self.plots['returns'].plot(
                    y=[], pen=(i, len(self.timeframes)), name=tf)
                self.lines['sharpe'][tf] = self.plots['sharpe'].plot(
                    y=[], pen=(i, len(self.timeframes)), name=tf)
                self.lines['sortino'][tf] = self.plots['sortino'].plot(
                    y=[], pen=(i, len(self.timeframes)), name=tf)
            if i in idxs:
                self.lines['returns'][tf].setData(
                    x=x, y=self.data[f'equity_returns_offset_{tf}'])
                self.lines['sharpe'][tf].setData(
                    x=x, y=self.data[f'equity_sharpe_offset_{tf}'])
                self.lines['sortino'][tf].setData(
                    x=x, y=self.data[f'equity_sortino_offset_{tf}'])
            else:
                self.lines['returns'][tf].clear()
                self.lines['sharpe'][tf].clear()
                self.lines['sortino'][tf].clear()

    def _set_data(self):
        if self.data is None or len(self.data) == 0:
            print("test self.data is empty")
            self.data = {k: [] for k in self.lines.keys()}
        x = self.data['training_steps']
        for label in ('mean_equity', 'final_equity', 'mean_reward',
                      'max_drawdown'):
            try:
                self.lines[label].setData(x=x,
                                          y=self.data[label],
                                          pen=self.colours[label])
            except Exception as E:
                traceback.print_exc()
                print('Skipping', label)
                continue

    def process_data(self, data):
        self.data = data
        self.update_assets_table(data)
        self.update_timeframes_table(data)

    def set_data(self, data):
        self.process_data(data)
        self.clear_plots()
        self._set_data()
        self._set_data_for_variable_timeframes()
        self._set_data_for_variable_assets()

    def set_datapath(self, path):
        self.datapath = path
        self.load_from_hdf(path)

    def load_from_hdf(self, path=None):
        path = path or self.datapath
        if path is not None:
            data = pd.read_hdf(path / 'test.hdf5', key='run_history')
            self.set_data(data)

    def export_plots(self, export_path):
        figs = make_figs(self.data, assets=self.assets, x_key='training_steps')
        x_range, y_range = self.plots['equity'].viewRange()
        for label, fig in figs.items():
            ax = fig.axes[0]
            ax.set_xlim(x_range)
            # ax.set_ylim(y_range)
            save_fig(fig, export_path / f'{label}_{int(x_range[0])}.pdf')
        plt.close('all')


class TestHistoryPlotsDQN(TestHistoryPlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({'mean_qvals': (255, 86, 0)})
        self.plots['qvals'] = self.graphs.addPlot(
            title='Mean Qvals over Episodes',
            bottom='training_steps',
            left='Value',
            row=1,
            col=3,
            colspan=1)
        # self.plots['qvals'] = pg.PlotWidget(
        #     title='Mean Qvals over Episodes',
        #     bottom='training_steps',
        #     left='Value')
        # self.addWidget(self.plots['qvals'], 1, 3, 1, 1)
        self.lines['mean_qvals'] = self.plots['qvals'].plot(y=[])
        self.plots['qvals'].showGrid(1, 1)
        self.plots['qvals'].setLabels()

    def _set_data(self):
        super()._set_data()
        if self.data is None or len(self.data) == 0:
            self.data = {k: [] for k in self.lines.keys()}
        x = self.data['training_steps']
        self.lines['mean_qvals'].setData(x=x,
                                         y=self.data['mean_qvals'],
                                         pen=self.colours['mean_qvals'])


TestHistoryDDPG = TestHistoryPlotsDQN


class TestHistoryPlotsSACD(TestHistoryPlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({
            'mean_qvals1': (255, 86, 0),
            'mean_qvals2': (86, 255, 0)
        })
        self.plots['qvals'] = self.graphs.addPlot(title='Mean Qvals',
                                                  bottom='training_steps',
                                                  left='Value')
        self.lines['mean_qvals1'] = self.plots['qvals'].plot(y=[])
        self.lines['mean_qvals2'] = self.plots['qvals'].plot(y=[])
        self.plots['qvals'].showGrid(1, 1)
        self.plots['qvals'].setLabels()

    def set_data(self, data):
        super().set_data(data)
        if data is None or len(data) == 0:
            data = {k: [] for k in self.lines.keys()}
        self.lines['mean_qvals1'].setData(y=data['mean_qvals1'],
                                          pen=self.colours['mean_qvals1'])
        self.lines['mean_qvals2'].setData(y=data['mean_qvals2'],
                                          pen=self.colours['mean_qvals2'])

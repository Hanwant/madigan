from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from PyQt5.QtGui import QTableWidget, QTableWidgetItem, QGridLayout
import pyqtgraph as pg

####################################################################################
############# Factory Methods ######################################################
####################################################################################
def make_train_plots(agent_type, title=None, **kw):
    if agent_type in ("DQN", ):
        return TrainPlotsDQN(title=title, **kw)
    raise NotImplementedError(f"Train plot widget for agent_type: {agent_type}"+\
                                "has not been implemented")

def make_test_episode_plots(agent_type, title=None, **kw):
    if agent_type in ("DQN", ):
        return TestEpisodePlotsDQN(title=title, **kw)
    raise NotImplementedError(f"Test episode plot widget for agent_type: {agent_type}"+\
                                "has not been implemented")
def make_test_history_plots(agent_type, title=None, **kw):
    if agent_type in ("DQN", ):
        return TestHistoryPlotsDQN(title=title, **kw)
    raise NotImplementedError(f"Test History plot widget for agent_type: {agent_type}"+\
                                "has not been implemented")

####################################################################################
############# Training Plots  ######################################################
####################################################################################

class TrainPlots(QGridLayout):
    """ Base class for train plots """
    def __init__(self, title=None):
        super().__init__()
        self.datapath = None
        self.data = None
        self.graphs = pg.GraphicsLayoutWidget(show=True, title=title)
        self.addWidget(self.graphs)
        self.colours = {'reward': (255, 228, 181), 'reward_mean': (255, 0, 0),
                        'loss': (242, 242, 242)}
        self.plots = {}
        self.lines = {}
        self.plots['loss'] = self.graphs.addPlot(title='Loss',
                                                 bottom='step', left='Loss')
        self.plots['loss'].showGrid(1, 1)
        self.plots['running_reward'] = self.graphs.addPlot(title='Episode Rewards',
                                                           bottom='step', left='reward')
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

    def clear_data(self):
        for line in self.lines.values():
            line.setData(y=[])

    def set_data(self, data):
        self.data = data
        if len(data) == 0:
            print("train data is empty")
            data = {k: [] for k in self.lines.keys()}
        self.lines['loss'].setData(y=data['loss'], pen=self.colours['loss'])
        rewards = data['running_reward']
        rewards_mean = pd.Series(data['running_reward']).rolling(
            self.reward_mean_window).mean().to_numpy()
        rewards_mean = np.nan_to_num(rewards_mean)
        self.lines['running_reward'].setData(y=rewards,
                                             pen=self.colours['reward'])
        self.lines['running_reward_mean'].setData(
            y=rewards_mean, pen=pg.mkPen(
                {'color': self.colours['reward_mean'], 'width': 2}))

    def set_datapath(self, path):
        self.datapath = Path(path)
        self.load_from_hdf()

    def load_from_hdf(self, path=None):
        path = path or self.datapath/'train.hdf5'
        if path is not None:
            data = pd.read_hdf(path, key='train')
            self.set_data(data)

class TrainPlotsDQN(TrainPlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({'Gt': (0, 255, 255), 'Qt': (255, 86, 0)})
        self.plots['values'] = self.graphs.addPlot(title='Values',
                                                    bottom='step', left='Value')
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
        self.lines['Gt'].setData(y=data['Gt'], pen=self.colours['Gt'])
        self.lines['Qt'].setData(y=data['Qt'], pen=self.colours['Qt'])

####################################################################################
############# Test Episode Plots  ##################################################
####################################################################################

class TestEpisodePlots(QGridLayout):
    def __init__(self, title=None, **kw):
        super().__init__()
        self.graphs = pg.GraphicsLayoutWidget(show=True,title=title)
        self.datapath = None
        self.episodes = []
        self.data = None
        self.colours = {'equity': (218, 112, 214), 'reward': (242, 242, 242),
                        'cash': (0, 255, 255), 'margin': (255, 86, 0)}
        self.heatmap_colors = [(0, 0, 85), (85, 0, 0)]
        cmap = pg.ColorMap(pos=np.linspace(-1., 1., 2),
                           color=self.heatmap_colors)
        self.plots = {}
        self.lines = {}
        self.plots['prices'] = self.graphs.addPlot(title='Prices', bottom='time',
                                                   left='prices')
        self.plots['equity'] = self.graphs.addPlot(title='Equity', bottom='time',
                                                   left='Denomination currency')
        self.plots['reward'] = self.graphs.addPlot(title='Rewards', bottom='time',
                                                   left='reward')
        self.lines['ledgerNormed'] = pg.ImageItem(np.empty(shape=(1, 1, 1)))
        self.lines['ledgerNormed'].setLookupTable(cmap.getLookupTable())
        self.plots['ledgerNormed'] = self.graphs.addPlot(title="Ledger")
        self.plots['ledgerNormed'].addItem(self.lines['ledgerNormed'])
        self.plots['equity'].setLabels()
        self.lines['equity'] = self.plots['equity'].plot(y=[])
        self.lines['reward'] = self.plots['reward'].plot(y=[])
        self.lines['prices'] = {}
        self.current_pos_line = pg.InfiniteLine(movable=True)
        self.plots['equity'].addItem(self.current_pos_line)
        self.plots['cash'] = self.graphs.addPlot(title="cash", row=1, col=0)
        self.plots['availableMargin'] = self.graphs.addPlot(title="available margin",
                                                             row=1, col=1)
        self.plots['transactions'] = self.graphs.addPlot(title="transactions", row=1, col=2)
        self.lines['cash'] = self.plots['cash'].plot(y=[])
        self.lines['availableMargin'] = self.plots['availableMargin'].plot(y=[])
        self.lines['transactions'] = {}

        self.plots['prices'].showGrid(1, 1)
        self.plots['equity'].showGrid(1, 1)
        self.plots['reward'].showGrid(1, 1)
        self.plots['prices'].showGrid(1, 1)
        self.plots['cash'].showGrid(1, 1)
        self.plots['availableMargin'].showGrid(1, 1)
        self.plots['transactions'].showGrid(1, 1)
        self.plots['ledgerNormed'].showGrid(1, 1)

        self.episode_table = QTableWidget(0, 1)
        self.episode_table.setHorizontalHeaderLabels(['Episode Name'])
        self.episode_table.setStyleSheet("background-color:rgb(99, 102, 49) ")
        self.accounting_table = QTableWidget(5, 1)
        self.accounting_table.setVerticalHeaderLabels(['Equity', 'Balance', 'Cash',
                                                      'Available Margin', 'Used Margin',
                                                      'Net PnL'])
        self.accounting_table.setStyleSheet("background-color:rgb(44, 107, 42) ")
        self.positions_table = QTableWidget(0, 2)
        self.positions_table.setHorizontalHeaderLabels(['Ledger', 'Ledger Normed'])
        self.positions_table.setStyleSheet("background-color:rgb(23, 46, 67) ")

        self.episode_table.horizontalHeader().setStretchLastSection(True)
        self.accounting_table.horizontalHeader().setStretchLastSection(True)
        self.positions_table.horizontalHeader().setStretchLastSection(True)

        self.addWidget(self.graphs, 0, 0, -1, 8)
        self.addWidget(self.episode_table, 0, 9, 1, 1)
        self.addWidget(self.accounting_table, 1, 9, 1, 1)
        self.addWidget(self.positions_table, 2, 9, 1, 1)
        self.setColumnStretch(0, 4)

        self.current_pos_line.sigPositionChanged.connect(self.update_accounting_table)
        self.current_pos_line.sigPositionChanged.connect(self.update_positions_table)
        self.episode_table.cellDoubleClicked.connect(
            lambda: self.load_from_hdf(self.datapath/self.episode_table.currentItem().text())
        )

        self.link_x_axes()
        # self.unlink_x_axes()

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

    def load_episode_list(self, path=None):
        path = path or self.datapath
        if path is not None:
            episodes = filter(lambda x: "episode" in str(x.name), path.iterdir())
            self.episodes = sorted(episodes,
                                   key=lambda x: int(str(x.name).split('_')[3]),
                                   reverse=True)
            self.episode_table.setRowCount(len(self.episodes))
            for i, episode in enumerate(self.episodes):
                val = QTableWidgetItem(str(episode.name))
                self.episode_table.setItem(i, 0, val)

    def load_from_hdf(self, path=None):
        if path is None:
            if len(self.episodes) > 0:
                latest_episode = self.episodes[-1]
            else:
                return
        else:
            latest_episode = path
        data = pd.read_hdf(latest_episode, key='full_run')
        self.set_data(data)

    def clear_data(self):
        for _, line in self.lines.items():
            if isinstance(line, dict):
                for _, sub_line in line.items():
                    sub_line.setData(y=[])
            else:
                try:
                    line.setData(y=[])
                except:
                    line.setImage(np.empty(shape=(1, 1, 1)))

    def set_data(self, data):
        self.data = data
        self.clear_data()
        if len(data) == 0:
            print('test episode data is empty')
            data = {k: [] for k in self.lines.keys()}
            # data['assets'] = 0
        self.lines['equity'].setData(y=data['equity'], pen=self.colours['equity'])
        self.lines['reward'].setData(y=data['reward'], pen=self.colours['reward'])
        self.lines['cash'].setData(y=data['cash'], pen=self.colours['cash'])
        self.lines['availableMargin'].setData(y=data['availableMargin'],
                                              pen=self.colours['cash'])
        if isinstance(data['prices'], (pd.Series, pd.DataFrame)):
            prices = np.array(data['prices'].tolist())
            transactions = np.array(data['transaction'].tolist()) # assuming this is also pd
            ledger = np.array(data['ledgerNormed'].tolist())
        else:
            prices = np.array(data['prices'])
            transactions = np.array(data['transaction'])
            ledger = np.array(data['ledgerNormed'])
            assert len(prices.shape) ==  len(transactions.shape) == len(ledger.shape) == 2
        for asset in range(prices.shape[1]):
            self.lines['prices'][asset] = self.plots['prices'].plot(y=prices[:, asset],
                                                                    pen=(asset, prices.shape[1]))
            self.lines['transactions'][asset] = self.plots['transactions'].plot(y=transactions[:, asset],
                                                                                pen=(asset, transactions.shape[1]))
            self.lines['ledgerNormed'].setImage(ledger, axes={'x': 0, 'y': 1})
            self.current_pos_line.setValue(len(data['equity'])-1)
            self.current_pos_line.setBounds((0, len(data['equity'])-1))
            self.update_accounting_table()
            self.positions_table.setRowCount(ledger.shape[1])
            self.update_positions_table()

    def update_accounting_table(self):
        current_timepoint = int(self.current_pos_line.value())
        try:
            for i, metric in enumerate(['equity', 'balance', 'cash', 'availableMargin',
                                        'usedMargin', 'pnl']):
                metric = QTableWidgetItem(str(self.data[metric][current_timepoint]))
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


class TestEpisodePlotsDQN(TestEpisodePlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({'qvals': (255, 86, 0)})
        self.lines['qvals'] = pg.ImageItem(np.empty(shape=(1, 1, 1)))
        self.plots['qvals'] = self.graphs.addPlot(title="qvals", row=1, col=3)
        self.plots['qvals'].addItem(self.lines['qvals'])
        self.plots['qvals'].showGrid(1, 1)

        self.link_x_axes()

    def set_data(self, data):
        super().set_data(data)
        if len(data) == 0:
            return
        if isinstance(data['qvals'], (pd.DataFrame, pd.Series)):
            qvals = np.array(data['qvals'].tolist())
        else:
            qvals = np.array(data['qvals'])
        assert len(qvals.shape) == 3
        qvals = qvals.reshape(qvals.shape[0], -1)
        self.lines['qvals'].setImage(qvals, axes={'x': 0, 'y': 1})
        cmap = pg.ColorMap(pos=np.linspace(np.nanmin(qvals), np.nanmax(qvals), 2),
                           color=self.heatmap_colors)
        self.lines['qvals'].setLookupTable(cmap.getLookupTable())

####################################################################################
############# Test History Plots  ##################################################
####################################################################################

class TestHistoryPlots(QGridLayout):
    def __init__(self, title=None):
        super().__init__()
        self.graphs = pg.GraphicsLayoutWidget(show=True, title=title)
        self.addWidget(self.graphs)
        self.colours = {'equity': (218, 112, 214), 'reward': (242, 242, 242),
                        'cash': (0, 255, 255), 'margin': (255, 86, 0)}
        self.data = None
        self.plots = {}
        self.plots['equity'] = self.graphs.addPlot(title='Equity Over Episodes',
                                                   bottom='training_steps',
                                                   left='Denomination currency')
        self.plots['reward'] = self.graphs.addPlot(title='Mean Returns over Episodes',
                                                   bottom='training_steps',
                                                   left='returns (proportion)')
        self.plots['margin'] = self.graphs.addPlot(title='Mean Cash over Episodes',
                                                   bottom='training steps',
                                                   left='returns (proportion)')
        self.plots['equity'].showGrid(1, 1)
        self.plots['reward'].showGrid(1, 1)
        self.plots['margin'].showGrid(1, 1)
        self.plots['margin'].setLabels()
        self.lines = {}
        self.lines['mean_equity'] = self.plots['equity'].plot(y=[])
        self.lines['final_equity'] = self.plots['equity'].plot(y=[])
        self.lines['mean_reward'] = self.plots['reward'].plot(y=[])
        # self.lines['mean_available_margin']= self.plots['margin'].plot(y=[])

    def clear_data(self):
        for _, line in self.lines.items():
            line.setData(y=[])

    def set_data(self, data):
        if data is None or len(data) == 0:
            print("test data is empty")
            data = {k: [] for k in self.lines.keys()}
        self.lines['mean_equity'].setData(y=data['mean_equity'],
                                            pen=self.colours['mean_equity'])
        self.lines['final_equity'].setData(y=data['final_equity'],
                                            pen=self.colours['final_equity'])
        self.lines['mean_reward'].setData(y=data['mean_reward'],
                                            pen=self.colours['mean_reward'])
        # self.lines['mean_transaction_cost'].setData(y=data['mean_transaction_cost'],
        #                                             pen=self.colours['mean_transaction_cost'])
        # self.lines['margin'].setData(y=data['cash'], pen=self.colours['margin'])

    def set_datapath(self, path):
        self.datapath = path
        self.load_from_hdf(path)

    def load_from_hdf(self, path=None):
        path = path or self.datapath
        if path is not None:
            data = []
            for run in self.datapath.iterdir():
                data.append(pd.read_hdf(run, key='run_summary'))
            data = pd.concatenate(data, axis=0)
            self.set_data(data)


class TestHistoryPlotsDQN(TestHistoryPlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({'mean_qvals': (255, 86, 0)})

    def set_data(self, data):
        super().set_data(data)
        if data is None or len(data) == 0:
            data = {k: [] for k in self.lines.keys()}
        self.lines['mean_qvals'].setData(y=data['mean_qvals'],
                                            pen=self.colours['mean_qvals'])


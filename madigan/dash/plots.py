from pathlib import Path
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
        self.graphs = pg.GraphicsLayoutWidget(show=True, title=title)
        self.addWidget(self.graphs)
        self.data = None
        self.colours = {'returns': (255, 228, 181),
                        'loss': (242, 242, 242)}
        self.plots = {}
        self.lines = {}
        self.plots['loss'] = self.graphs.addPlot(title='Loss',
                                          bottom='step', left='Loss')
        self.plots['running_reward'] = self.graphs.addPlot(title='Rewards',
                                                    bottom='step', left='reward')
        self.plots['loss'].showGrid(1, 1)
        self.plots['running_reward'].showGrid(1, 1)
        self.lines['loss'] = self.plots['loss'].plot(y=[])
        self.lines['running_reward'] = self.plots['running_reward'].plot(y=[])

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
        self.lines['running_reward'].setData(y=rewards, pen=self.colours['rewards'])

    def load_from_hdf(self, path):
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
        self.lines['Gt_line'] = self.plots['values'].plot(y=[], name='Gt')
        self.lines['Qt_line'] = self.plots['values'].plot(y=[], name='Qt')
        self.plots['values'].setLabels()

    def set_data(self, data):
        super().set_data(data)
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
        self.episodes = None
        self.data = None
        self.colours = {'equity': (218, 112, 214), 'reward': (242, 242, 242),
                        'cash': (0, 255, 255), 'margin': (255, 86, 0)}
        self.subplots = {}
        self.subplots['prices'] = self.graphs.addPlot(title='Prices', bottom='time',
                                               left='price')
        self.subplots['equity'] = self.graphs.addPlot(title='Equity', bottom='time',
                                               left='Denomination currency')
        self.subplots['reward'] = self.graphs.addPlot(title='Rewards', bottom='time',
                                                left='reward')
        # self.ledgerNormedView = self.addViewBox()
        # self.ledgerNormed = pg.ImageItem(np.empty((1, 1)))
        # self.subplots['ledgerNormed'] = self.ledgerNormedView.addItem(self.ledgerNormed)
        self.subplots['ledgerNormed'] = self.graphs.addViewBox()
        self.subplots['ledgerNormed'].addItem(pg.ImageItem(np.random.randn(10, 100, 1)))
        self.subplots['prices'].showGrid(1, 1)
        self.subplots['equity'].showGrid(1, 1)
        self.subplots['reward'].showGrid(1, 1)
        # self.subplots['ledgerNormed'].showGrid(1, 1)
        self.subplots['equity'].setLabels()
        self.lines={}
        self.lines['equity'] = self.subplots['equity'].plot(y=[])
        self.lines['reward'] = self.subplots['reward'].plot(y=[])
        # self.lines['ledgerNormed'] = self.subplots['ledgerNormed'].plot(y=[])
        self.lines['prices'] = {}

        # self.price_lines= {}
        self.current_pos_line = pg.InfiniteLine(movable=True)
        self.subplots['equity'].addItem(self.current_pos_line)

        self.episode_table = QTableWidget(0, 1)
        self.episode_table.setHorizontalHeaderLabels(['Episode Name'])
        self.positions_table = QTableWidget(0, 2)
        self.positions_table.setHorizontalHeaderLabels(['Ledger', 'Ledger Normed'])
        self.accounting_table = QTableWidget(5, 1)
        self.accounting_table.setVerticalHeaderLabels(['Equity', 'Balance', 'Cash',
                                                      'Available Margin', 'Used Margin',
                                                      'Net PnL'])
        self.addWidget(self.graphs, 0, 0, -1, 8)
        self.addWidget(self.episode_table, 0, 9, 1, 1)
        self.addWidget(self.accounting_table, 1, 9, 1, 1)
        self.addWidget(self.positions_table, 2, 9, 1, 1)
        # self.tables = self.addLayout()
        # self.tables.addItem(self.episode_table)
        self.current_pos_line.sigPositionChanged.connect(self.update_accounting_table)
        self.current_pos_line.sigPositionChanged.connect(self.update_positions_table)

    def clear_data(self):
        for _, line in self.lines.items():
            if isinstance(line, dict):
                for _, price_line in line.items():
                    price_line.setData(y=[])
            else:
                line.setData(y=[])

    def set_data(self, data):
        self.data = data
        if len(data) == 0:
            print('test episode data is empty')
            data = {k: [] for k in self.lines.keys()}
            data['assets'] = 0
        self.lines['equity'].setData(y=data['equity'], pen=self.colours['equity'])
        self.lines['reward'].setData(y=data['reward'], pen=self.colours['reward'])
        # for i, asset in enumerate(data['price']):
        if isinstance(data['price'], (pd.Series, pd.DataFrame)):
            price = np.array(data['price'].tolist())
        else:
            price = np.array(data['price'])
        assert len(price.shape) == 2
        for asset in range(price.shape[1]):
            self.lines['price'][asset] = self.plots['price'].plot(y=price[:, asset],
                                                                  pen=(asset, price.shape[1]))
        if isinstance(data['ledgerNormed'], (pd.DataFrame, pd.Series)):
            ledger = np.array(data['ledgerNormed'].tolist())
        else:
            ledger = np.array(data['ledgerNormed'])
        assert len(ledger.shape) == 2
        self.lines['ledgerNormed'].setData(ledger)
        self.current_pos_line.setValue(len(data['positions'])-1)
        self.update_account_tables()

    def add_test_episode_dir(self, path):
        self.datapath = Path(path)
        self.episodes = list(filter(lambda x: "episode" in str(x), path.iterdir()))
        self.episodes = [f for f in  path.iterdir() if "episode" in str(f.stem)]
        self.episode_table.setRowCount(len(self.episodes))
        for i, episode in enumerate(self.episodes):
            val = QTableWidgetItem(str(episode.stem))
            self.episode_table.setItem(i, 0, val)

    def update_accounting_table(self):
        current_timepoint = int(self.current_pos_line.value())
        try:
            cash = QTableWidgetItem(str(self.data['cash'][current_timepoint]))
            margin = QTableWidgetItem(str(self.data['margin'][current_timepoint]))
            eq = QTableWidgetItem(str(self.data['equity'][current_timepoint]))
            self.accountingTable.setItem(0, 0, cash)
            self.accountingTable.setItem(1, 0, margin)
            self.accountingTable.setItem(2, 0, eq)
        except IndexError:
            print("IndexError")

    def update_positions_table(self):
        current_timepoint = int(self.current_pos_line.value())
        try:
            for asset in range(len(self.data['ledger'][0])):
                # self.positions_table.setItem(asset, 0, QTableWidgetItem(asset))
                # if 0 < current_timepoint <= len(self.positions)-1:
                pos = self.data['ledgerNomed'][current_timepoint][asset]
                self.positions_table.setItem(asset, 0, QTableWidgetItem(str(pos)))
                pos = self.data['ledger'][current_timepoint][asset]
                self.positions_table.setItem(asset, 1, QTableWidgetItem(str(pos)))
        except IndexError:
            import traceback
            traceback.print_exc()

    def load_from_hdf(self, path):
        data = pd.read_hdf(path, key='full_run')
        self.set_data(data)


class TestEpisodePlotsDQN(TestEpisodePlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({'Qt': (255, 86, 0)})

    def set_data(self, data):
        super().set_data(data)
        self.lines['Qt'].setData(y=data['Qt'], pen=self.colours['Qt'])

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
        self.plots['equity'] = self.graphs.addPlot(title='Mean Equity over Episodes',
                                                              bottom='training_steps',
                                                              left='Denomination currency')
        self.plots['reward'] = self.graphs.addPlot(title='Mean Returns over Episodes',
                                                               bottom='training_steps',
                                                               left='returns (proportion)')
        self.plots['cash'] = self.graphs.addPlot(title='Mean Cash over Episodes',
                                                            bottom='training steps',
                                                            left='returns (proportion)')
        self.plots['equity'].showGrid(1, 1)
        self.plots['reward'].showGrid(1, 1)
        self.plots['cash'].showGrid(1, 1)
        self.plots['cash'].setLabels()
        self.lines = {}
        self.lines['equity'] = self.plots['equity'].plot(y=[])
        self.lines['reward'] = self.plots['reward'].plot(y=[])
        self.lines['cash']= self.plots['cash'].plot(y=[])
        # self.lines['margin']= self.plots['margin'].plot(y=[])

    def clear_data(self):
        for _, line in self.lines.items():
            line.setData(y=[])

    def set_data(self, data):
        if len(data) == 0:
            print("test data is empty")
            data = {k: [] for k in self.lines.keys()}
        self.lines['equity'].setData(y=data['equity'], pen=self.colours['equity'])
        self.lines['reward'].setData(y=data['reward'], pen=self.colours['reward'])
        self.lines['cash'].setData(y=data['cash'], pen=self.colours['cash'])
        self.lines['margin'].setData(y=data['cash'], pen=self.colours['margin'])


class TestHistoryPlotsDQN(TestHistoryPlots):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.colours.update({'Qt': (255, 86, 0)})

    def set_data(self, data):
        super().set_data(data)
        self.lines['Qt'].setData(y=data['Qt'], pen=self.colours['Qt'])


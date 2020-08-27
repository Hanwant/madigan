from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtWidgets import QMessageBox
import pyqtgraph as pg
from .utils import make_dark_palette


def on_top_click():
    alert = QMessageBox()
    alert.setText('You clicked top button')
    alert.exec_()

def run_test_dash():
    import numpy as np

    x = np.linspace(-10, 10, 10000)
    y = np.sin(x) + np.random.normal(0, 0.1, len(x))

    app = QApplication([])

    app.setStyle('Fusion')
    palette = make_dark_palette()
    app.setPalette(palette)

    window = QWidget()
    layout = QVBoxLayout()

    button1 = QPushButton('Top')
    button1.clicked.connect(on_top_click)
    layout.addWidget(button1)
    layout.addWidget(QPushButton('Bottom'))
    label = QLabel('Yoyo')
    layout.addWidget(label)

    plot1 = pg.PlotWidget()
    plot1.plot(x, y, pen='w')
    layout.addWidget(plot1)


    window.setLayout(layout)
    window.show()
    app.exec_()





if __name__ == "__main__":
    run_test_dash()




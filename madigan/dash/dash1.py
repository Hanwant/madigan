from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtWidgets import QMessageBox
import pyqtgraph as pg

dark_palette = QPalette()
dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
dark_palette.setColor(QPalette.WindowText, Qt.white)
# dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
dark_palette.setColor(QPalette.ToolTipText, Qt.white)
dark_palette.setColor(QPalette.Text, Qt.white)
dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
dark_palette.setColor(QPalette.ButtonText, Qt.white)
dark_palette.setColor(QPalette.BrightText, Qt.red)
dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
dark_palette.setColor(QPalette.HighlightedText, Qt.black)

def on_top_click():
    alert = QMessageBox()
    alert.setText('You clicked top button')
    alert.exec_()


if __name__ == "__main__":
    import numpy as np

    app = QApplication([])

    app.setStyle('Fusion')
    app.setPalette(dark_palette)

    window = QWidget()
    layout = QVBoxLayout()

    button1 = QPushButton('Top')
    button1.clicked.connect(on_top_click)
    layout.addWidget(button1)
    layout.addWidget(QPushButton('Bottom'))
    label = QLabel('Yoyo')
    layout.addWidget(label)

    x = np.linspace(-10, 10, 10000)
    y = np.sin(x) + np.random.normal(0, 0.1, len(x))
    plot1 = pg.PlotWidget()
    plot1.plot(x, y, pen='w')
    layout.addWidget(plot1)


    window.setLayout(layout)
    window.show()
    app.exec_()



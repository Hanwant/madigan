# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dash/dash_synth_base.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1326, 705)
        MainWindow.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(53, 50, 47);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.PlotsWidget = QtWidgets.QWidget(self.centralwidget)
        self.PlotsWidget.setGeometry(QtCore.QRect(230, 80, 771, 531))
        self.PlotsWidget.setObjectName("PlotsWidget")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(10, 570, 211, 21))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.PlotsLabel = QtWidgets.QLabel(self.centralwidget)
        self.PlotsLabel.setGeometry(QtCore.QRect(230, 60, 681, 17))
        self.PlotsLabel.setObjectName("PlotsLabel")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 80, 211, 491))
        self.layoutWidget.setObjectName("layoutWidget")
        self.ControlsGridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.ControlsGridLayout.setContentsMargins(0, 0, 0, 0)
        self.ControlsGridLayout.setObjectName("ControlsGridLayout")
        self.ParamsEdit = QtWidgets.QTextEdit(self.layoutWidget)
        self.ParamsEdit.setStyleSheet("")
        self.ParamsEdit.setObjectName("ParamsEdit")
        self.ControlsGridLayout.addWidget(self.ParamsEdit, 2, 0, 1, 2)
        self.LoadConfigButton = QtWidgets.QPushButton(self.layoutWidget)
        self.LoadConfigButton.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.LoadConfigButton.setObjectName("LoadConfigButton")
        self.ControlsGridLayout.addWidget(self.LoadConfigButton, 1, 1, 1, 1)
        self.FilenameLabel = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.FilenameLabel.setFont(font)
        self.FilenameLabel.setStyleSheet("")
        self.FilenameLabel.setObjectName("FilenameLabel")
        self.ControlsGridLayout.addWidget(self.FilenameLabel, 1, 0, 1, 1)
        self.SaveConfigButton = QtWidgets.QPushButton(self.layoutWidget)
        self.SaveConfigButton.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.SaveConfigButton.setObjectName("SaveConfigButton")
        self.ControlsGridLayout.addWidget(self.SaveConfigButton, 3, 1, 1, 1)
        self.RunCommand = QtWidgets.QPushButton(self.layoutWidget)
        self.RunCommand.setStyleSheet("background-color: rgb(181, 255, 160);\n"
"background-color: rgb(0, 0, 85);")
        self.RunCommand.setObjectName("RunCommand")
        self.ControlsGridLayout.addWidget(self.RunCommand, 4, 0, 1, 2)
        self.ConfigLabel = QtWidgets.QLabel(self.layoutWidget)
        self.ConfigLabel.setStyleSheet("")
        self.ConfigLabel.setObjectName("ConfigLabel")
        self.ControlsGridLayout.addWidget(self.ConfigLabel, 0, 0, 1, 2)
        self.CashTable = QtWidgets.QTableWidget(self.centralwidget)
        self.CashTable.setGeometry(QtCore.QRect(1000, 430, 221, 121))
        self.CashTable.setStyleSheet("background-color: rgb(0, 85, 127);\n"
"background-color: rgb(0, 0, 49);")
        self.CashTable.setObjectName("CashTable")
        self.CashTable.setColumnCount(0)
        self.CashTable.setRowCount(3)
        item = QtWidgets.QTableWidgetItem()
        self.CashTable.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.CashTable.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.CashTable.setVerticalHeaderItem(2, item)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(1000, 70, 221, 361))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.PositionsLayout = QtWidgets.QGridLayout(self.layoutWidget1)
        self.PositionsLayout.setContentsMargins(0, 0, 0, 0)
        self.PositionsLayout.setObjectName("PositionsLayout")
        self.PortfolioLabel = QtWidgets.QLabel(self.layoutWidget1)
        self.PortfolioLabel.setObjectName("PortfolioLabel")
        self.PositionsLayout.addWidget(self.PortfolioLabel, 0, 1, 1, 1)
        self.PositionsTable = QtWidgets.QTableWidget(self.layoutWidget1)
        self.PositionsTable.setStyleSheet("background-color: rgb(37, 74, 55);\n"
"background-color: rgb(0, 52, 0);")
        self.PositionsTable.setObjectName("PositionsTable")
        self.PositionsTable.setColumnCount(2)
        self.PositionsTable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.PositionsTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.PositionsTable.setHorizontalHeaderItem(1, item)
        self.PositionsLayout.addWidget(self.PositionsTable, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1326, 20))
        self.menubar.setObjectName("menubar")
        self.menuDash_Synth = QtWidgets.QMenu(self.menubar)
        self.menuDash_Synth.setObjectName("menuDash_Synth")
        MainWindow.setMenuBar(self.menubar)
        self.menuDash_Synth.addSeparator()
        self.menuDash_Synth.addSeparator()
        self.menubar.addAction(self.menuDash_Synth.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.PlotsLabel.setText(_translate("MainWindow", "Plots"))
        self.ParamsEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans Serif\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">{\'name\': \'run0\',</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">\'discrete_actions\': true,</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">\'action_atoms\':11</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">\'nsteps\':100</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">}</span></p></body></html>"))
        self.LoadConfigButton.setText(_translate("MainWindow", "Load Config"))
        self.FilenameLabel.setText(_translate("MainWindow", "config.json"))
        self.SaveConfigButton.setText(_translate("MainWindow", "Save"))
        self.RunCommand.setText(_translate("MainWindow", "Run"))
        self.ConfigLabel.setText(_translate("MainWindow", "CONFIG "))
        item = self.CashTable.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Cash"))
        item = self.CashTable.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Available Margin"))
        item = self.CashTable.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "Equity"))
        self.PortfolioLabel.setText(_translate("MainWindow", "Portfolio"))
        item = self.PositionsTable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Asset"))
        item = self.PositionsTable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Position"))
        self.menuDash_Synth.setTitle(_translate("MainWindow", "Dash-Synth"))

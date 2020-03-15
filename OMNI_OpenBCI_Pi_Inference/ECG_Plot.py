from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import random
import numpy as np
from numpy import arange, sin, cos, pi
import sys 
import csv
import numpy as np
import tkinter as tk
import pyqtgraph.ptime as ptime


def create_dashboard(data, peaks):
    global i, cnt, j, pk, xk, yk, posx, posy, text, hr

    class KeyPressWindow(pg.GraphicsLayoutWidget):
        sigKeyPress = QtCore.pyqtSignal(object)
        def keyPressEvent(self, ev):
            self.scene().keyPressEvent(ev)
            self.sigKeyPress.emit(ev)

    def keyPressed(evt):
        if evt.key() == QtCore.Qt.Key_P:
            timer.stop()
        if evt.key() == QtCore.Qt.Key_S:
            timer.start()
    

    timer = QtCore.QTimer()
    app = QtGui.QApplication(sys.argv)

   
    w1 = KeyPressWindow()
    w1.setBackground((155, 155, 155))
    w1.sigKeyPress.connect(keyPressed)
    w1.show()
    w1.resize(tk.Tk().winfo_screenwidth(), tk.Tk().winfo_screenheight())
    
    cnt = 0
    s4 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(0.5), brush=pg.mkBrush(0, 0, 255, 120))
    st =1
    j = 0
    ax=[]
    hr =50
   ############ECG PLOT#############
    p = w1.addPlot(row = 0, col = 0, colspan = 3)
    p.hideAxis('left')
    p.hideAxis('bottom')
    plt_grid = pg.GridItem()
    # plt_grid.setPen()
    # plt_grid.hideAxis('left')
    vb = p.getViewBox()
    vb.setBackgroundColor((255, 255, 255))
    curveax = p.plot( pen=pg.mkPen((0, 0, 0), width=1))
    p.addItem(curveax)
    p.addItem(s4)
    p.addItem(plt_grid)
    # p.showGrid(x = True, y = True)
    #####################################3


    #####HEART RATE#############
    x = np.linspace(-5, 5, 10)
    y = np.sin(x) / x

    p1 = w1.addPlot(row = 1, col = 0)
    p1.plot(x = x,y = y, pen=(255, 255, 255))
    
    vb1 = p1.getViewBox()
    vb1.setBackgroundColor((255, 255, 255))

    
    fileName1 = '/home/hticpose/Pictures/Picture3.jpg'
    img1 = pg.QtGui.QGraphicsPixmapItem(pg.QtGui.QPixmap(fileName1))
    img1.scale(1, -1)
    
    p1.hideAxis('left')
    p1.hideAxis('bottom') 
    p1.addItem(img1)

    text = pg.TextItem( "test",color = (0,0,0), anchor=(-0.3,0.5))
    changingLabel = QtGui.QLabel()
    font = changingLabel.font()
    font.setPointSize(86)
    text.setFont(font)

    texthr = pg.TextItem( "test",color = (0,0,0), anchor=(-0.3,0.5))
    changingLabel = QtGui.QLabel()
    fonthr = changingLabel.font()
    fonthr.setPointSize(40)
    texthr.setFont(fonthr)
    texthr.setText("HR")
    texthr.setPos(130, y.max()/2 - 220)
    
    p1.addItem(text) 
    p1.addItem(texthr) 
    
    #################################
    
    #####BREATHING RATE#############
    x1 = np.linspace(-5, 5, 10)
    y1 = np.sin(x1) / x1
    
    p2 = w1.addPlot(row = 1, col = 1)
    p2.plot(x = x1,y = y1, pen=(255, 255, 255))

    vb2 = p2.getViewBox()
    vb2.setBackgroundColor((255, 255, 255))
    
    fileName2 = '/home/hticpose/Pictures/edit br.jpg'
    img2 = pg.QtGui.QGraphicsPixmapItem(pg.QtGui.QPixmap(fileName2))
    img2.scale(1, -1)
    p2.addItem(img2)
    p2.hideAxis('left')
    p2.hideAxis('bottom') 
    text1 = pg.TextItem( "test",color = (0,0,0), anchor=(-0.3,0.5))
    
    changingLabel = QtGui.QLabel()
    font1 = changingLabel.font()
    font1.setPointSize(86)
    text1.setFont(font1)
    
    
    textbr = pg.TextItem( "test",color = (0,0,0), anchor=(-0.3,0.5))
    changingLabel = QtGui.QLabel()
    fontbr = changingLabel.font()
    fontbr.setPointSize(40)
    textbr.setFont(fontbr)
    textbr.setText("BR")
    textbr.setPos(130, y.max()/2 - 220)
    p2.addItem(text1) 
    p2.addItem(textbr) 
    #################################
    

    ###### MESSAGE #############
    p3 = w1.addPlot(row = 1, col = 2)
    p3.plot(x = x1,y = y1, pen=(255, 255, 255))
    vb3 = p3.getViewBox()
    vb3.setBackgroundColor((255, 255, 255))
    
    p3.hideAxis('left')
    p3.hideAxis('bottom')   
    ##########################

    

    xk = [x for x in peaks[0] if x <= 1000]
    yk = [float(data[0][y]) for y in xk]
    
    for i in range(st,st+1000):
        ax.append(float(data[0][i]))
    cnt = len(xk)
   
    def update():
        global i,j,cnt,xk,yk,text,k,hr
        
        s4.clear()
        ax.pop(0)
        
        if(len(xk)!=0 and xk[0] < 1):
            xk.pop(0)
            yk.pop(0)
        xk = [x-1 for x in xk]
        
        if(i+1<len(data[j])):
            i += 1
            if(cnt < len(peaks[j]) and i == peaks[j][cnt]):
                cnt += 1
                xk.append(1000)
                yk.append(data[j][i])        
        else:
            hr += 100
            cnt = 0 
            j += 1
            i = 0
        if(hr>150):
            hr = 50
        
        ax.append(float(data[j][i]))
        posx = [x - 1 for x in xk]
        posy = yk
        print(i)
        # if i%50 == 0:
        #     p1.removeItem(img1)
        #     p1.removeItem(text)
        #     p1.removeItem(texthr)
        #     # text.setText('')
        # else:
        #     p1.addItem(img1)
        #     p1.addItem(text) 
        #     p1.addItem(texthr) 
    
        
        text.setText('{}' .format(hr))

        
        text1.setText('{}' .format(hr))
        if hr<100:
            text.setPos(95, y.max()/2 - 160)
            text1.setPos(95, y.max()/2 - 160)
        else:
            text.setPos(60, y.max()/2 - 160)
            text1.setPos(60, y.max()/2 - 160)
        s4.addPoints(x = posx, y = posy)
        curveax.setData(ax)
        
    
    
    timer.timeout.connect(update)
    timer.start(5)
         
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
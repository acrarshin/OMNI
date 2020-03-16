from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import sys 
import tkinter as tk


def create_dashboard(data, peaks,all_hr, all_br):
    """[summary]
    
    Arguments:
        data {numpy.ndarray} -- ECG plot
        peaks {list} -- R peaks
        all_hr {list} -- Heart rate per minute
        all_br {list} -- Breathing rate per minute
    """
    global i, rpindex, win, rpeakx, rpeaky, hr, br, hrslid, brslid, slidflaghr, slidflagbr, slidPoshr, slidPosbr, hrindex, chk, chkbr, popflag, trigflag, popwind
  
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
    
    mscreen = KeyPressWindow()
    mscreen.sigKeyPress.connect(keyPressed)
    mscreen.show()
    mscreen.resize(tk.Tk().winfo_screenwidth(), tk.Tk().winfo_screenheight())
    
    popwind = QtWidgets.QMessageBox()
    popwind.setIcon(QtWidgets.QMessageBox.Critical)
    popwind.setWindowTitle("WARNING/COURSE OF ACTION")
    popwind.setFont(QtGui.QFont("sans-serif", 16))
    changingLabel = QtGui.QLabel()

    #########IDENTIFIERS###################

    rpindex = 0
    win = 37
    hrindex = 37
    slidPoshr = 0.5
    slidPosbr = 0.5
    popflag = 0 
    trigflag = 1
    ecg = []
    slidflaghr = 0
    slidflagbr = 0
    hr = all_hr[hrindex]
    br = all_br[hrindex]
   ############ECG PLOT#############

    rpeaks = pg.ScatterPlotItem(size=10, pen=pg.mkPen(0.5), brush=pg.mkBrush(0, 0, 255, 120))
    ecgwind = mscreen.addPlot(row = 0, col = 0, colspan = 3)
    ecgwind.hideAxis('left')
    ecgwind.hideAxis('bottom')
    plt_grid = pg.GridItem()
    ecgbg = ecgwind.getViewBox()
    ecgbg.setBackgroundColor((255, 255, 255))
    curve_ecg = ecgwind.plot( pen=pg.mkPen((0, 0, 0), width=1))
    ecgwind.addItem(curve_ecg)
    ecgwind.addItem(rpeaks)
    ecgwind.addItem(plt_grid)

    #####################################3

    #####HEART RATE#############
    curvex1 = np.linspace(-5, 5, 10)
    curvey1 = np.sin(curvex1) / curvex1

    hrwind = mscreen.addPlot(row = 1, col = 0)
    hrwind.plot(x = curvex1, y = curvey1, pen=(255, 255, 255))
    
    hrbg = hrwind.getViewBox()
    hrbg.setBackgroundColor((255, 255, 255))

    fileNamehr = '/home/hticpose/Pictures/Picture3.jpg'
    imghr = pg.QtGui.QGraphicsPixmapItem(pg.QtGui.QPixmap(fileNamehr))
    imghr.scale(1, -1)
    
    hrwind.hideAxis('left')
    hrwind.hideAxis('bottom') 
    hrwind.addItem(imghr)

    hrval = pg.TextItem(color = (0,0,0), anchor=(-0.3,0.5))
    fonthrval = changingLabel.font()
    fonthrval.setPointSize(86)
    hrval.setFont(fonthrval)
    fonthrval.setPointSize(40)

    texthr = pg.TextItem(color = (0,0,0), anchor=(-0.3,0.5))
    fonthr = changingLabel.font()
    fonthr.setPointSize(40)
    texthr.setFont(fonthr)
    texthr.setText("HR")
    texthr.setPos(130, curvey1.max()/2 - 220)
    
    hrwind.addItem(hrval) 
    hrwind.addItem(texthr) 
    
    #################################
    
    #####BREATHING RATE#############
    curvex2 = np.linspace(-5, 5, 10)
    curvey2 = np.sin(curvex2) / curvex2
    
    brwind = mscreen.addPlot(row = 1, col = 1)
    brwind.plot(x = curvex2,y = curvey2, pen=(255, 255, 255))

    brbg = brwind.getViewBox()
    brbg.setBackgroundColor((255, 255, 255))
    
    filebr = '/home/hticpose/Pictures/edit br.jpg'
    imgbr = pg.QtGui.QGraphicsPixmapItem(pg.QtGui.QPixmap(filebr))
    imgbr.scale(1, -1)
    brwind.addItem(imgbr)
    brwind.hideAxis('left')
    brwind.hideAxis('bottom') 

    brval = pg.TextItem(color = (0,0,0), anchor=(-0.3,0.5))
    font1 = changingLabel.font()
    font1.setPointSize(86)
    brval.setFont(font1)
    
    textbr = pg.TextItem(color = (0,0,0), anchor=(-0.3,0.5))
    fontbr = changingLabel.font()
    fontbr.setPointSize(40)
    textbr.setFont(fontbr)
    textbr.setText("BR")
    textbr.setPos(130, curvey2.max()/2 - 220)

    brwind.addItem(brval) 
    brwind.addItem(textbr) 
    #################################
    

    ###### SLIDER #############
    slidwind = mscreen.addPlot(row = 1, col = 2)
    slidbg = slidwind.getViewBox()
    slidbg.setBackgroundColor((255, 255, 255))
    curvex3 = np.linspace(0, 860, 2)
    curvey3 = [-158 for x in curvex3]

    fileslider = '/home/hticpose/Pictures/red(1).jpg'
    imgslid = pg.QtGui.QGraphicsPixmapItem(pg.QtGui.QPixmap(fileslider))
    imgslid.scale(1, -1)
    slidwind.addItem(imgslid)

    linehr = slidwind.plot(x = curvex3,y = curvey3, pen=(255, 255, 255))
    hrslid = pg.CurvePoint(linehr)
    slidwind.addItem(hrslid)

    texthr1 = pg.TextItem(color = (0,0,0), anchor=(-0.3,0.5))
    fonthr1 = changingLabel.font()
    fonthr1.setPointSize(40)
    texthr1.setFont(fonthr1)
    texthr1.setText("HR")
    texthr1.setPos(320, curvey1.max() - 270)

    slidwind.addItem(texthr1) 
    arrowhr = pg.ArrowItem(angle=90)
    arrowhr.setStyle(headLen = 30)
    arrowhr.setParentItem(hrslid)
    hrslid.setPos(slidPoshr) 

    curvey4 = [-465 for x in curvex3]
    linebr = slidwind.plot(x = curvex3 ,y = curvey4 , pen=(255, 255, 255))
    brslid = pg.CurvePoint(linebr)
    slidwind.addItem(brslid)

    textbr1 = pg.TextItem(color = (0,0,0), anchor=(-0.3,0.5))
    fontbr1 = changingLabel.font()
    fontbr1.setPointSize(40)
    textbr1.setFont(fontbr1)
    textbr1.setText("BR")
    textbr1.setPos(320, curvey1.max() - 570)

    slidwind.addItem(textbr1) 
    arrowbr = pg.ArrowItem(angle=90)
    arrowbr.setStyle(headLen = 30)
    arrowbr.setParentItem(brslid)
    brslid.setPos(slidPosbr) 

    slidwind.hideAxis('left')
    slidwind.hideAxis('bottom')   
    ##########################

    rpeakx = [x for x in peaks[win] if x <= 1000]
    rpeaky = [float(data[win][y]) for y in rpeakx]
    chk = 1
    chkbr = 1
    for i in range(1,1001):
        ecg.append(float(data[win][i]))
    rpindex = len(rpeakx)
   
    def update():
        global i, win, rpindex, rpeakx, rpeaky, hr, br, hrslid, brslid, slidflaghr, slidflagbr, slidPoshr, slidPosbr, hrindex, chk, chkbr, popflag, trigflag, popwind
        
        rpeaks.clear()
        ecg.pop(0)
  
        if(len(rpeakx)!=0 and rpeakx[0] < 1):
            rpeakx.pop(0)
            rpeaky.pop(0)
        rpeakx = [x-1 for x in rpeakx]
        
        if(i+1<len(data[win])):
            i += 1
            if(rpindex < len(peaks[win]) and i == peaks[win][rpindex]):
                rpindex += 1
                rpeakx.append(1000)
                rpeaky.append(data[win][i])        
        else:
            rpindex = 0 
            win += 1
            i = 0
      
        if 100 <= hr <110:
            meterhr = -1
            slidflaghr = 1
        elif hr < 100:
            popflag = 1
            meterhr = -2
            slidflaghr = 1
        elif 150 < hr < 160:
            meterhr = 1
            slidflaghr = 1
        elif hr >= 160:
            popflag = 1
            meterhr = 2
            slidflaghr = 1  
        else:
            if(slidPoshr != 0.5):
                slidflaghr = 1
                meterhr = 0
        if br > 45:
            popflag = 1
            slidflagbr = 1
            meterbr = 2

        if popflag == 1 and trigflag == 1:
            trigflag = 0
            if hr >= 160:
                popwind.setText("Heart Rate High\nSeek medical attention if high heart rate persists")
            elif hr <= 110:
                popwind.setText("Heart Rate Low\nSeek medical attention if low heart rate persists")
            else:
                popwind.setText("Breathing Rate Abnormal\nSeek medical attention if following symptoms are displayed:\n1) Blueness\n2) Severe chest indrawing")
            
            popwind.show()
        
        if slidflaghr == 1 and chk == 1:
            
            slidDest, direct = slider(meterhr, slidPoshr)
            slidPoshr += direct * 0.002
            
            if(direct == -1):
                if slidPoshr <= slidDest:
                    slidflaghr = 0
                    chk = 0
            elif(direct == 1):
                if slidPoshr >= slidDest:
                    slidflaghr = 0
                    chk = 0
            hrslid.setPos(slidPoshr)    

        if slidflagbr == 1 and chkbr == 1:
            slidDestbr, directbr = slider(meterbr, slidPosbr)
            slidPosbr += directbr * 0.002
            if(directbr == -1):
                if slidPosbr <= slidDestbr:
                    slidflagbr = 0
                    chkbr = 0
            elif(directbr == 1):
                if slidPosbr >= slidDestbr:
                    slidflagbr = 0
                    chkbr = 0
            brslid.setPos(slidPosbr)  

        ecg.append(float(data[win][i]))
        rpkx = [x - 1 for x in rpeakx]
        rpky = rpeaky
     
        if i % 2500 == 0:
            
            hrindex += 1
            hr = all_hr[hrindex] - 15
            br = all_br[hrindex] + 31
            chk = 1
            chkbr = 1
            trigflag = 1
            
        hrval.setText('{} ' .format(hr))
        brval.setText('{} ' .format(br))
        popflag = 0
        
        if hr<100:
            hrval.setPos(95, -159.5253)
        else:
            hrval.setPos(60, -159.5253)

        brval.setPos(95, -159.5253)
        rpeaks.addPoints(x = rpkx, y = rpky)
        curve_ecg.setData(ecg)

    def slider(meter, slidPoshr):

        if meter == -2:
            slidDest = 0
            direct = -1
        elif meter == -1:
            slidDest = 0.25
            if slidPoshr>0.5:
                direct = -1
            elif slidPoshr<0.5:
                direct = 1
            direct = -1
        elif meter == 1:
            slidDest = 0.75
            if slidPoshr>0.5:
                direct = -1
            elif slidPoshr<0.5:
                direct = 1
        elif meter == 2:
            slidDest = 1
            direct = 1
        elif meter == 0:
            slidDest = 0.5
            if slidPoshr>0.5:
                direct = -1
            elif slidPoshr<0.5:
                direct = 1

        return slidDest, direct
        
    timer.timeout.connect(update)
    timer.start(5)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
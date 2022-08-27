import sys
from udp import UdpReceiver
import threading
import time
import numpy as np
import configparser
import csv
import random
import math

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

DEBUG = 1

# コンフィグ
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
SEND_CHUNK = config["SEND"].getint("CHUNK")   # sendの処理単位(sample)
DISP_CHUNK = config["DISPLAY"].getint("CHUNK")  # 表示の処理単位(sample)


class Window(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        # Window設定
        self.resize(500, 500)
        self.setBackground("w")
        self.setWindowTitle("BirdResearch")
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption("foreground", (0, 0, 0))

        # PlotItem()の追加
        self.p = self.addPlot(title="localization", row=0, col=0)
        self.p.showGrid(x=True, y=True)
        self.p.setXRange(-5, 5)
        self.p.setYRange(-5, 5)
        self.p.enableAutoRange("xy", False)
        # PlotDataItem()をaddItem()する
        pen = pg.mkPen((0, 0, 0), width=2)
        self.lines = [
            self.p.plot(pen=pen, name="Mic0"),
            self.p.plot(pen=pen, name="Mic1")
        ]
        # TextItem()の追加
        font = QtGui.QFont()
        font.setPixelSize(20)
        self.t = pg.TextItem(text="unknown", color=(0, 0, 0))
        self.t.setPos(1.5, 5.3)
        self.t.setFont(font)
        self.p.addItem(self.t)
        self.t2 = pg.TextItem(text="1.00", color=(0, 0, 0))
        self.t2.setPos(1.5, 4.8)
        self.p.addItem(self.t2)
        self.t2.setFont(font)

        # 更新に使われるデータ変数
        plot_data_unit = {
            "x": np.linspace(0, 20, 10),
            "y": np.linspace(0, 20, 10)
        }
        self.plot_data = [plot_data_unit.copy(), plot_data_unit.copy()]
        self.label = "unknown"
        self.prob = str(1.00)

        # タイマー設定
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(100)

    def _update(self):
        for mic_id in range(2):
            x = self.plot_data[mic_id]["x"]
            y = self.plot_data[mic_id]["y"]
            self.lines[mic_id].setData(x, y)
            self.t.setText(self.label)
            self.t2.setText(self.prob)

class Display():
    def __init__(self, receiver:UdpReceiver):
        self.receiver = receiver
        self.mic_loc = [np.array([-1, 0]), np.array([1, 0])]

        display_buff_unit = {
            "direction": np.empty(0, dtype=np.int),
            "audio_id": np.empty(0, dtype=np.int),
            "pred_idx": np.empty(0, dtype=np.int),
            "pred_prob": np.empty(0, dtype=np.float)
        }
        self.display_buff = [display_buff_unit.copy(), display_buff_unit.copy()]

        # 識別ラベル
        self.label_list = []
        with open("label01_mapping.tsv", "r") as f:
            tsv = csv.reader(f, delimiter="\t")
            for row in tsv:
                self.label_list.append(row[1])
        
        # ウィンドウ表示
        self.win = Window()
        self.win.show()

    def start(self):
        print("Starting display......")
        process_thread = threading.Thread(target=self.__process)
        process_thread.setDaemon(True)
        process_thread.start()
        print("Started display!")
        pass

    def __process(self):
        while True:
            if not self.receiver.receive_queue.empty():
                self.__get_audio_from_queue(SEND_CHUNK)
            for mic_id in range(2):
                if self.display_buff[mic_id]["direction"].shape[0] > DISP_CHUNK:
                    self.__display(
                        mic_id=mic_id,
                        chunk=DISP_CHUNK
                    )
            time.sleep(0.01)

    def __get_audio_from_queue(self, chunk):
        # 受信キューからデータ取り出し・分割
        data_bin = self.receiver.receive_queue.get()
        data = np.frombuffer(data_bin, dtype="float")
        direction = data[:chunk].astype(np.int)
        audio_id = data[chunk:chunk*2].astype(np.int)
        pred_idx = data[chunk*2:chunk*3].astype(np.int)
        pred_prob = data[chunk*3:chunk*4].astype(np.float)
        mic_id = data[-1].astype(np.int)

        # データをバッファに追加
        self.display_buff[mic_id]["direction"] = np.concatenate([self.display_buff[mic_id]["direction"], direction], axis=0)
        self.display_buff[mic_id]["audio_id"] = np.concatenate([self.display_buff[mic_id]["audio_id"], audio_id], axis=0)
        self.display_buff[mic_id]["pred_idx"] = np.concatenate([self.display_buff[mic_id]["pred_idx"], pred_idx], axis=0)
        self.display_buff[mic_id]["pred_prob"] = np.concatenate([self.display_buff[mic_id]["pred_prob"], pred_prob], axis=0)

    def __display(self, mic_id, chunk):
        direction = self.display_buff[mic_id]["direction"][:chunk]
        self.display_buff[mic_id]["direction"] = self.display_buff[mic_id]["direction"][chunk:]
        audio_id = self.display_buff[mic_id]["audio_id"][:chunk]
        self.display_buff[mic_id]["audio_id"] = self.display_buff[mic_id]["audio_id"][chunk:]
        pred_idx = self.display_buff[mic_id]["pred_idx"][:chunk]
        self.display_buff[mic_id]["pred_idx"] = self.display_buff[mic_id]["pred_idx"][chunk:]
        pred_prob = self.display_buff[mic_id]["pred_prob"][:chunk]
        self.display_buff[mic_id]["pred_prob"] = self.display_buff[mic_id]["pred_prob"][chunk:]
        theta = 2 * np.pi * direction[0] / 72

        r = np.linspace(0, 20, 10)
        x = self.mic_loc[mic_id][0] + r * np.cos(theta)
        y = self.mic_loc[mic_id][1] + r * np.sin(theta)

        # ウィンドウ表示用のデータ設定
        self.win.plot_data[mic_id]["x"] = x
        self.win.plot_data[mic_id]["y"] = y
        self.win.label = self.label_list[pred_idx[0]]
        self.win.prob = str(round(pred_prob[0], 3))


def main():
    # Udp受信側の設定
    receiver_from_discriminator = UdpReceiver(
        src_ip="127.0.0.1",
        src_port=30000
    )
    # ウィンドウの設定
    app = QtGui.QApplication([])
    # 表示器の設定
    display = Display(
        receiver=receiver_from_discriminator
    )
    # 表示器の開始
    display.start()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        receiver_from_discriminator.close()
        sys.exit(0)


if __name__ == "__main__":
    main()

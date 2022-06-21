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

# コンフィグ
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
SEND_CHUNK = config["SEND"].getint("CHUNK")   # sendの処理単位(sample)
DISP_CHUNK = config["DISPLAY"].getint("CHUNK")  # 表示の処理単位(sample)


class Window(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.resize(600, 500)
        self.setWindowTitle("BirdResearch")
        pg.setConfigOptions(antialias=True)

        self.p = self.addPlot(title="localization")
        self.p.showGrid(x=True, y=True)
        self.p.setXRange(-5, 5)
        self.p.setYRange(-5, 5)
        self.curve = [
            self.p.plot(pen='c', name="Mic0"),
            self.p.plot(pen='c', name="Mic1")
        ]
        r = np.linspace(0, 20, 10)
        plot_data_unit = {
            "x": np.linspace(0, 20, 10),
            "y": np.linspace(0, 20, 10)
        }
        self.plot_data = [plot_data_unit.copy(), plot_data_unit.copy()]
        self.label_data = ["unknown", "unknown"]

        self.label = [pg.LabelItem(justify="right"), pg.LabelItem(justify="right")]
        self.label[0].setText("unknown")
        self.label[1].setText("unknown")
        self.addItem(self.label[0])
        self.addItem(self.label[1])

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(100)

    def _update(self):
        for mic_id in range(2):
            x = self.plot_data[mic_id]["x"]
            y = self.plot_data[mic_id]["y"]
            self.curve[mic_id].setData(x, y)
            self.p.enableAutoRange("xy", False)
            self.label[mic_id].setText(self.label_data[mic_id])

class Display():
    def __init__(self, receiver:UdpReceiver):
        self.receiver = receiver
        self.mic_loc = [np.array([-1, 0]), np.array([1, 0])]
        self.win = Window()
        display_buff_unit = {
            "direction": np.empty(0, dtype=np.int),
            "audio_id": np.empty(0, dtype=np.int),
            "label": np.empty(0, dtype=np.int)
        }
        self.display_buff = [display_buff_unit.copy(), display_buff_unit.copy()]
        self.label_list = []
        with open("label01_mapping.tsv", "r") as f:
            tsv = csv.reader(f, delimiter="\t")
            for row in tsv:
                self.label_list.append(row[1])
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

    def __get_audio_from_queue(self, chunk):
        data_bin = self.receiver.receive_queue.get()
        data = np.frombuffer(data_bin, dtype="int64")
        direction = data[:chunk].astype(np.int)
        audio_id = data[chunk:chunk*2].astype(np.int)
        label = data[chunk*2:chunk*3].astype(np.int)
        mic_id = data[-1].astype(np.int)
        self.display_buff[mic_id]["direction"] = np.concatenate([self.display_buff[mic_id]["direction"], direction], axis=0)
        self.display_buff[mic_id]["audio_id"] = np.concatenate([self.display_buff[mic_id]["audio_id"], audio_id], axis=0)
        self.display_buff[mic_id]["label"] = np.concatenate([self.display_buff[mic_id]["label"], label], axis=0)
        a = self.display_buff[0]["direction"].shape
        b = self.display_buff[1]["direction"].shape
        print(f"audio_buff: {a} {b}")

    def __display(self, mic_id, chunk):
        direction = self.display_buff[mic_id]["direction"][:chunk]
        self.display_buff[mic_id]["direction"] = self.display_buff[mic_id]["direction"][chunk:]
        audio_id = self.display_buff[mic_id]["audio_id"][:chunk]
        self.display_buff[mic_id]["audio_id"] = self.display_buff[mic_id]["audio_id"][chunk:]
        label = self.display_buff[mic_id]["label"][:chunk]
        self.display_buff[mic_id]["label"] = self.display_buff[mic_id]["label"][chunk:]
        theta = 2 * np.pi * direction[0] / 72
        r = np.linspace(0, 20, 10)
        x = self.mic_loc[mic_id][0] + r * np.cos(theta)
        y = self.mic_loc[mic_id][1] + r * np.sin(theta)
        self.win.plot_data[mic_id]["x"] = x
        self.win.plot_data[mic_id]["y"] = y
        self.win.label_data[mic_id] = self.label_list[label[0]]


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

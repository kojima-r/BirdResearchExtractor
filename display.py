import sys
from udp import UdpReceiver
import threading
import time
import numpy as np
import configparser
import csv


# コンフィグ
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
SEND_CHUNK = config["SEND"].getint("CHUNK")   # sendの処理単位(sample)
DISP_CHUNK = config["DISPLAY"].getint("CHUNK")  # 表示の処理単位(sample)


class Display():
    def __init__(self, receiver:UdpReceiver):
        self.receiver = receiver
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

    def start(self):
        print("Starting display......")
        discriminating_thread = threading.Thread(target=self.__process)
        discriminating_thread.setDaemon(True)
        discriminating_thread.start()
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

    def __display(self, mic_id, chunk):
        direction = self.display_buff[mic_id]["direction"][:chunk]
        self.display_buff[mic_id]["direction"] = self.display_buff[mic_id]["direction"][chunk:]
        audio_id = self.display_buff[mic_id]["audio_id"][:chunk]
        self.display_buff[mic_id]["audio_id"] = self.display_buff[mic_id]["audio_id"][chunk:]
        label = self.display_buff[mic_id]["label"][:chunk]
        self.display_buff[mic_id]["label"] = self.display_buff[mic_id]["label"][chunk:]
        print(f"mic_id: {mic_id}, audio_id: {audio_id[0]}, direction: {direction[0]}, label: {self.label_list[label[0]]}")


def main():
    # Udp受信側の設定
    receiver_from_discriminator = UdpReceiver(
        src_ip="127.0.0.1",
        src_port=30000
    )
    # 表示器の設定
    display = Display(
        receiver=receiver_from_discriminator
    )
    # 表示器の開始
    display.start()

    while True:
        try:
            a = display.display_buff[0]["direction"].shape
            b = display.display_buff[1]["direction"].shape
            print(f"display_buff: {a} {b}")
            time.sleep(1)
        except KeyboardInterrupt:
            print("Key interrupted")
            receiver_from_discriminator.close()
            break


if __name__ == "__main__":
    main()

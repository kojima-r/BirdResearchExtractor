import numpy as np
import socket
import threading
import time
import queue
import numpy as np
import torch
import torch.nn.functional as F
import csv
from model import Transfer_Cnn14
import configparser
import pickle

class UdpReceiver():
    def __init__(self, receive_queue):
        print("Initializing receiver...")
        src_ip = "127.0.0.1"
        src_port = 22222
        self.src_addr = (src_ip, src_port)

        self.SIZE = 32768
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.src_addr)

        self.receive_queue = receive_queue
        thread = threading.Thread(target=self.recv)
        thread.setDaemon(True)
        thread.start()
        print("Initialized receiver!")
        print("listening...")

    def recv(self):
        while True:
            data, addr = self.sock.recvfrom(self.SIZE)
            print(pickle.loads(data)["packet_id"])
            time.sleep(0.05)


if __name__ == "__main__":
    # UDP受信側の初期設定
    receive_queue = queue.Queue()
    receiver = UdpReceiver(receive_queue)
    while True:
        pass
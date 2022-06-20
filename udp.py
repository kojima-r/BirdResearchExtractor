import socket
import time
import threading
import queue
from typing import NoReturn
import numpy as np


DEBUG = 0


class UdpSender():
    def __init__(self, src_ip:str, src_port:int, dst_ip:str, dst_port:int):
        """UdpSenderの初期化

        Args:
            src_ip (str): 送信元（自身）のipアドレス
            src_port (int): 送信元（自身）のポート番号
            dst_ip (str): 送信先のipアドレス
            dst_port (int): 送信先のポート番号
        """
        self.src_addr = (src_ip, src_port)
        self.dst_addr = (dst_ip, dst_port)
        self.send_queue = queue.Queue()
        self.open()
    
    def open(self):
        """udpソケットを開く
        """
        print("Opening udp sender......")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.src_addr)
        thread = threading.Thread(target=self.process_send_queue)
        thread.setDaemon(True)
        thread.start()
        print(f"src_ip: {self.src_addr[0]}, src_port: {self.src_addr[1]}")
        print(f"dst_ip: {self.dst_addr[0]}, dst_port: {self.dst_addr[1]}")
        print("Opened udp sender!")

    def process_send_queue(self):
        """送信キューにデータがあれば送信する
        """
        while True:
            if not self.send_queue.empty():
                data = self.send_queue.get()
                self.send(data)
            time.sleep(0.01)

    def send(self, data:bytes) -> NoReturn:
        """データを送信する

        Args:
            data (bytes): 送信データ
        
        Returns:
            NoReturns
        """
        self.sock.sendto(data, self.dst_addr)
        if DEBUG:
            print(f"send_data: {len(data)}")

    def close(self):
        """udpソケットを閉じる
        """
        print("Closing udp sender......")
        self.sock.close()
        print(f"src_ip: {self.src_addr[0]}, src_port: {self.src_addr[1]}")
        print(f"dst_ip: {self.dst_addr[0]}, dst_port: {self.dst_addr[1]}")
        print("Closed udp sender!")


class UdpReceiver():
    def __init__(self, src_ip:str, src_port:int):
        """UdpReceiverの初期化

        Args:
            src_ip (str): 受信元（自身）のipアドレス
            src_port (int): 受信元（自身）のポート番号
        """
        self.src_ip = src_ip
        self.src_port = src_port
        self.src_addr = (self.src_ip, self.src_port)
        self.SIZE = 65536
        self.receive_queue = queue.Queue()
        self.open()

    def open(self):
        """udpソケットを開く
        """
        print("Opening udp receiver......")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.src_addr)
        thread = threading.Thread(target=self.receive)
        thread.setDaemon(True)
        thread.start()
        print(f"src_ip: {self.src_addr[0]}, src_port: {self.src_addr[1]}")
        print("Opened udp receiver!")

    def receive(self):
        """データを受け取り受信キューに追加
        """
        while True:
            data, addr = self.sock.recvfrom(self.SIZE)
            if DEBUG:
                print(f"receive_data: {len(data)}")
            self.receive_queue.put(data)
            time.sleep(0.01)

    def close(self):
        """udpソケットを閉じる
        """
        print("Closing udp receiver......")
        self.sock.close()
        print(f"src_ip: {self.src_addr[0]}, src_port: {self.src_addr[1]}")
        print("Closed udp receiver!")
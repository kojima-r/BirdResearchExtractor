import socket
import time
import threading
import queue
import numpy as np


class UdpSender():
    def __init__(self, src_ip, src_port, dst_ip, dst_port):
        print("Initializing udp sender......")
        self.src_addr = (src_ip, src_port)
        self.dst_addr = (dst_ip, dst_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.src_addr)
        self.send_queue = queue.Queue()
        thread = threading.Thread(target=self.send)
        thread.setDaemon(True)
        thread.start()
        print("Initialized udp sender!")

    def send(self):
        while True:
            if not self.send_queue.empty():
                data = self.send_queue.get()
                self.sock.sendto(data, self.dst_addr)
                time.sleep(0.01)
            else:
                time.sleep(0.01)


class UdpReceiver():
    def __init__(self, src_ip, src_port):
        print("Initializing udp receiver...")
        self.src_addr = (src_ip, src_port)
        self.SIZE = 16384
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.src_addr)
        self.receive_queue = queue.Queue()
        thread = threading.Thread(target=self.recv)
        thread.setDaemon(True)
        thread.start()
        print("Initialized udp receiver!")

    def recv(self):
        while True:
            data, addr = self.sock.recvfrom(self.SIZE)
            self.receive_queue.put(data)
            time.sleep(0.01)

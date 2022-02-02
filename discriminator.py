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

class UdpReceiver():
    def __init__(self, receive_queue):
        src_ip = "127.0.0.1"
        src_port = 22222
        self.src_addr = (src_ip, src_port)

        self.SIZE = 8192
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.src_addr)

        self.receive_queue = receive_queue
        thread = threading.Thread(target=self.recv)
        thread.setDaemon(True)
        thread.start()
        print("Initialized receiver")

    def recv(self):
        while True:
            data, addr = self.sock.recvfrom(self.SIZE)
            self.receive_queue.put(np.frombuffer(data))
            time.sleep(0.05)


def pred(data):
    classes_num=291
    sample_rate=16000
    window_size=512
    hop_size=160
    mel_bins=64
    fmin=50
    fmax=8000
    model_path="best_models/best_model.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # loading model
    model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # makeing input tensor 
    input_tensor = torch.from_numpy(data)
    input_tensor = input_tensor.to(torch.float32)
    input_tensor = input_tensor.to(device)

    # prediction
    with torch.no_grad():   
        pred_y = model(input_tensor)
        idx = pred_y[0].argmax()
        label = label_list[idx]

    return label


config = configparser.ConfigParser()
config.read("discriminator_config.ini", encoding="utf-8")
dc_config = config["DISCRIMINATOR"]
CHUNK = dc_config.getint("CHUNK")
STEP = dc_config.getint("STEP")


if __name__ == "__main__":
    receive_queue = queue.Queue()
    audio_buff = np.empty(0)
    receiver = UdpReceiver(receive_queue)

    label_list = []
    with open("label01_mapping.tsv", "r") as f:
        tsv = csv.reader(f, delimiter="\t")
        for row in tsv:
            label_list.append(row[1])

    while True:
        try:
            # UDP通信を経由してオーディオデータを取得
            while not receive_queue.empty():
                audio = receive_queue.get()
                audio_buff = np.concatenate([audio_buff, audio])
            
            # 識別器を用いてラベル予測
            if audio_buff.shape[0] > CHUNK:
                audio = audio_buff[:CHUNK]
                audio_buff = audio_buff[STEP:]
                label = pred(audio[None, :])
                print(label)

        except KeyboardInterrupt:
            print("Key interrupted")
            break
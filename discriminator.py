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


def pred(data, model):
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


# 環境変数設定
config = configparser.ConfigParser()
config.read("discriminator_config.ini", encoding="utf-8")
dc_config = config["DISCRIMINATOR"]
CHUNK = dc_config.getint("CHUNK")  # 音声識別の長さ単位
STEP = dc_config.getint("STEP")  # 音声識別のステップ
label_list = [] # クラス名リスト
with open("label01_mapping.tsv", "r") as f:
    tsv = csv.reader(f, delimiter="\t")
    for row in tsv:
        label_list.append(row[1])

if __name__ == "__main__":
    # モデルの初期設定、ロード
    print("Loading a model...")
    classes_num=291
    sample_rate=16000
    window_size=512
    hop_size=160
    mel_bins=64
    fmin=50
    fmax=8000
    model_path="best_models/best_model.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    print("Loaded a model!")

    # UDP受信側の初期設定
    receive_queue = queue.Queue()
    audio_buff = np.empty(0)
    receiver = UdpReceiver(receive_queue)

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
                label = pred(audio[None, :], model)
                print(label)

        except KeyboardInterrupt:
            print("Key interrupted")
            break
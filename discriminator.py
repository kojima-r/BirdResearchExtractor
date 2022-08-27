import sys
from udp import UdpReceiver, UdpSender
import pickle
import time
import threading
import numpy as np
import configparser
from hark_tf.read_mat import read_hark_tf
import torch
import torch.nn.functional as F
import csv
from model import Transfer_Cnn14

DEBUG = 1

# コンフィグ
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
TF_CONFIG = read_hark_tf("tamago_rectf.zip")  # マイクの伝達関数など
MIC_CHANNELS = config["MIC"].getint("CHANNELS")  # チャンネル数
MIC_BIT = config["MIC"].getint("BIT")  # ビット数
MIC_SAMPLE_RATE = config["MIC"].getint("SAMPLE_RATE")  # サンプルレート
STREAM_CHUNK = config["STREAM"].getint("CHUNK")  # ストリーミングからの読み込み単位
STFT_CHUNK = config["STFT"].getint("CHUNK")  # stftの処理単位(sample)
STFT_WIN = config["STFT"].getint("WIN")  # stftの窓幅
STFT_STEP = config["STFT"].getint("STEP")  # stftのステップ幅
MUSIC_CHUNK = config["MUSIC"].getint("CHUNK")  # music法の処理単位(frame)
MUSIC_WIN = config["MUSIC"].getint("WIN")  # music法の窓幅
MUSIC_STEP = config["MUSIC"].getint("STEP")  # music法のステップ幅
ISTFT_CHUNK = config["ISTFT"].getint("CHUNK")  # istftの処理単位(frame)
SEND_CHUNK = config["SEND"].getint("CHUNK")   # sendの処理単位(sample)
DISC_MIN_CHUNK = config["DISCRIMINATOR"].getint("MIN_CHUNK")  # 識別に必要な最小長さ(sample)
DISC_MAX_CHUNK = config["DISCRIMINATOR"].getint("MAX_CHUNK")  # 識別に必要な最大長さ(sample)
MODEL_CLASSES_NUM = config["MODEL"].getint("CLASSES_NUM")
MODEL_WIN = config["MODEL"].getint("WIN")
MODEL_HOP = config["MODEL"].getint("HOP")
MODEL_MEL_BINS = config["MODEL"].getint("MEL_BINS")
MODEL_FMIN = config["MODEL"].getint("FMIN")
MODEL_FMAX = config["MODEL"].getint("FMAX")


class Discriminator():
    def __init__(self, receiver: UdpReceiver, sender: UdpSender):
        self.receiver = receiver
        self.sender = sender
        audio_buff_unit = {
            "audio": np.empty(0, dtype=np.float),
            "direction": np.empty(0, dtype=np.int),
            "audio_id": np.empty(0, dtype=np.int)
        }
        self.audio_buff = [audio_buff_unit.copy(), audio_buff_unit.copy()]
        self.now_audio = [audio_buff_unit.copy(), audio_buff_unit.copy()]
        tagged_audio_buff_unit = {
            "direction": np.empty(0, dtype=np.int),
            "audio_id": np.empty(0, dtype=np.int),
            "pred_idx": np.empty(0, dtype=np.int),
            "pred_prob": np.empty(0, dtype=np.float)
        }
        self.tagged_audio_buff = [tagged_audio_buff_unit.copy(), tagged_audio_buff_unit.copy()]

        # モデルのロード
        print("Loading a model...")
        self.classes_num = MODEL_CLASSES_NUM
        self.sample_rate = MIC_SAMPLE_RATE
        self.window_size = MODEL_WIN
        self.hop_size = MODEL_HOP
        self.mel_bins = MODEL_MEL_BINS
        self.fmin = MODEL_FMIN
        self.fmax = MODEL_FMAX
        self.model_path = "best_models/best_model.pth"
        
        self.label_list = []
        with open("label01_mapping.tsv", "r") as f:
            tsv = csv.reader(f, delimiter="\t")
            for row in tsv:
                self.label_list.append(row[1])
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = Transfer_Cnn14(self.sample_rate, self.window_size, self.hop_size, self.mel_bins, self.fmin, self.fmax, self.classes_num, freeze_base=False)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()
        print("Loaded a model!")

    def start(self):
        print("Starting discriminator......")
        discriminating_thread = threading.Thread(target=self.__process)
        discriminating_thread.setDaemon(True)
        discriminating_thread.start()
        print("Started discriminator!")

    def __process(self):
        while True:
            while not self.receiver.receive_queue.empty():
                self.__get_audio_from_queue(
                    chunk=SEND_CHUNK
                )
            for mic_id in range(2):
                if self.audio_buff[mic_id]["audio"].shape[0] > 0:
                    self.__discriminate(
                        mic_id=mic_id,
                        min_chunk=DISC_MIN_CHUNK,
                        max_chunk=DISC_MAX_CHUNK
                    )
                if self.tagged_audio_buff[mic_id]["audio_id"].shape[0] >= SEND_CHUNK:
                    self.__put_to_send_queue(
                        mic_id=mic_id,
                        chunk=SEND_CHUNK
                    )

    def __get_audio_from_queue(self, chunk):
        data_bin = self.receiver.receive_queue.get()
        data = np.frombuffer(data_bin, dtype="float64")
        audio = data[:chunk]
        direction = data[chunk:chunk*2].astype(np.int)
        audio_id = data[chunk*2:chunk*3].astype(np.int)
        mic_id = data[-1].astype(np.int)
        self.audio_buff[mic_id]["audio"] = np.concatenate([self.audio_buff[mic_id]["audio"], audio], axis=0)
        self.audio_buff[mic_id]["direction"] = np.concatenate([self.audio_buff[mic_id]["direction"], direction], axis=0)
        self.audio_buff[mic_id]["audio_id"] = np.concatenate([self.audio_buff[mic_id]["audio_id"], audio_id], axis=0)
    
    def __discriminate(self, mic_id, min_chunk, max_chunk):
        # 今の（最初に処理するべき）audio_idのオーディオを取り出す
        now_audio_id = self.audio_buff[mic_id]["audio_id"][0]
        audio = self.audio_buff[mic_id]["audio"][self.audio_buff[mic_id]["audio_id"] == now_audio_id]
        direction = self.audio_buff[mic_id]["direction"][self.audio_buff[mic_id]["audio_id"] == now_audio_id]
        audio_id = self.audio_buff[mic_id]["audio_id"][self.audio_buff[mic_id]["audio_id"] == now_audio_id]
        self.audio_buff[mic_id]["audio"] = self.audio_buff[mic_id]["audio"][self.audio_buff[mic_id]["audio_id"] != now_audio_id]
        self.audio_buff[mic_id]["direction"] = self.audio_buff[mic_id]["direction"][self.audio_buff[mic_id]["audio_id"] != now_audio_id]
        self.audio_buff[mic_id]["audio_id"] = self.audio_buff[mic_id]["audio_id"][self.audio_buff[mic_id]["audio_id"] != now_audio_id]

        # audio_idが同じか空ならnow_audioに追加
        if self.now_audio[mic_id]["audio"].shape[0] == 0:
            self.now_audio[mic_id]["audio"] = np.concatenate([self.now_audio[mic_id]["audio"], audio], axis=0)
            self.now_audio[mic_id]["direction"] = np.concatenate([self.now_audio[mic_id]["direction"], direction], axis=0)
            self.now_audio[mic_id]["audio_id"] = np.concatenate([self.now_audio[mic_id]["audio_id"], audio_id], axis=0)
        elif now_audio_id == self.now_audio[mic_id]["audio_id"][-1]:
            self.now_audio[mic_id]["audio"] = np.concatenate([self.now_audio[mic_id]["audio"], audio], axis=0)
            self.now_audio[mic_id]["direction"] = np.concatenate([self.now_audio[mic_id]["direction"], direction], axis=0)
            self.now_audio[mic_id]["audio_id"] = np.concatenate([self.now_audio[mic_id]["audio_id"], audio_id], axis=0)
        else:
            self.now_audio[mic_id]["audio"] = audio
            self.now_audio[mic_id]["direction"] = direction
            self.now_audio[mic_id]["audio_id"] = audio_id

        # モデルに渡すオーディオの作成・予測
        if self.now_audio[mic_id]["audio"].shape[0] >= min_chunk:
            if self.now_audio[mic_id]["audio"].shape[0] >= max_chunk:
                audio_for_pred = self.now_audio[mic_id]["audio"][-max_chunk:]
            else:
                audio_for_pred = self.now_audio[mic_id]["audio"]
            idx, prob = self.__pred(audio_for_pred[None, :])
        
            # 予測結果をバッファに追加
            now_len = self.now_audio[mic_id]["audio"].shape[0]
            prev_len = now_len - audio.shape[0]
            if prev_len < min_chunk:
                # 前の長さがmin_chunk以下だと前回までの値を予測に使っていないので全て認識に使って結果を追加
                self.tagged_audio_buff[mic_id]["direction"] = np.concatenate([self.tagged_audio_buff[mic_id]["direction"], self.now_audio[mic_id]["direction"]], axis=0)
                self.tagged_audio_buff[mic_id]["audio_id"] = np.concatenate([self.tagged_audio_buff[mic_id]["audio_id"], self.now_audio[mic_id]["audio_id"]], axis=0)
                self.tagged_audio_buff[mic_id]["pred_idx"] = np.concatenate([self.tagged_audio_buff[mic_id]["pred_idx"], np.repeat(idx, now_len)], axis=0)
                self.tagged_audio_buff[mic_id]["pred_prob"] = np.concatenate([self.tagged_audio_buff[mic_id]["pred_prob"], np.repeat(prob, now_len)], axis=0)
            else:
                # 増分だけ追加
                self.tagged_audio_buff[mic_id]["direction"] = np.concatenate([self.tagged_audio_buff[mic_id]["direction"], direction], axis=0)
                self.tagged_audio_buff[mic_id]["audio_id"] = np.concatenate([self.tagged_audio_buff[mic_id]["audio_id"], audio_id], axis=0)
                self.tagged_audio_buff[mic_id]["label"] = np.concatenate([self.tagged_audio_buff[mic_id]["pred_idx"], np.repeat(idx, audio.shape[0])], axis=0)
                self.tagged_audio_buff[mic_id]["pred_prob"] = np.concatenate([self.tagged_audio_buff[mic_id]["pred_prob"], np.repeat(prob, audio.shape[0])], axis=0)

    def __pred(self, audio):
        input_tensor = torch.from_numpy(audio)
        input_tensor = input_tensor.to(torch.float32)
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            pred_y = self.model(input_tensor)
            softmax = torch.nn.Softmax(dim=0)
            idx = pred_y[0].argmax().item()
            prob = softmax(pred_y[0]).max().item()
            if DEBUG:
                print(idx, prob)
        return idx, prob

    def __put_to_send_queue(self, mic_id, chunk):
        direction = self.tagged_audio_buff[mic_id]["direction"][:chunk]
        self.tagged_audio_buff[mic_id]["direction"] = self.tagged_audio_buff[mic_id]["direction"][chunk:]
        audio_id = self.tagged_audio_buff[mic_id]["audio_id"][:chunk]
        self.tagged_audio_buff[mic_id]["audio_id"] = self.tagged_audio_buff[mic_id]["audio_id"][chunk:]
        pred_idx = self.tagged_audio_buff[mic_id]["pred_idx"][:chunk]
        self.tagged_audio_buff[mic_id]["pred_idx"] = self.tagged_audio_buff[mic_id]["pred_idx"][chunk:]
        pred_prob = self.tagged_audio_buff[mic_id]["pred_prob"][:chunk]
        self.tagged_audio_buff[mic_id]["pred_idx"] = self.tagged_audio_buff[mic_id]["pred_prob"][chunk:]
        data = np.concatenate([direction, audio_id, pred_idx, pred_prob, np.array([mic_id])], axis=0).astype(np.float)
        data_bin = data.tobytes()
        self.sender.send_queue.put(data_bin)

def main():
    # Udp受信側の設定
    receiver_from_extractor = UdpReceiver(
        src_ip="127.0.0.1",
        src_port=20000
    )
    # Udp送信側の設定
    sender_to_display = UdpSender(
        src_ip="127.0.0.1",
        src_port=21000,
        dst_ip="127.0.0.1",
        dst_port=30000
    )
    # 識別器の設定
    discriminator = Discriminator(
        receiver=receiver_from_extractor,
        sender=sender_to_display
    )
    # 識別器の開始
    discriminator.start()

    while True:
        try:
            if DEBUG:
                a = discriminator.audio_buff[0]["audio"].shape
                b = discriminator.audio_buff[1]["audio"].shape
                print(f"audio_buff: {a} {b}")
                c = discriminator.tagged_audio_buff[0]["audio_id"].shape
                d = discriminator.tagged_audio_buff[1]["audio_id"].shape
                print(f"tagged_audio_buff: {c} {d}")
                e = receiver_from_extractor.receive_queue.qsize()
                print(f"receive_queue: {e}")
                f = sender_to_display.send_queue.qsize()
                print(f"send_queue: {f}")
            time.sleep(1)
        except KeyboardInterrupt:
            print("Key interrupted")
            receiver_from_extractor.close()
            sender_to_display.close()
            break


if __name__ == "__main__":
    main()

# ライブラリのインポート
import sys
from email.mime import audio
from re import S
import numpy as np
import pyaudio
from hark_tf.read_mat import read_hark_tf
import micarrayx
from gsc import beamforming_ds, beamforming_ds2
from micarrayx.localization.music import compute_music_spec
import queue
import configparser
import pickle
import math
import time
from udp import UdpSender


class AudioStreamer():
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        self.open()

    def __convert_to_np_from_buff(self, in_data):
        """ バッファ形式の音声データをnumpy形式に変換
        Args:
            data (buff): バッファデータ
        Returns:
            ndarray: numpy形式のデータ(チャンネル数, データ長)
        """
        bit_dtype = {
            8: "int8",
            16: "int16",
            24: "int24",
            32: "int32"
        }
        out_data = np.frombuffer(in_data, dtype=bit_dtype[BIT])
        out_data = out_data / (2 ** (BIT - 1))
        out_data = out_data.reshape(-1, CHANNELS).T
        return out_data

    def __callback(self, in_data, frame_count, time_info, status):
        audio = self.__convert_to_np_from_buff(in_data)
        self.audio_queue.put(audio)
        return (in_data, pyaudio.paContinue)

    def open(self):
        print("Initializing audio streaming......")
        p_format = {
            8: pyaudio.paInt8,
            16: pyaudio.paInt16,
            24: pyaudio.paInt24,
            32: pyaudio.paInt32
        }
        self.stream = self.p.open(
            format=p_format[BIT],
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=CHUNK,
            input_device_index=INPUT_DEVICE_INDEX,
            input=True,
            stream_callback=self.__callback
        )
        print("Initialized and started audio streaming!")

    def close(self):
        print("Closing audio streaming......")
        self.stream.stop_stream()
        self.stream.close()
        print("Closed audio streaming!")


class AudioProcessor():
    def __init__(self):
        self.audio_buff = np.empty((CHANNELS, 0))
        self.spec_buff = np.empty((CHANNELS, 0, STFT_LEN//2 + 1))
        self.empha_spec_list = []
        self.least_tagged_empha_spec_list = []
        self.audio_id = 0
        self.time = 0
        self.tagged_empha_spec_list = []
        self.tagged_empha_audio_list = []

    def __stft(self, audio):
        """ numpy形式の音声データをスペクトログラムに変換
        Args:
            data (ndarray): numpy形式の音声データ(チャンネル数, データ長)
        Returns:
            ndarray: スペクトログラム(チャンネル数, データ長, 周波数ビン)
        """
        spec = micarrayx.stft_mch(audio, STFT_WIN, STFT_STEP)
        return spec

    def __music(self, spec):
        """ MUSIC法による音源方向の取得
        Args:
            spec (ndarray): スペクトログラム(チャンネル数, データ長, 周波数ビン)
        Returns:
            int: 音源方向（分解能72）
        """
        power = compute_music_spec(
            spec=spec,
            src_num=3,
            tf_config=TF_CONFIG,
            df=SAMPLE_RATE/STFT_LEN,
            min_freq_bin=0,
            win_size=MUSIC_CHUNK,
            step=MUSIC_STEP
        )
        p = np.sum(np.real(power), axis=1)
        m_power = 10 * np.log10(p + 1.0)
        direction = m_power.argmax(axis=1)[0]
        return direction

    def __bf(self, spec, direction):
        """ 遅延和ビームフォーミングによる音源強調
        Args:
            spec (ndarray): スペクトログラム(チャンネル数, データ長, 周波数ビン)
            direction: 強調する方向(分解能72)
        Returns:
            ndarray: 強調音源のスペクトログラム(データ長, 周波数ビン)
        """
        theta = direction * 360 / 72
        bf = beamforming_ds2(TF_CONFIG, spec, theta=theta)
        return bf

    def __istft(self, tagged_empha_spec):
        spec = tagged_empha_spec["spec"][None, :]
        audio = micarrayx.istft_mch(spec, STFT_WIN, STFT_STEP)[0]
        tagged_empha_audio = tagged_empha_spec.copy()
        del tagged_empha_audio["spec"]
        tagged_empha_audio["audio"] = audio[:-STFT_STEP]
        return tagged_empha_audio

    def __is_same_source(self, empha_spec, least_empha_spec):
        condition1 = (empha_spec["time"] - least_empha_spec["time"]) == 1
        condition2 = abs(empha_spec["direction"] -
                         least_empha_spec["direction"]) % 72 <= 1
        is_same = condition1 and condition2
        return is_same

    def get_audio_from_queue(self, audio_queue):
        audio = audio_queue.get()
        self.audio_buff = np.concatenate([self.audio_buff, audio], axis=1)

    def convert_to_spectrum_from_audio(self):
        slice_audio = self.audio_buff[:, :STFT_CHUNK]
        self.audio_buff = self.audio_buff[:,
                                          STFT_CHUNK - STFT_LEN + STFT_STEP:]
        spec = self.__stft(slice_audio)
        self.spec_buff = np.concatenate([self.spec_buff, spec], axis=1)

    def emphasize_spectrum(self):
        slice_spec = self.spec_buff[:, :MUSIC_CHUNK, :]
        self.spec_buff = self.spec_buff[:, MUSIC_STEP:, :]
        direction = self.__music(slice_spec)
        bf = self.__bf(slice_spec, direction)
        empha_spec = {
            "mic_id": MIC_ID,
            "time": self.time,
            "direction": direction,
            "spec": bf
        }
        self.empha_spec_list.append(empha_spec)
        self.time += 1

    def tagging_spectrum(self):
        empha_spec = self.empha_spec_list[0]
        self.empha_spec_list = self.empha_spec_list[1:]
        tagged_empha_spec = empha_spec
        if len(self.least_tagged_empha_spec_list) == 0:
            tagged_empha_spec["audio_id"] = self.audio_id
            self.tagged_empha_spec_list.append(tagged_empha_spec)
            self.least_tagged_empha_spec_list.append(tagged_empha_spec)
            self.audio_id += 1
        else:
            max_unnecessary_idx = -1
            continuous = False
            for idx in range(len(self.least_tagged_empha_spec_list)):
                least_tagged_empha_spec = self.least_tagged_empha_spec_list[idx]
                if (empha_spec["time"] - least_tagged_empha_spec["time"]) >= 2:
                    max_unnecessary_idx = idx
                if self.__is_same_source(empha_spec, least_tagged_empha_spec):
                    tagged_empha_spec["audio_id"] = least_tagged_empha_spec["audio_id"]
                    continuous = True
                    break
                else:
                    tagged_empha_spec["audio_id"] = self.audio_id
            if not continuous:
                self.audio_id += 1
            self.tagged_empha_spec_list.append(tagged_empha_spec)
            self.least_tagged_empha_spec_list.append(tagged_empha_spec)
            self.least_tagged_empha_spec_list = self.least_tagged_empha_spec_list[
                max_unnecessary_idx + 1:]
        # time = self.tagged_empha_spec_list[-1]["time"]
        # audio_id = self.tagged_empha_spec_list[-1]["audio_id"]
        # direction = self.tagged_empha_spec_list[-1]["direction"]
        # print(f"time: {time}, audio_id: {audio_id}, direction: {direction}")

    def convert_to_audio_from_spectrum(self):
        tagged_empha_spec = self.tagged_empha_spec_list[0]
        self.tagged_empha_spec_list = self.tagged_empha_spec_list[1:]
        tagged_empha_audio = self.__istft(tagged_empha_spec)
        self.tagged_empha_audio_list.append(tagged_empha_audio)
        # time = self.tagged_empha_audio_list[-1]["time"]
        # audio_id = self.tagged_empha_audio_list[-1]["audio_id"]
        # direction = self.tagged_empha_audio_list[-1]["direction"]
        # print(f"time: {time}, audio_id: {audio_id}, direction: {direction}")

    def push_audio_to_queue(self, send_queue):
        tagged_empha_audio = self.tagged_empha_audio_list[0]
        self.tagged_empha_audio_list = self.tagged_empha_audio_list[1:]
        div_n = math.ceil(
            tagged_empha_audio["audio"].shape[0] / SEND_CHUNK)  # 分割数
        for i in range(div_n):
            splitted_audio = tagged_empha_audio["audio"][i * SEND_CHUNK:(i + 1) * SEND_CHUNK]
            splitted_data = {
                "audio_id": tagged_empha_audio["audio_id"],
                "packet_id": i,
                "mic_id": tagged_empha_audio["mic_id"],
                "time": tagged_empha_audio["time"],
                "direction": tagged_empha_audio["direction"],
                "audio": splitted_audio
            }
            splitted_data_bin = pickle.dumps(splitted_data)
            send_queue.put(splitted_data_bin)


def get_device_index(keyword):
    """ 音声デバイスのインデックスを取得
    Args:
        keyword (str): キーワード
    Returns:
        int or None: キーワードを含む音声デバイスのインデックス
    """
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        if keyword in audio.get_device_info_by_index(i)["name"]:
            return i
    return None


# コンフィグ
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
MIC_ID = int(sys.argv[1])  # マイクid
INPUT_DEVICE_INDEX = int(sys.argv[2])  # デバイス番号
PORT = int(sys.argv[3])
TF_CONFIG = read_hark_tf("tamago_rectf.zip")  # マイクの伝達関数など
CHANNELS = config["MIC"].getint("CHANNELS")  # チャンネル数
BIT = config["MIC"].getint("BIT")  # ビット数
SAMPLE_RATE = config["MIC"].getint("SAMPLE_RATE")  # サンプルレート
CHUNK = config["STREAM"].getint("CHUNK")  # ストリーミングからの読み込み単位
STFT_CHUNK = config["STFT"].getint("CHUNK")  # stftの処理単位(sample)
STFT_LEN = config["STFT"].getint("LEN")  # stftの窓幅
STFT_STEP = config["STFT"].getint("STEP")  # stftのステップ幅
STFT_WIN = np.hamming(STFT_LEN)  # stftの窓関数
MUSIC_CHUNK = config["MUSIC"].getint("CHUNK")  # music法の処理単位(frame)
MUSIC_STEP = config["MUSIC"].getint("STEP")  # music法のステップ幅
ISTFT_CHUNK = config["ISTFT"].getint("CHUNK")  # istftの処理単位(frame)
SEND_CHUNK = config["SEND"].getint("CHUNK")   # sendの処理単位(sample)


def main():
    # Udp送信側の設定
    sender = UdpSender(
        src_ip="127.0.0.1",
        src_port=PORT,
        dst_ip="127.0.0.1",
        dst_port=22222,
    )
    # オーディオストリーミングの設定
    streamer = AudioStreamer()
    # オーディオプロセッサの設定
    processor = AudioProcessor()

    while streamer.stream.is_active():
        try:
            # ストリーミングから音声取得
            if not streamer.audio_queue.empty():
                processor.get_audio_from_queue(streamer.audio_queue)

            # 短時間フーリエ変換で周波数領域へと変換
            if processor.audio_buff.shape[1] > STFT_CHUNK:
                processor.convert_to_spectrum_from_audio()

            # MUSIC法とBeamForming法で音源の強調
            if processor.spec_buff.shape[1] > MUSIC_CHUNK:
                processor.emphasize_spectrum()

            # 音源別にid付けをする
            if len(processor.empha_spec_list) > 0:
                processor.tagging_spectrum()

            # 逆短時間フーリエ変換で時間領域へと変換
            if len(processor.tagged_empha_spec_list) > 0:
                processor.convert_to_audio_from_spectrum()

            # id付けした音声を送信キューへ追加
            if len(processor.tagged_empha_audio_list) > 0:
                processor.push_audio_to_queue(sender.send_queue)

        except KeyboardInterrupt:
            print("Key interrupted")
            streamer.close()
            break


if __name__ == "__main__":
    main()

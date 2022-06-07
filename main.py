# ライブラリのインポート
from email.mime import audio
from re import S
import numpy as np
import pyaudio
from hark_tf.read_mat import read_hark_tf
import micarrayx
from gsc import beamforming_ds, beamforming_ds2
from micarrayx.localization.music import compute_music_spec
import queue
import time
import threading
import configparser
import socket
import pickle
import math


class UdpSender():
    def __init__(self):
        print("Initializing sender......")
        src_ip = "127.0.0.1"
        src_port = 11111
        self.src_addr = (src_ip, src_port)

        dst_ip = "127.0.0.1"
        dst_port = 22222
        self.dst_addr = (dst_ip, dst_port)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.src_addr)

        self.send_queue = queue.Queue()
        thread = threading.Thread(target=self.send)
        thread.setDaemon(True)
        thread.start()
        print("Initialized sender!")

    def send(self):
        while True:
            if not self.send_queue.empty():
                data = self.send_queue.get()
                self.sock.sendto(data, self.dst_addr)
            time.sleep(0.01)


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
        self.least_empha_spec = {}
        self.audio_id = 0
        self.t = 0
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
        tagged_empha_audio = {
            "mic_id": tagged_empha_spec["mic_id"],
            "time": tagged_empha_spec["time"],
            "direction": tagged_empha_spec["direction"],
            "id": tagged_empha_spec["id"],
            "audio": audio[:-STFT_STEP]
        }
        return tagged_empha_audio

    def __is_same_source(self, empha_spec, least_empha_spec):
        is_same = abs(empha_spec["direction"][0] -
                      least_empha_spec["direction"][-1]) % 72 <= 2
        return is_same
    
    def __convert_to_buff_from_np(self, in_data):
        """ numpy形式の音声データをバッファ形式に変換
        Args:
            data (ndarray): numpy形式のデータ(チャンネル数, データ長)
        Returns:
            buff: バッファ形式のデータ
        """
        np_dtype = {
            8: np.int8,
            16: np.int16,
            24: np.int32,
            32: np.int32
        }
        out_data = np.ravel(in_data.T)
        out_data = out_data * (2 ** (BIT - 1))
        out_data = out_data.astype(np_dtype[BIT])
        out_data = out_data.tobytes()
        return out_data

    def get_audio_from_queue(self, audio_queue):
        audio = audio_queue.get()
        self.audio_buff = np.concatenate([self.audio_buff, audio], axis=1)

    def convert_to_spectrum_from_audio(self):
        slice_audio = self.audio_buff[:, :STFT_CHUNK]
        self.audio_buff = self.audio_buff[:,
                                          STFT_CHUNK - STFT_LEN + STFT_STEP:]
        spec = self.__stft(slice_audio)
        self.spec_buff = np.concatenate([self.spec_buff, spec], axis=1)

    def emphasize_audio_source(self):
        slice_spec = self.spec_buff[:, :MUSIC_CHUNK, :]
        self.spec_buff = self.spec_buff[:, MUSIC_STEP:, :]
        direction = self.__music(slice_spec)
        bf = self.__bf(slice_spec, direction)
        empha_spec = {
            "mic_id": 0,
            "time": np.array([self.t]),
            "direction": np.array([direction]),
            "spec": bf
        }
        self.empha_spec_list.append(empha_spec)
        self.t += 1

    def tagging_audio_source(self):
        empha_spec = self.empha_spec_list[0]
        self.empha_spec_list = self.empha_spec_list[1:]
        if self.least_empha_spec == {}:
            self.least_empha_spec = empha_spec
        else:
            if self.__is_same_source(empha_spec, self.least_empha_spec):
                combined_time = np.concatenate(
                    [self.least_empha_spec["time"], empha_spec["time"]], axis=0
                )
                combined_direction = np.concatenate(
                    [self.least_empha_spec["direction"], empha_spec["direction"]], axis=0)
                combined_spec = np.concatenate(
                    [self.least_empha_spec["spec"], empha_spec["spec"]], axis=0)
                self.least_empha_spec = {
                    "mic_id": 0,
                    "time": combined_time,
                    "direction": combined_direction,
                    "spec": combined_spec
                }
            else:
                self.least_empha_spec["id"] = self.audio_id
                self.tagged_empha_spec_list.append(self.least_empha_spec)
                self.least_empha_spec = empha_spec
                self.audio_id += 1

    def convert_to_audio_from_spectrum(self):
        tagged_empha_spec = self.tagged_empha_spec_list[0]
        self.tagged_empha_spec_list = self.tagged_empha_spec_list[1:]
        tagged_empha_audio = self.__istft(tagged_empha_spec)
        self.tagged_empha_audio_list.append(tagged_empha_audio)

    def push_audio_to_queue(self, send_queue):
        tagged_empha_audio = self.tagged_empha_audio_list[0]
        self.tagged_empha_audio_list = self.tagged_empha_audio_list[1:]
        a = tagged_empha_audio["audio"]
        # = STFT_STEP * MUSIC_CHUNK
        div_l = 1024  # audioを送信用に分割する
        div_n = math.ceil(tagged_empha_audio["audio"].shape[0] / div_l)  # 分割数
        for i in range(div_n):
            splitted_audio = tagged_empha_audio["audio"][i * div_l:(i + 1) * div_l]
            mic_id = tagged_empha_audio["mic_id"]
            direction = tagged_empha_audio["direction"][div_l * i // (STFT_STEP * MUSIC_CHUNK)]
            audio_id = tagged_empha_audio["id"] % (2 ** 8)
            packet_id = i
            time = tagged_empha_audio["time"][div_l * i // (STFT_STEP * MUSIC_CHUNK)]
            splitted_data = {
                "audio_id": audio_id,
                "packet_id": packet_id,
                "mic_id": mic_id,
                "time": time,
                "direction": direction,
                "audio": splitted_audio
            }
            splitted_audio_bin = pickle.dumps(splitted_data)
            send_queue.put(splitted_audio_bin)


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


def main():
    # Udp送信側の設定
    sender = UdpSender()
    # オーディオストリーミングの設定
    streamer = AudioStreamer()
    # オーディオプロセッサの設定
    processor = AudioProcessor()

    while streamer.stream.is_active():
        try:
            # ストリーミングから音声取得
            while not streamer.audio_queue.empty():
                processor.get_audio_from_queue(streamer.audio_queue)

            # 短時間フーリエ変換で周波数領域へと変換
            if processor.audio_buff.shape[1] > STFT_CHUNK:
                processor.convert_to_spectrum_from_audio()

            # MUSIC法とBeamForming法で音源の強調
            if processor.spec_buff.shape[1] > MUSIC_CHUNK:
                processor.emphasize_audio_source()

            # 音源別にid付けをする
            if len(processor.empha_spec_list) > 0:
                processor.tagging_audio_source()

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


# コンフィグ
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
INPUT_DEVICE_INDEX = get_device_index("TAMAGO")
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


if __name__ == "__main__":
    main()

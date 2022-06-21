# ライブラリのインポート
import sys
from re import S
import numpy as np
import pyaudio
from hark_tf.read_mat import read_hark_tf
import micarrayx as mx
from gsc import beamforming_ds, beamforming_ds2
from micarrayx.localization.music import compute_music_spec
import queue
import configparser
import pickle
import math
import time
import threading
from udp import UdpSender


# コンフィグ
MIC_ID = int(sys.argv[1])  # マイクid
PORT_DISC = 10000 + 100 * MIC_ID
PORT_DISP = 11000 + 100 * MIC_ID
if MIC_ID == 0:
    MIC_DEVICE_INDEX = 24
elif MIC_ID == 1:
    MIC_DEVICE_INDEX = 25
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


class ArrayMic():
    def __init__(self, mic_id: int, device_index: int, channels: int, bit: int, sample_rate: int, tf_config: dict):
        self.mic_id = mic_id
        self.device_index = device_index
        self.channels = channels
        self.bit = bit
        self.sample_rate = sample_rate
        self.tf_config = tf_config


class AudioStreamer():
    def __init__(self, mic: ArrayMic):
        self.mic = mic
        self.audio_buff = np.empty((self.mic.channels, 0))
        self.p = pyaudio.PyAudio()

    def __convert_to_np_from_buff(self, in_data: bytes) -> np.ndarray:
        """ バッファ形式の音声データをnumpy形式に変換

        Args:
            in_data (bytes): バッファデータ

        Returns:
            out_data (ndarray): (チャンネル数, データ長)
        """
        bit_dtype = {
            8: "int8",
            16: "int16",
            24: "int24",
            32: "int32"
        }
        out_data = np.frombuffer(in_data, dtype=bit_dtype[self.mic.bit])
        out_data = out_data / (2 ** (self.mic.bit - 1))
        out_data = out_data.reshape(-1, self.mic.channels).T
        return out_data

    def __callback(self, in_data: bytes, frame_count, time_info, status) -> tuple:
        """ オーディオ取得時のコールバック関数

        Args:
            in_data (bytes): オーディオのバッファデータ
            frame_count (int): フレーム数

        Returns:
            out_data ((bytes, int)): タプル
        """
        audio = self.__convert_to_np_from_buff(in_data)
        self.audio_buff = np.concatenate([self.audio_buff, audio], axis=1)
        return (in_data, pyaudio.paContinue)

    def open(self, chunk) -> None:
        """ オーディオストリーミングを開始

        Args:
            chunk (bytes):

        Returns:
            out_data ((bytes, int)): タプル
        """
        print("Opening audio streaming......")
        p_format = {
            8: pyaudio.paInt8,
            16: pyaudio.paInt16,
            24: pyaudio.paInt24,
            32: pyaudio.paInt32
        }
        self.stream = self.p.open(
            format=p_format[self.mic.bit],
            channels=self.mic.channels,
            rate=self.mic.sample_rate,
            frames_per_buffer=chunk,
            input_device_index=self.mic.device_index,
            input=True,
            stream_callback=self.__callback
        )
        print("Opened audio streaming!")

    def close(self) -> None:
        """オーディオストリーミングを終了
        """
        print("Closing audio streaming......")
        self.stream.stop_stream()
        self.stream.close()
        print("Closed audio streaming!")


class AudioProcessor():
    def __init__(self, streamer: AudioStreamer, sender: UdpSender):
        self.streamer = streamer
        self.sender = sender
        self.config = config
        self.spec_buff = np.empty(
            (self.streamer.mic.channels, 0, STFT_WIN//2 + 1))
        self.empha_spec_buff = {
            "spec": np.empty((0, STFT_WIN//2 + 1)),
            "direction": np.empty(0),
            "audio_id": np.empty(0)
        }
        self.empha_audio_buff = {
            "audio": np.empty(0),
            "direction": np.empty(0),
            "audio_id": np.empty(0)
        }
        self.latest_direction = None
        self.next_audio_id = 0

    def start(self):
        print("Stopping processor......")
        processing_thread = threading.Thread(target=self.__process)
        processing_thread.setDaemon(True)
        processing_thread.start()
        print("Started processor!")

    def __process(self):
        while True:
            if self.streamer.audio_buff.shape[1] >= STFT_CHUNK:
                self.__convert_to_spectrum_from_audio(
                    chunk=STFT_CHUNK,
                    win=STFT_WIN,
                    step=STFT_STEP
                )
            if self.spec_buff.shape[1] >= MUSIC_CHUNK:
                self.__emphasize_spectrum(
                    chunk=MUSIC_CHUNK,
                    win=MUSIC_WIN,
                    step=MUSIC_STEP,
                    df=MIC_SAMPLE_RATE/STFT_WIN
                )
            if self.empha_spec_buff["spec"].shape[0] >= ISTFT_CHUNK:
                self.__convert_to_audio_from_spectrum(
                    chunk=ISTFT_CHUNK,
                    stft_win=STFT_WIN,
                    stft_step=STFT_STEP
                )
            if self.empha_audio_buff["audio"].shape[0] >= SEND_CHUNK + STFT_WIN:
                self.__put_to_send_queue(
                    chunk=SEND_CHUNK
                )

    def __convert_to_spectrum_from_audio(self, chunk, win, step):
        slice_audio = self.streamer.audio_buff[:, :chunk]
        self.streamer.audio_buff = self.streamer.audio_buff[:,
                                                            chunk - win + step:]
        spec = mx.stft_mch(slice_audio, np.hanning(win), step)
        self.spec_buff = np.concatenate([self.spec_buff, spec], axis=1)

    def __emphasize_spectrum(self, chunk, win, step, df):
        slice_spec = self.spec_buff[:, :chunk, :]
        slice_spec_for_music = slice_spec[:, :win, :]
        self.spec_buff = self.spec_buff[:, chunk:, :]
        direction = self.__music(slice_spec_for_music, win, step, df)
        bf = self.__bf(slice_spec, direction)
        # bf.shape = (128, 257) = (chunk, stft_win//2+1)
        if self.latest_direction is None:
            self.latest_direction = direction
            audio_id = self.next_audio_id
            self.next_audio_id += 1
        else:
            if abs(direction - self.latest_direction) % 72 <= 3:
                audio_id = self.next_audio_id - 1
            else:
                audio_id = self.next_audio_id
                self.next_audio_id += 1
        self.latest_direction = direction
        self.empha_spec_buff["spec"] = np.concatenate(
            [self.empha_spec_buff["spec"], bf], axis=0)
        self.empha_spec_buff["direction"] = np.concatenate(
            [self.empha_spec_buff["direction"], np.repeat(direction, chunk)], axis=0)
        self.empha_spec_buff["audio_id"] = np.concatenate(
            [self.empha_spec_buff["audio_id"], np.repeat(audio_id, chunk)], axis=0)

    def __music(self, spec, win, step, df):
        """ MUSIC法による音源方向の取得
        Args:
            spec (ndarray): スペクトログラム(チャンネル数, データ長, 周波数ビン)
        Returns:
            int: 音源方向（分解能72）
        """
        power = compute_music_spec(
            spec=spec,
            src_num=3,
            tf_config=self.streamer.mic.tf_config,
            df=df,
            min_freq_bin=0,
            win_size=win,
            step=step
        )
        # power.shape = (1, 257, 72)
        p = np.sum(np.real(power), axis=1)
        # p.shape = (1, 72)
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
        bf = beamforming_ds2(self.streamer.mic.tf_config, spec, theta=theta)
        return bf

    def __convert_to_audio_from_spectrum(self, chunk, stft_win, stft_step):
        spec = self.empha_spec_buff["spec"][:chunk, :]
        self.empha_spec_buff["spec"] = self.empha_spec_buff["spec"][chunk:, :]
        direction = self.empha_spec_buff["direction"][:chunk]
        self.empha_spec_buff["direction"] = self.empha_spec_buff["direction"][chunk:]
        audio_id = self.empha_spec_buff["audio_id"][:chunk]
        self.empha_spec_buff["audio_id"] = self.empha_spec_buff["audio_id"][chunk:]
        audio = self.__istft(spec, stft_win, stft_step)
        if self.empha_audio_buff["audio"].shape[0] >= stft_win - stft_step:
            self.empha_audio_buff["audio"][stft_step -
                                           stft_win:] += audio[:stft_win - stft_step]
            self.empha_audio_buff["audio"] = np.concatenate(
                [self.empha_audio_buff["audio"], audio[stft_win - stft_step:]], axis=0)
        else:
            self.empha_audio_buff["audio"] = np.concatenate(
                [self.empha_audio_buff["audio"], audio], axis=0)
            self.empha_audio_buff["direction"] = np.concatenate(
                [self.empha_audio_buff["direction"], np.repeat(0, stft_win - stft_step)], axis=0)
            self.empha_audio_buff["audio_id"] = np.concatenate(
                [self.empha_audio_buff["audio_id"], np.repeat(0, stft_win - stft_step)], axis=0)
        direction = np.repeat(direction, stft_step)
        self.empha_audio_buff["direction"] = np.concatenate(
            [self.empha_audio_buff["direction"], direction], axis=0)
        audio_id = np.repeat(audio_id, stft_step)
        self.empha_audio_buff["audio_id"] = np.concatenate(
            [self.empha_audio_buff["audio_id"], audio_id], axis=0)

    def __istft(self, spec, stft_win_size, stft_step):
        spec = spec[None, :]
        audio = mx.istft_mch(spec, np.hanning(stft_win_size), stft_step)[0]
        return audio

    def __put_to_send_queue(self, chunk):
        audio = self.empha_audio_buff["audio"][:chunk]
        self.empha_audio_buff["audio"] = self.empha_audio_buff["audio"][chunk:]
        direction = self.empha_audio_buff["direction"][:chunk]
        self.empha_audio_buff["direction"] = self.empha_audio_buff["direction"][chunk:]
        audio_id = self.empha_audio_buff["audio_id"][:chunk]
        self.empha_audio_buff["audio_id"] = self.empha_audio_buff["audio_id"][chunk:]
        # audio *= 2 ** (self.streamer.mic.bit - 1)
        mic_id = np.array([self.streamer.mic.mic_id])
        data = np.concatenate([audio, direction, audio_id, mic_id], axis=0)
        data_bin = data.tobytes()
        self.sender.send_queue.put(data_bin)


def main():
    # Udp送信側の設定
    sender_to_discriminator = UdpSender(
        src_ip="127.0.0.1",
        src_port=PORT_DISC,
        dst_ip="127.0.0.1",
        dst_port=20000
    )
    # アレイマイクの設定
    mic = ArrayMic(
        mic_id=MIC_ID,
        device_index=MIC_DEVICE_INDEX,
        channels=MIC_CHANNELS,
        bit=MIC_BIT,
        sample_rate=MIC_SAMPLE_RATE,
        tf_config=TF_CONFIG
    )
    # オーディオストリーミングの設定
    streamer = AudioStreamer(mic=mic)
    processor = AudioProcessor(
        streamer=streamer,
        sender=sender_to_discriminator
    )
    # オーディオストリーミングを開始
    streamer.open(chunk=STREAM_CHUNK)
    # オーディオプロセッサを開始
    processor.start()

    while True:
        try:
            print(f"audio_buff: {processor.streamer.audio_buff.shape}")
            time.sleep(1)
        except KeyboardInterrupt:
            print("Key interrupted")
            sender_to_discriminator.close()
            streamer.close()
            break


if __name__ == "__main__":
    main()

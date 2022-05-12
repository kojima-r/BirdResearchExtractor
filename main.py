# ライブラリのインポート
from re import S
import numpy as np
from numpy.core.fromnumeric import shape
import pyaudio
import wave
from hark_tf.read_mat import read_hark_tf
import micarrayx
from gsc import beamforming_ds, beamforming_ds2
from micarrayx.localization.music import compute_music_spec
import queue
import time
import threading
import configparser
import socket


class UdpSender():
    def __init__(self, send_queue):
        print("Initializing sender!")
        src_ip = "127.0.0.1"
        src_port = 11111
        self.src_addr = (src_ip, src_port)

        dst_ip = "127.0.0.1"
        dst_port = 22222
        self.dst_addr = (dst_ip, dst_port)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.src_addr)

        self.send_queue = send_queue
        thread = threading.Thread(target=self.send)
        thread.setDaemon(True)
        thread.start()
        print("Initialized sender!")


    def send(self):
        while True:
            if not self.send_queue.empty():
                data = self.send_queue.get()
                self.sock.sendto(data.tobytes(), self.dst_addr)
                # print(f"Sent {data.shape}, audio_buff: {audio_buff.shape}, spec_buff: {spec_buff.shape}")
            time.sleep(0.05)


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


def conv_to_np_from_buff(data):
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
    data = np.frombuffer(data, dtype=bit_dtype[BIT])
    data = data / (2 ** (BIT - 1))
    data = data.reshape(-1, CHANNELS).T
    return data


def conv_to_buff_from_np(data):
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
    data = np.ravel(data.T)
    data = data * (2 ** (BIT - 1))
    data = data.astype(np_dtype[BIT])
    data = data.tobytes()
    return data


def stream_callback(in_data, frame_count, time_info, status):
    global audio_buff
    audio = conv_to_np_from_buff(in_data)
    audio_queue.put(audio)
    return (in_data, pyaudio.paContinue)


def get_spectrum(data):
    """ numpy形式の音声データをバッファ形式に変換
    Args:
        data (ndarray): numpy形式の音声データ(チャンネル数, データ長)
    Returns:
        ndarray: スペクトログラム(チャンネル数, データ長, 周波数ビン)
    """
    spec = micarrayx.stft_mch(data, STFT_WIN, STFT_STEP)
    return spec


def get_music_direction(spec):
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


def get_bf(spec, direction):
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


def rebuild_audio(spec):
    """ スペクトログラムから波形データを復元
    Args:
        spec (ndarray): スペクトログラム(チャンネル数, データ長, 周波数ビン)
    Returns:
        ndarray: 波形データ(1(チャンネル数), データ長)
    """
    audio = micarrayx.istft_mch(spec, STFT_WIN, STFT_STEP)
    return audio


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    INPUT_DEVICE_INDEX = get_device_index("TAMAGO")
    TF_CONFIG = read_hark_tf("tamago_rectf.zip")  # マイクの伝達関数など

    mic_config = config["MIC"]
    CHANNELS = mic_config.getint("CHANNELS")  # チャンネル数
    BIT = mic_config.getint("BIT")  # ビット数
    SAMPLE_RATE = mic_config.getint("SAMPLE_RATE")  # サンプルレート

    stream_config = config["STREAM"]
    CHUNK = stream_config.getint("CHUNK")  # ストリーミングからの読み込み単位

    stft_config = config["STFT"]
    STFT_CHUNK = stft_config.getint("CHUNK")  # stftの処理単位(sample)
    STFT_LEN = stft_config.getint("LEN")  # stftの窓幅
    STFT_STEP = stft_config.getint("STEP")  # stftのステップ幅
    STFT_WIN = np.hamming(STFT_LEN)  # stftの窓関数

    music_config = config["MUSIC"]
    MUSIC_CHUNK = music_config.getint("CHUNK")  # music法の処理単位(frame)
    MUSIC_STEP = music_config.getint("STEP")  # music法のステップ幅

    istft_config = config["ISTFT"]
    ISTFT_CHUNK = istft_config.getint("CHUNK")  # istftの処理単位(frame)

    send_config = config["SEND"]
    SEND_CHUNK = send_config.getint("CHUNK")   # sendの処理単位(sample)

    wav_n = 0  # ファイル番号
    audio_queue = queue.Queue()
    send_queue = queue.Queue()
    audio_buff = np.empty((CHANNELS, 0))
    spec_buff = np.empty((CHANNELS, 0, STFT_LEN//2 + 1))
    empha_buff = np.empty((0, STFT_LEN//2 + 1))
    reaudio_buff = np.empty(0)

    # Udp送信側の設定
    sender = UdpSender(send_queue)

    # オーディオストリーミングの設定
    p = pyaudio.PyAudio()
    p_format = {
        8: pyaudio.paInt8,
        16: pyaudio.paInt16,
        24: pyaudio.paInt24,
        32: pyaudio.paInt32
    }
    stream = p.open(
        format=p_format[BIT],
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        frames_per_buffer=CHUNK,
        input_device_index=INPUT_DEVICE_INDEX,
        input=True,
        stream_callback=stream_callback
    )
    print("Started audio streaming")

    while stream.is_active():
        try:
            # ストリーミングから音声取得
            while not audio_queue.empty():
                audio = audio_queue.get()
                audio_buff = np.concatenate([audio_buff, audio], axis=1)

            # 短時間フーリエ変換で周波数領域へと変換
            if audio_buff.shape[1] > STFT_CHUNK:
                slice_audio = audio_buff[:, :STFT_CHUNK]
                audio_buff = audio_buff[:, STFT_CHUNK - STFT_LEN + STFT_STEP:]
                spec = get_spectrum(slice_audio)
                spec_buff = np.concatenate([spec_buff, spec], axis=1)

            # MUSIC法とBeamForming法で音源の強調
            if spec_buff.shape[1] > MUSIC_CHUNK:
                slice_spec = spec_buff[:, :MUSIC_CHUNK, :]
                spec_buff = spec_buff[:, MUSIC_STEP:, :]
                direction = get_music_direction(slice_spec)
                bf = get_bf(slice_spec, direction)
                empha_buff = np.concatenate([empha_buff, bf])

            # 逆短時間フーリエ変換で時間領域へと変換
            if empha_buff.shape[0] > ISTFT_CHUNK:
                slice_empha = empha_buff[:ISTFT_CHUNK, :]
                empha_buff = empha_buff[ISTFT_CHUNK:, :]
                audio = rebuild_audio(np.array([slice_empha]))
                if reaudio_buff.shape[0] == 0:
                    reaudio_buff = audio[0]
                else:
                    reaudio_buff_head = reaudio_buff[:-STFT_STEP]
                    reaudio_buff_tail = reaudio_buff[-STFT_STEP:]
                    audio_head = audio[0, :STFT_STEP]
                    audio_tail = audio[0, STFT_STEP:]
                    overlap = reaudio_buff_tail + audio_head
                    reaudio_buff = np.concatenate([reaudio_buff_head, overlap, audio_tail])

            # 強調されたオーディオを送信用キューに追加
            if reaudio_buff.shape[0] > SEND_CHUNK * 2:
                slice_reaudio = reaudio_buff[:SEND_CHUNK]
                reaudio_buff = reaudio_buff[SEND_CHUNK:]
                send_queue.put(slice_reaudio)

        except KeyboardInterrupt:
            print("Key interrupted")
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Stoped audio streaming")
            break

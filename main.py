# ライブラリのインポート
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


def get_device_index(keyword):
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        if keyword in audio.get_device_info_by_index(i)["name"]:
            return i
    return None


def conv_to_np_from_buff(data):
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


def save_wav(data, filename="output.wav"):
    data = conv_to_buff_from_np(data)
    with wave.open(filename, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(BIT // 8)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data)
    print(f"saved {filename}")


def get_spectrum(data):
    spec = micarrayx.stft_mch(data, STFT_WIN, STFT_STEP)
    return spec


def get_music_direction(spec):
    power = compute_music_spec(
        spec=spec,
        src_num=1,
        tf_config=TF_CONFIG,
        df=SAMPLE_RATE/STFT_LEN,
        min_freq_bin=1,
        win_size=MUSIC_CHUNK,
        step=MUSIC_STEP
    )
    p = np.sum(np.real(power), axis=1)
    m_power = 10 * np.log10(p + 1.0)
    direction = m_power.argmax(axis=1)
    return direction


def get_bf(spec, direction):
    theta = direction * 360 / 72
    bf = beamforming_ds2(TF_CONFIG, spec, theta=theta)
    return bf


def rebuild_audio(spec):
    audio = micarrayx.istft_mch(spec, STFT_WIN, STFT_STEP)
    return audio


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

reaudio_config = config["REAUDIO"]
REAUDIO_CHUNK = reaudio_config.getint("CHUNK")   # reaudioの処理単位(sample)


if __name__ == "__main__":
    wav_n = 0  # ファイル番号
    audio_queue = queue.Queue()
    audio_buff = np.empty((CHANNELS, 0))
    spec_buff = np.empty((CHANNELS, 0, STFT_LEN//2 + 1))
    empha_buff = np.empty((0, STFT_LEN//2 + 1))
    reaudio_buff = np.empty(0)

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

            # オーディオの保存
            if reaudio_buff.shape[0] > REAUDIO_CHUNK * 2:
                slice_reaudio = reaudio_buff[:REAUDIO_CHUNK]
                reaudio_buff = reaudio_buff[REAUDIO_CHUNK:]
                # save_wav(slice_reaudio, filename=f"wav/output{wav_n}.wav")
                wav_n = wav_n + 1
                # print(f"audio_buff.shape: {audio_buff.shape}")
                # print(f"spec_buff.shape: {spec_buff.shape}")
                # print(f"reaudio_buff.shape: {reaudio_buff.shape}")

        except KeyboardInterrupt:
            print("Key interrupted")
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Stoped audio streaming")
            break

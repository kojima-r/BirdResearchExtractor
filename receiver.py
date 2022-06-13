from udp import UdpReceiver
import time
import pickle

def main():
    receiver = UdpReceiver(
        src_ip="127.0.0.1",
        src_port=22222
    )
    while True:
        if not receiver.receive_queue.empty():
            data_bin = receiver.receive_queue.get()
            data = pickle.loads(data_bin)
            print(data)
        time.sleep(0.01)

if __name__ == "__main__":
    main()
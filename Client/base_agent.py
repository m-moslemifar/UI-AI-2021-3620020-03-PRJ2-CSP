from game_data import GameData
import socket
import pickle
import struct
import abc


class BaseAgent(metaclass=abc.ABCMeta):
    def __init__(self):
        self.HOST = '127.0.0.1'
        self.PORT = 1234
        # Create a socket connection.
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.connect((self.HOST, self.PORT))

    @abc.abstractmethod
    def do_move(self, game_data: GameData):
        pass

    def play(self):
        while True:
            message = recv_msg(self.s)
            if message == "end" or not message:
                self.s.close()
                break
            send_msg(self.s, self.do_move(pickle.loads(message)).encode())


def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

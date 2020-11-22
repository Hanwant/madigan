import zmq
from multiprocessing import Process

def server(port=9000):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PAIR)
    socket.bind(f"tcp://*:{port}")
    print('Running server on port: ', port)
    for i in range(5):
        msg = socket.recv_string()
        print('msg: ', msg)
        socket.send_string(f'got the message - server {port}')

def client(port=9000):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PAIR)
    print(f"connecting to server with port {port}")
    socket.connect(f"tcp://127.0.0.1:{port}")
    for i in range(5):
        print('sending message')
        socket.send(f"hi from client, msg # {i}".encode('utf-8'))
        msg = socket.recv_string()
        print('reply: ', msg)

if __name__ == "__main__":
    server_port = 9000
    Process(target=server, args=(server_port, )).start()
    Process(target=client, args=(server_port, )).start()



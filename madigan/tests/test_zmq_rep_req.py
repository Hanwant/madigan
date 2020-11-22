import zmq
from multiprocessing import Process

def server(port=9000):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print('Running server on port: ', port)
    for i in range(5):
        msg = socket.recv_string()
        print('msg: ', msg)
        socket.send_string(f'got the message - server {port}')

def client(ports=[9000]):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    print(f"connecting to server with ports {ports}")
    for port in ports:
        socket.connect(f"tcp://127.0.0.1:{port}")
    for i in range(20):
        print('sending message')
        socket.send_string(f'hi from client, msg # {i}')
        msg = socket.recv_string()
        print('reply: ', msg)

if __name__ == "__main__":
    server_ports = range(9000, 9004)
    for server_port in server_ports:
        Process(target=server, args=(server_port, )).start()
    Process(target=client, args=(server_ports, )).start()



import zmq
from zmq.eventloop.zmqstream import ZMQStream
from tornado import ioloop
from multiprocessing import Process

def server(port=9000):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PAIR)
    socket.bind(f"tcp://*:{port}")
    stream = ZMQStream(socket)
    print('Running server on port: ', port)
    global num_msgs
    num_msgs = 0
    def on_recv(msg):
        # msg = socket.recv_string()
        print('msg: ', msg[0])
        socket.send_string(f'got the message - server {port}')
        global num_msgs
        num_msgs += 1
        if num_msgs > 4:
            stream.close()
            ioloop.IOLoop.instance().stop()
    stream.on_recv(on_recv)
    ioloop.IOLoop.instance().start()

def client(port=9000):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PAIR)
    print(f"connecting to server with port {port}")
    socket.connect(f"tcp://127.0.0.1:{port}")
    stream = ZMQStream(socket)
    global num_msg
    num_msg = 0
    def on_recv(msg):
        global num_msg
        num_msg += 1
        print('reply: ', msg)
        if num_msg > 4:
            stream.close()
            ioloop.IOLoop.instance().stop()
    for i in range(5):
        print('sending message')
        socket.send(f"hi from client, msg # {i}".encode('utf-8'))
    stream.on_recv(on_recv)
    ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    server_port = 9000
    Process(target=server, args=(server_port, )).start()
    Process(target=client, args=(server_port, )).start()



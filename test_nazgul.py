import sys
import socket
import json
from argparse import ArgumentParser

parser = ArgumentParser(description="For testing.")
parser.add_argument("--port", "-p", type=int, default=12345,
                    help="Port to run the service on.")
parser.add_argument("--ip", "-i", type=str, default="127.0.0.1",
                    help="IP to run the service on")
parser.add_argument("--message", "-m", type=str, help="msg to send")
parser.add_argument("--conf", "-c", type=str, help="conf to send")
args = parser.parse_args()


server = (args.ip, args.port)
msg = {"src": args.message, "conf": args.conf}

jmsg = bytes(json.dumps(msg), 'ascii')

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect(server)
    sock.sendall(b'HI')
    preresponse = sock.recv(5)
    print(preresponse)
    sock.sendall(jmsg)
    rawresponse = sock.recv(65536)
    print(rawresponse)

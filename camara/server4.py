# importing libraries
import argparse
import pickle
import socket
import struct

import config
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-H", "--host", required=True,
                help="host IP")
ap.add_argument("-p", "--port", required=True,
                help="port")
args = vars(ap.parse_args())

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ip = args["host"]
print('HOST IP:', host_ip)
port = int(args["port"])
socket_address = (host_ip, port)
print('Socket created')
server_socket.bind(socket_address)
print('Socket bind complete')
server_socket.listen(5)
print('Socket now listening')

# Inicializacion de la cámara
camera = config.get_camera()
camera.start()

while True:
    client_socket, addr = server_socket.accept()
    print('Connection from:', addr)
    if client_socket:

        while True:
            # obteniendo la imagen
            # image = camera.read()
            image = camera.get_image()

            a = pickle.dumps(image)
            message = struct.pack("Q", len(a))+a
            client_socket.sendall(message)
            cv2.imshow('Server...', image)
            key = cv2.waitKey(1)
            if key == 27:
                client_socket.close()

server_socket.close()
cv2.destroyAllWindows()

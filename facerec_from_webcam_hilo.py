# importing libraries
import logging
import pickle
import socket
import struct
import threading
import time
from multiprocessing import Semaphore

import cv2

import datos
from datos import imagenes, tracking_general
from functions import resize, unir_rostros_cuerpos
from metodos_deteccion import (
    deteccion_personas_yolo,
    deteccion_personas_yolo_identificacion,
    deteccion_yunet_identificacion_rostros,
)
from metodos_seguimiento import (
    rectangulo_nombre_rostros,
    rectangulos_entrada_salida,
    seguimiento,
    seguimiento_cuerpo_2,
)

semaforo = Semaphore(1)


logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] (%(threadName)-s) %(message)s"
)


def procesamiento(
    frame, coordenadas_local, nombre_camara, camara, net, output_layers, detector_yunet
):
    rostros = []
    cuerpos = []
    deteccion_personas_yolo_identificacion(
        frame,
        cuerpos,
        coordenadas_local,
        nombre_camara,
        net,
        output_layers,
        detector_yunet,
    )

    seguimiento_cuerpo_2(cuerpos, nombre_camara)
    rectangulo_nombre_rostros(coordenadas_local, nombre_camara, frame, camara)
    rectangulos_entrada_salida(camara, frame)


def facerec_from_webcam(local, camara, pos):
    video_capture = cv2.VideoCapture(0)
    coordenadas_local = local["coordenadas"]
    nombre_camara = camara["nombre_camara"]

    net_yolo = cv2.dnn.readNet("modelos/yolov3.weights", "modelos/yolov3.cfg")

    layer_names = net_yolo.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net_yolo.getUnconnectedOutLayers()]

    detector_yunet = cv2.FaceDetectorYN.create(
        "modelos/face_detection_yunet_2022mar.onnx", "", (320, 320)
    )

    while True:
        ret, frame = video_capture.read()

        procesamiento(
            frame,
            coordenadas_local,
            nombre_camara,
            camara,
            net_yolo,
            output_layers,
            detector_yunet,
        )

        semaforo.acquire()
        imagenes[pos] = frame
        semaforo.release()


def facerec_from_video(local, camara, pos, ruta_video):
    video_capture = cv2.VideoCapture(ruta_video)
    coordenadas_local = local["coordenadas"]
    nombre_camara = camara["nombre_camara"]

    net_yolo = cv2.dnn.readNet("modelos/yolov3.weights", "modelos/yolov3.cfg")

    layer_names = net_yolo.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net_yolo.getUnconnectedOutLayers()]

    detector_yunet = cv2.FaceDetectorYN.create(
        "modelos/face_detection_yunet_2022mar.onnx", "", (320, 320)
    )

    while True:
        ret, frame = video_capture.read()

        procesamiento(
            frame,
            coordenadas_local,
            nombre_camara,
            camara,
            net_yolo,
            output_layers,
            detector_yunet,
        )

        semaforo.acquire()
        imagenes[pos] = frame
        semaforo.release()


def facerec_from_socket(host_ip, port, local, camara, pos):
    # Recibir datos desde las camaras con socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host_ip, port))
    data = b""
    payload_size = struct.calcsize("Q")

    coordenadas_local = local["coordenadas"]
    nombre_camara = camara["nombre_camara"]

    net_yolo = cv2.dnn.readNet("modelos/yolov3.weights", "modelos/yolov3.cfg")

    layer_names = net_yolo.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net_yolo.getUnconnectedOutLayers()]

    detector_yunet = cv2.FaceDetectorYN.create(
        "modelos/face_detection_yunet_2022mar.onnx", "", (320, 320)
    )

    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4 * 1024)
            if not packet:
                break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]
        while len(data) < msg_size:
            data += client_socket.recv(4 * 1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        procesamiento(
            frame,
            coordenadas_local,
            nombre_camara,
            camara,
            net_yolo,
            output_layers,
            detector_yunet,
        )

        semaforo.acquire()
        imagenes[pos] = frame
        semaforo.release()

    client_socket.close()


def mostrar_mapa(pos):
    while True:
        img = cv2.imread(datos.mapa)

        i = 0
        for persona in tracking_general:
            cv2.putText(
                img,
                persona["nombre"],
                persona["coordenadas_mapa"],
                datos.font,
                0.75,
                datos.NEGRO,
                1,
            )

            if persona["nombre_camara"] != "NINGUNO":
                semaforo.acquire()
                persona["ttl"] = persona["ttl"] - 1
                semaforo.release()

            x_mapa = persona["coordenadas_local"][0] + 40
            y_mapa = persona["coordenadas_local"][1] + i * 25 + 40

            semaforo.acquire()
            persona["coordenadas_mapa"] = (x_mapa, y_mapa)
            semaforo.release()

            if persona["ttl"] < 0:
                semaforo.acquire()
                tracking_general.remove(persona)
                semaforo.release()

            i = i + 1

        # logging.info(tracking_general)

        semaforo.acquire()
        imagenes[pos] = img
        semaforo.release()


def mostrar_imagenes():
    while True:
        semaforo.acquire()

        imagenes[0] = resize(imagenes[0], height=840, width=597)
        imagenes[1] = resize(imagenes[1], height=280, width=373)
        imagenes[2] = resize(imagenes[2], height=280, width=373)
        imagenes[3] = resize(imagenes[3], height=280, width=373)

        concat_v = cv2.vconcat([imagenes[1], imagenes[2], imagenes[3]])
        concat_h = cv2.hconcat([imagenes[0], concat_v])

        semaforo.release()

        cv2.imshow("Mapa y camaras", concat_h)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break


hilo1 = threading.Thread(
    target=facerec_from_video,
    args=(datos.LOCAL3, datos.CAM1, 1, "hamilton_clip.mp4"),
    name="VIDEO",
)
hilo2 = threading.Thread(
    target=facerec_from_socket,
    args=("10.30.125.149", 10500, datos.LOBBY, datos.CAM1, 2),
    name="CAMARA 1",
)
hilo3 = threading.Thread(
    target=facerec_from_socket,
    args=("10.30.125.150", 10510, datos.AULA_PRE, datos.CAM2, 3),
    name="CAMARA 2",
)
hilo4 = threading.Thread(target=mostrar_mapa, args=(0,), name="PLANO")
hilo5 = threading.Thread(target=mostrar_imagenes, name="VISUALIZACION")

hilo1.start()
hilo2.start()
hilo3.start()
hilo4.start()
hilo5.start()

hilo1.join()
hilo2.join()
hilo3.join()
hilo4.join()
hilo5.join()

cv2.destroyAllWindows()

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
from functions import contenido_en, datos_camara, grid, resize, unir_rostros_cuerpos
from metodos_deteccion import (
    deteccion_personas_yolo_identificacion,
    deteccion_yunet_identificacion_rostros,
)
from metodos_seguimiento import (
    rectangulo_nombre_rostros,
    rectangulos_entrada_salida,
    seguimiento_cuerpo_2,
    transferencia,
)

CANT_FRAMES_1 = 5
CANT_FRAMES_2 = 10

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

    # transferencia(coordenadas_local, nombre_camara, camara)


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

    frame_id = 0
    arreglo_hilos = []

    while True:
        ret, frame = video_capture.read()

        if ret == False:
            break

        frame_id += 1

        if frame_id % CANT_FRAMES_1 == 0:
            hilo_procesamiento = threading.Thread(
                target=procesamiento,
                args=(
                    frame,
                    coordenadas_local,
                    nombre_camara,
                    camara,
                    net_yolo,
                    output_layers,
                    detector_yunet,
                ),
                name="PROCESAMIENTO",
            )

            hilo_procesamiento.start()
            arreglo_hilos.append((hilo_procesamiento, frame_id))

        for hilo, fr_id in arreglo_hilos:
            if frame_id - fr_id > CANT_FRAMES_2:
                hilo.join()
                arreglo_hilos.remove((hilo, fr_id))

        rectangulo_nombre_rostros(coordenadas_local, nombre_camara, frame, camara)
        rectangulos_entrada_salida(camara, frame)

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

    frame_id = 0
    arreglo_hilos = []

    while True:
        ret, frame = video_capture.read()

        if ret == False:
            break

        frame_id += 1

        if frame_id % CANT_FRAMES_1 == 0:
            hilo_procesamiento = threading.Thread(
                target=procesamiento,
                args=(
                    frame,
                    coordenadas_local,
                    nombre_camara,
                    camara,
                    net_yolo,
                    output_layers,
                    detector_yunet,
                ),
                name="PROCESAMIENTO",
            )

            hilo_procesamiento.start()
            arreglo_hilos.append((hilo_procesamiento, frame_id))

        for hilo, fr_id in arreglo_hilos:
            if frame_id - fr_id > CANT_FRAMES_2:
                hilo.join()
                arreglo_hilos.remove((hilo, fr_id))

        rectangulo_nombre_rostros(coordenadas_local, nombre_camara, frame, camara)
        rectangulos_entrada_salida(camara, frame)

        # height, width, _ = frame.shape
        # grid(frame, width, height)

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

    frame_id = 0
    arreglo_hilos = []

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

        frame_id += 1

        if frame_id % CANT_FRAMES_1 == 0:
            hilo_procesamiento = threading.Thread(
                target=procesamiento,
                args=(
                    frame,
                    coordenadas_local,
                    nombre_camara,
                    camara,
                    net_yolo,
                    output_layers,
                    detector_yunet,
                ),
                name="PROCESAMIENTO",
            )

            hilo_procesamiento.start()
            arreglo_hilos.append((hilo_procesamiento, frame_id))

        for hilo, fr_id in arreglo_hilos:
            if frame_id - fr_id > CANT_FRAMES_2:
                hilo.join()
                arreglo_hilos.remove((hilo, fr_id))

        rectangulo_nombre_rostros(coordenadas_local, nombre_camara, frame, camara)
        rectangulos_entrada_salida(camara, frame)

        semaforo.acquire()
        imagenes[pos] = frame
        semaforo.release()

    client_socket.close()


def mostrar_mapa(pos):
    while True:
        img = cv2.imread(datos.mapa)

        x, y, y1 = 0, 0, 0
        for local in datos.LOCALES:
            x = local["coordenadas"][0] + 10
            y = local["coordenadas"][1] + 30
            coord = (x, y)
            cv2.putText(
                img, local["nombre_local"], coord, datos.font, 0.8, datos.ROJO, 1
            )
            y1 = y
            k = 0

            for camara in local["camaras"]:
                nombre = camara["nombre_camara"]
                coord = (x + 10, y + 20 + k * 20)
                cv2.putText(img, nombre, coord, datos.font, 0.8, datos.ROJO, 1)

                y1 = y + 20 + k * 20
                k = k + 1

            i = 0
            for persona in tracking_general:
                if persona["coordenadas_local"] == local["coordenadas"]:
                    x_mapa = x + 30
                    y_mapa = y1 + i * 20 + 20

                    semaforo.acquire()
                    persona["coordenadas_mapa"] = (x_mapa, y_mapa)
                    semaforo.release()

                i = i + 1

        i = 0
        for persona in tracking_general:
            cv2.putText(
                img,
                persona["nombre"] + " " + str(persona["ttl"]),
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

            # if persona["ttl"] == 0:
            #     cam = datos_camara(persona["nombre_camara"])
            #     if cam is not None:
            #         k = 0
            #         for left, top, right, bottom in cam["rectangulos"]:
            #             if contenido_en(
            #                 persona["coordenadas_cuerpo"], (left, top, right, bottom)
            #             ) and (left, top, right, bottom) != (0, 0, 0, 0):
            #                 semaforo.acquire()
            #                 persona["coordenadas_cuerpo"] = cam[
            #                     "rectangulos_relacionados"
            #                 ][k]
            #                 persona["coordenadas_local"] = cam["locales_relacionados"][
            #                     k
            #                 ]
            #                 persona["nombre_camara"] = cam["camaras_relacionadas"][k]
            #                 persona["ttl"] = datos.TTL_MAX
            #                 semaforo.release()

            #                 print("***********************************************")
            #             k = k + 1

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
    args=(datos.AULA, datos.CAM3, 1, "video 2022-09-10 07.29.31.avi"),
    name="CAMARA 3",
)
hilo2 = threading.Thread(
    target=facerec_from_socket,
    args=("10.30.125.149", 10500, datos.LOCAL3, datos.CAM4, 2),
    name="CAMARA 4",
)
hilo3 = threading.Thread(
    target=facerec_from_socket,
    args=("10.30.125.150", 10510, datos.LOCAL2, datos.CAM5, 3),
    name="CAMARA 5",
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

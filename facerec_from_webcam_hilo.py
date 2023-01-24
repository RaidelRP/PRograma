# importing libraries
from multiprocessing import Semaphore
import socket
import cv2
import pickle
import struct

import numpy as np
import cv2

import pickle

import threading
import logging

import face_recognition

from datos import tracking_general, imagenes, known_face_encodings, known_face_names
import datos
from functions import contar_desconocidos, existe_en_tracking, get_iou, resize, historial

semaforo = Semaphore(1)

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] (%(threadName)-s) %(message)s')

def deteccion_cuerpo(frame):
    body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    upper_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    lower_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
    upper = upper_cascade.detectMultiScale(gray, 1.3, 5)
    lower = lower_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in bodies:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # cv2.putText(frame,"body",(x,y+20),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,0,0),1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    for (x, y, w, h) in upper:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.putText(frame,"upper",(x,y+20),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,255,0),1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    for (x, y, w, h) in lower:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        # cv2.putText(frame,"lower",(x,y+20),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,0,255),1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

def deteccion_rostros_haar_cascade(frame):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "rostro haar", (x, y+20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)


def deteccion_identificacion_rostros(frame, coordenadas_local, personas, nombre_camara):
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Persona con nombre, distancia, coordenadas y ttl
        name = "Desconocido"
        p = {'nombre': name, 'distancia': 0.0,
             'coordenadas': (left, top, right, bottom), 'ttl': 0, 'coordenadas_local': coordenadas_local, 'coordenadas_mapa': (0, 0), 'nombre_camara': nombre_camara}

        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)

        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:  # Rostros identificados
            name = known_face_names[best_match_index]
            p["nombre"] = name
            p["distancia"] = face_distances[best_match_index]
            personas.append(p)

        else:  # Rostros desconocidos
            id = contar_desconocidos()
            # myPath = os.path.abspath(os.getcwd())
            myPath = "rostros"
            rostro = frame[top:bottom, left:right]
            name += '_{}'.format(id)
            rostro = cv2.resize(rostro, (150, 150),
                                interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(myPath + '\\' + name + '.jpg', rostro)

            image = face_recognition.load_image_file(
                myPath + '\\' + name + '.jpg')

            if(len(face_recognition.face_encodings(image)) > 0):
                face_encoding = face_recognition.face_encodings(image)[0]

                semaforo.acquire()
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
                semaforo.release()

            cv2.rectangle(frame, (left, top),
                          (right, bottom), datos.ROJO, 2)
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), datos.ROJO, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        datos.font, 1.0, datos.BLANCO, 1)


def seguimiento(personas, nombre_camara):
    for persona_detectada in personas:
        # verifico si no se ha incluido en el tracking para insertarla
        if not existe_en_tracking(persona_detectada["nombre"], tracking_general):
            persona_detectada["ttl"] = 10

            semaforo.acquire()
            tracking_general.append(persona_detectada)
            semaforo.release()

        for persona_seguida in tracking_general:
            # si se detecta procedente de un local sin camara, asignarle la camara actual
            if persona_detectada["nombre"] == persona_seguida["nombre"] and persona_seguida["nombre_camara"] == "NINGUNO":
                semaforo.acquire()
                persona_seguida["nombre_camara"] = nombre_camara
                semaforo.release()

            if get_iou(persona_detectada["coordenadas"], persona_seguida["coordenadas"]) > 0.1:

                semaforo.acquire()
                persona_seguida["coordenadas"] = persona_detectada["coordenadas"]
                semaforo.release()

                if persona_detectada["distancia"] < persona_seguida["distancia"]:

                    semaforo.acquire()

                    persona_seguida["distancia"] = persona_seguida["distancia"]
                    persona_seguida["nombre"] = persona_seguida["nombre"]

                    semaforo.release()

                semaforo.acquire()
                persona_seguida["ttl"] = 10
                semaforo.release()

            historial(persona_seguida["nombre"])


def rectangulo_nombre_rostros(coordenadas_local, nombre_camara, frame, camara):
    for persona in tracking_general:
        # Si se encuentra en el local actual y la camara actual
        if persona['coordenadas_local'] == coordenadas_local and persona["nombre_camara"] == nombre_camara:
            cv2.rectangle(frame, (persona["coordenadas"][0], persona["coordenadas"][1]), (
                persona["coordenadas"][2], persona["coordenadas"][3]), datos.VERDE, 2)

            cv2.rectangle(frame, (persona["coordenadas"][0], persona["coordenadas"][3] - 35),
                          (persona["coordenadas"][2], persona["coordenadas"][3]), datos.VERDE, cv2.FILLED)
            cv2.putText(frame, persona["nombre"], (persona["coordenadas"][0] + 6, persona["coordenadas"][3] - 6),
                        datos.font, 1.0, datos.BLANCO, 1)

            if persona["ttl"] == 0:
                i = 0
                for (top, right, bottom, left) in camara["rectangulos"]:
                    if get_iou(persona["coordenadas"], (top, right, bottom, left)) > 0.1:
                        semaforo.acquire()
                        # persona["coordenadas"] = camara["rectangulos_relacionados"][i]
                        persona['coordenadas_local'] = camara["locales_relacionados"][i]
                        persona['nombre_camara'] = camara["camaras_relacionadas"][i]
                        persona["ttl"] = 10
                        semaforo.release()
                    i = i + 1


def rectangulos_entrada_salida(camara, frame):
    for (left, top, right, bottom) in camara["rectangulos"]:
        cv2.rectangle(frame, (left, top), (right, bottom), datos.MARRON, 2)


def facerec_from_webcam(local, camara, pos):
    video_capture = cv2.VideoCapture(0)
    coordenadas_local = local['coordenadas']
    nombre_camara = camara['nombre_camara']

    while True:
        personas = []
        ret, frame = video_capture.read()
        deteccion_identificacion_rostros(
            frame, coordenadas_local, personas, nombre_camara)
        seguimiento(personas, nombre_camara)
        rectangulo_nombre_rostros(
            coordenadas_local, nombre_camara, frame, camara)
        # rectangulos_entrada_salida(camara, frame)

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
    nombre_camara = camara['nombre_camara']

    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet:
                break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]
        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        personas = []
        deteccion_identificacion_rostros(
            frame, coordenadas_local, personas, nombre_camara)
        seguimiento(personas, nombre_camara)
        rectangulo_nombre_rostros(
            coordenadas_local, nombre_camara, frame, camara)
        # rectangulos_entrada_salida(camara, frame)
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
                img, persona["nombre"], persona["coordenadas_mapa"], datos.font, 0.75, datos.NEGRO, 1)

            if persona["nombre_camara"] != "NINGUNO":
                semaforo.acquire()
                persona["ttl"] = persona["ttl"] - 1
                semaforo.release()

            x_mapa = persona["coordenadas_local"][0] + 40
            y_mapa = persona["coordenadas_local"][1] + i*25 + 40

            semaforo.acquire()
            persona["coordenadas_mapa"] = (x_mapa, y_mapa)
            semaforo.release()

            if persona["ttl"] < 0:
                semaforo.acquire()
                tracking_general.remove(persona)
                semaforo.release()

            i = i + 1

        logging.info(tracking_general)

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

        cv2.imshow('Mapa y camaras', concat_h)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


hilo1 = threading.Thread(target=facerec_from_webcam,
                         args=(datos.LOCAL3, datos.CAM1, 1), name="CAMARA 1")
hilo2 = threading.Thread(target=facerec_from_socket,
                         args=("10.30.125.149", 10500, datos.LOCAL2, datos.CAM2, 2), name="CAMARA 2")
hilo3 = threading.Thread(target=facerec_from_socket,
                         args=("10.30.125.150", 10510, datos.LOBBY, datos.CAM2, 3), name="CAMARA 3")
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

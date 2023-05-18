from multiprocessing import Semaphore

import cv2
import face_recognition
import imutils
import numpy as np

import datos
from functions import contar_desconocidos, contenido_en, get_iou

semaforo = Semaphore(1)


def detecion_cuerpo_hog(frame, coordenadas_local, nombre_camara):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    image = frame
    longitud_inicial = image.shape[1]
    print("Longitud inicial:", longitud_inicial)

    image = imutils.resize(image, width=min(500, image.shape[1]))
    longitud_nueva = image.shape[1]
    print("Longitud nueva:", longitud_nueva)

    ratio = longitud_inicial * 1.0 / longitud_nueva

    (humans, _) = hog.detectMultiScale(
        image, winStride=(5, 5), padding=(3, 3), scale=1.21
    )

    print("Human Detected : ", len(humans))

    for x, y, w, h in humans:
        x1 = int(x * ratio)
        y1 = int(y * ratio)
        w1 = int(w * ratio)
        h1 = int(h * ratio)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "cuerpo hog",
            (x1, y1 + 20),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (0, 0, 255),
            1,
        )


def deteccion_cuerpo(frame):
    body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
    upper_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
    lower_cascade = cv2.CascadeClassifier("haarcascade_lowerbody.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
    upper = upper_cascade.detectMultiScale(gray, 1.3, 5)
    lower = lower_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame, "body", (x, y + 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 1
        )
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

    for x, y, w, h in upper:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, "upper", (x, y + 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1
        )
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

    for x, y, w, h in lower:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            frame, "lower", (x, y + 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1
        )
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]


def deteccion_personas_yolo(
    frame, cuerpos, coordenadas_local, nombre_camara, net, output_layers
):
    height, width, _ = frame.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    for i in indexes:
        box = boxes[i]
        if class_ids[i] == 0:
            x1 = round(box[0])
            y1 = round(box[1])
            x2 = round(box[0] + box[2])
            y2 = round(box[1] + box[3])
            cuerpo = {
                "nombre": "Sin identificar",
                "coordenadas_local": coordenadas_local,
                "nombre_camara": nombre_camara,
                "coordenadas_mapa": (0, 0),
                "ttl": 0,
                "coordenadas_rostro": (0, 0, 0, 0),
                "distancia_rostro": 0.0,
                "confianza_cuerpo": confidences[i],
                "coordenadas_cuerpo": (x1, y1, x2, y2),
            }
            cuerpos.append(cuerpo)


def deteccion_personas_yolo_identificacion(
    frame, cuerpos, coordenadas_local, nombre_camara, net, output_layers, detector_yunet
):
    height, width, _ = frame.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )

    semaforo.acquire()
    net.setInput(blob)
    outs = net.forward(output_layers)
    semaforo.release()

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    for i in indexes:
        box = boxes[i]
        if class_ids[i] == 0:
            x1 = round(box[0])
            y1 = round(box[1])
            x2 = round(box[0] + box[2])
            y2 = round(box[1] + box[3])
            cuerpo = {
                "nombre": "Sin identificar",
                "coordenadas_local": coordenadas_local,
                "nombre_camara": nombre_camara,
                "coordenadas_mapa": (0, 0),
                "ttl": 0,
                "coordenadas_rostro": (0, 0, 0, 0),
                "distancia_rostro": 0.0,
                "confianza_cuerpo": confidences[i],
                "coordenadas_cuerpo": (x1, y1, x2, y2),
            }
            cuerpo_img = frame[y1:y2, x1:x2]
            # cv2.imshow("cuerpo_img",cuerpo_img)
            # cv2.waitKey(0)
            deteccion_yunet_identificacion_rostros_desde_cuerpo(
                frame, cuerpo, detector_yunet
            )
            cuerpos.append(cuerpo)


def deteccion_yunet_identificacion_rostros_desde_cuerpo(frame, cuerpo, detector_yunet):
    rgb_frame = frame[:, :, ::-1]

    img_W = int(frame.shape[1])
    img_H = int(frame.shape[0])

    semaforo.acquire()
    detector_yunet.setInputSize((img_W, img_H))
    detections = detector_yunet.detect(frame)
    semaforo.release()

    face_locations = coordenadas_yunet_a_facerec(detections)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        name = "Desconocido"

        matches = face_recognition.compare_faces(
            datos.known_face_encodings, face_encoding
        )

        face_distances = face_recognition.face_distance(
            datos.known_face_encodings, face_encoding
        )

        best_match_index = np.argmin(face_distances)

        if contenido_en((left, top, right, bottom), cuerpo["coordenadas_cuerpo"]):
            cuerpo["coordenadas_rostro"] = (left, top, right, bottom)           

            if matches[best_match_index]:  # Rostros identificados
                name = datos.known_face_names[best_match_index]
                cuerpo["nombre"] = name
                cuerpo["distancia_rostro"] = face_distances[best_match_index]

            else:  # Rostros desconocidos
                id = contar_desconocidos()
                # myPath = os.path.abspath(os.getcwd())
                myPath = "rostros"
                rostro = frame[top:bottom, left:right]
                name += "_{}".format(id)
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

                # if not(comparar_imagen_con_varias(rostro, myPath+"\*")):
                cv2.imwrite(myPath + "\\" + name + ".jpg", rostro)

                image = face_recognition.load_image_file(myPath + "\\" + name + ".jpg")

                if len(face_recognition.face_encodings(image)) > 0:
                    face_encoding = face_recognition.face_encodings(image)[0]

                    semaforo.acquire()
                    datos.known_face_encodings.append(face_encoding)
                    datos.known_face_names.append(name)
                    semaforo.release()

                cv2.rectangle(frame, (left, top), (right, bottom), datos.ROJO, 2)
                cv2.rectangle(
                    frame, (left, bottom - 35), (right, bottom), datos.ROJO, cv2.FILLED
                )
                cv2.putText(
                    frame,
                    name,
                    (left + 6, bottom - 6),
                    datos.font,
                    1.0,
                    datos.BLANCO,
                    1,
                )


def deteccion_rostros_haar_cascade(frame):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "rostro haar",
            (x, y + 20),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (0, 255, 0),
            1,
        )


def coordenadas_yunet_a_facerec(detections):
    locations = []
    if detections[1] is not None:
        for detection in detections[1]:
            coordenadas_fr = (
                int(detection[1]),
                int(detection[0] + detection[2]),
                int(detection[1] + detection[3]),
                int(detection[0]),
            )
            locations.append(coordenadas_fr)
    return locations


def deteccion_identificacion_rostros(frame, coordenadas_local, rostros, nombre_camara):
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        # Persona con nombre, distancia, coordenadas y ttl
        name = "Desconocido"
        p = {
            "nombre": name,
            "coordenadas_local": coordenadas_local,
            "nombre_camara": nombre_camara,
            "coordenadas_mapa": (0, 0),
            "ttl": 0,
            "coordenadas_rostro": (left, top, right, bottom),
            "distancia_rostro": 0.0,
            "coordenadas_cuerpo": (0, 0, 0, 0),
            "confianza_cuerpo": 0.0,
        }

        matches = face_recognition.compare_faces(
            datos.known_face_encodings, face_encoding
        )

        face_distances = face_recognition.face_distance(
            datos.known_face_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:  # Rostros identificados
            name = datos.known_face_names[best_match_index]
            p["nombre"] = name
            p["distancia_rostro"] = face_distances[best_match_index]
            rostros.append(p)

        else:  # Rostros desconocidos
            id = contar_desconocidos()
            # myPath = os.path.abspath(os.getcwd())
            myPath = "rostros"
            rostro = frame[top:bottom, left:right]
            name += "_{}".format(id)
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            # if not(comparar_imagen_con_varias(rostro, myPath+"\*")):
            cv2.imwrite(myPath + "\\" + name + ".jpg", rostro)

            image = face_recognition.load_image_file(myPath + "\\" + name + ".jpg")

            if len(face_recognition.face_encodings(image)) > 0:
                face_encoding = face_recognition.face_encodings(image)[0]

                semaforo.acquire()
                datos.known_face_encodings.append(face_encoding)
                datos.known_face_names.append(name)
                semaforo.release()

            cv2.rectangle(frame, (left, top), (right, bottom), datos.ROJO, 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), datos.ROJO, cv2.FILLED
            )
            cv2.putText(
                frame, name, (left + 6, bottom - 6), datos.font, 1.0, datos.BLANCO, 1
            )


def deteccion_yunet_identificacion_rostros(
    frame, coordenadas_local, rostros, nombre_camara, detector_yunet
):
    rgb_frame = frame[:, :, ::-1]

    img_W = int(frame.shape[1])
    img_H = int(frame.shape[0])

    detector_yunet.setInputSize((img_W, img_H))
    detections = detector_yunet.detect(frame)

    face_locations = coordenadas_yunet_a_facerec(detections)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        # Persona con nombre, distancia, coordenadas y ttl
        name = "Desconocido"
        p = {
            "nombre": name,
            "coordenadas_local": coordenadas_local,
            "nombre_camara": nombre_camara,
            "coordenadas_mapa": (0, 0),
            "ttl": 0,
            "coordenadas_rostro": (left, top, right, bottom),
            "distancia_rostro": 0.0,
            "coordenadas_cuerpo": (0, 0, 0, 0),
            "confianza_cuerpo": 0.0,
        }

        matches = face_recognition.compare_faces(
            datos.known_face_encodings, face_encoding
        )

        face_distances = face_recognition.face_distance(
            datos.known_face_encodings, face_encoding
        )

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:  # Rostros identificados
            name = datos.known_face_names[best_match_index]
            p["nombre"] = name
            p["distancia_rostro"] = face_distances[best_match_index]
            rostros.append(p)

        else:  # Rostros desconocidos
            id = contar_desconocidos()
            # myPath = os.path.abspath(os.getcwd())
            myPath = "rostros"
            rostro = frame[top:bottom, left:right]
            name += "_{}".format(id)
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            # if not(comparar_imagen_con_varias(rostro, myPath+"\*")):
            cv2.imwrite(myPath + "\\" + name + ".jpg", rostro)

            image = face_recognition.load_image_file(myPath + "\\" + name + ".jpg")

            if len(face_recognition.face_encodings(image)) > 0:
                face_encoding = face_recognition.face_encodings(image)[0]

                semaforo.acquire()
                datos.known_face_encodings.append(face_encoding)
                datos.known_face_names.append(name)
                semaforo.release()

            cv2.rectangle(frame, (left, top), (right, bottom), datos.ROJO, 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), datos.ROJO, cv2.FILLED
            )
            cv2.putText(
                frame, name, (left + 6, bottom - 6), datos.font, 1.0, datos.BLANCO, 1
            )

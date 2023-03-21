import imutils
import cv2
import numpy as np
import face_recognition
import datos


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
        image, winStride=(5, 5), padding=(3, 3), scale=1.21)

    print('Human Detected : ', len(humans))

    for (x, y, w, h) in humans:
        x1 = int(x*ratio)
        y1 = int(y*ratio)
        w1 = int(w*ratio)
        h1 = int(h*ratio)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
        cv2.putText(frame, "cuerpo hog", (x1, y1 + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)


def deteccion_cuerpo(frame):
    body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    upper_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    lower_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
    upper = upper_cascade.detectMultiScale(gray, 1.3, 5)
    lower = lower_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "body", (x, y+20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    for (x, y, w, h) in upper:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "upper", (x, y+20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    for (x, y, w, h) in lower:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "lower", (x, y+20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]


def deteccion_personas_yolo(frame, cuerpos, coordenadas_local, nombre_camara, net, output_layers):
    height, width, _ = frame.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

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
            cuerpo = {"nombre": "Sin identificar",
                      "coordenadas_local": coordenadas_local,
                      "nombre_camara": nombre_camara,
                      "coordenadas_mapa": (0, 0),
                      "ttl": 0,
                      "coordenadas_rostro": (0, 0, 0, 0),
                      "distancia_rostro": 0.0,
                      "confianza_cuerpo": confidences[i],
                      "coordenadas_cuerpo": (x1, y1, x2, y2)}
            cuerpos.append(cuerpo)
            # label = "Persona: {:.3f}".format(confidences[i])
            # cv2.rectangle(frame, (round(box[0]), round(box[1])), (round(
            #     box[0]+box[2]), round(box[1]+box[3])), (255, 255, 255), 2)
            # cv2.putText(frame, label, (round(
            #     box[0]+15), round(box[1])+15), datos.font, 0.5, (255, 255, 255), 2)


def deteccion_rostros_haar_cascade(frame):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "rostro haar", (x, y+20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1)


def coordenadas_yunet_a_facerec(detections):
    locations = []
    if detections[1] is not None:
        for detection in detections[1]:
            coordenadas_fr = (int(detection[1]),
                              int(detection[0] + detection[2]),
                              int(detection[1] + detection[3]),
                              int(detection[0]))
            locations.append(coordenadas_fr)
    return locations

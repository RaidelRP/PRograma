# importing libraries
from multiprocessing import Semaphore

import cv2

import datos
from datos import TTL_MAX, tracking_general
from functions import (
    coincide_cuerpo_en_tracking,
    coincide_rostro_en_tracking,
    get_iou,
    contenido_en,
)

semaforo = Semaphore(1)


def seguimiento_cuerpo(cuerpos, nombre_camara):
    for cuerpo in cuerpos:
        # verifico si no se ha incluido en el tracking para insertarla
        if not coincide_cuerpo_en_tracking(cuerpo, tracking_general):
            cuerpo["ttl"] = TTL_MAX

            semaforo.acquire()
            tracking_general.append(cuerpo)
            semaforo.release()

        for persona_seguida in tracking_general:
            # si se detecta procedente de un local sin camara, asignarle la camara actual
            if (
                cuerpo["nombre"] == persona_seguida["nombre"]
                and persona_seguida["nombre_camara"] == "NINGUNO"
            ):
                semaforo.acquire()
                persona_seguida["nombre_camara"] = nombre_camara
                semaforo.release()

            if (
                get_iou(
                    cuerpo["coordenadas_cuerpo"], persona_seguida["coordenadas_cuerpo"]
                )
                > 0.1
            ):
                semaforo.acquire()
                persona_seguida["coordenadas_cuerpo"] = cuerpo["coordenadas_cuerpo"]
                semaforo.release()

                if cuerpo["confianza_cuerpo"] > persona_seguida["confianza_cuerpo"]:
                    semaforo.acquire()

                    persona_seguida["confianza_cuerpo"] = cuerpo["confianza_cuerpo"]
                    persona_seguida["nombre"] = cuerpo["nombre"]

                    semaforo.release()

                semaforo.acquire()
                persona_seguida["ttl"] = TTL_MAX
                semaforo.release()


def seguimiento_rostro(rostros, nombre_camara):
    for rostro in rostros:
        # verifico si no se ha incluido en el tracking para insertarla
        if not coincide_rostro_en_tracking(rostro, tracking_general):
            rostro["ttl"] = TTL_MAX

            semaforo.acquire()
            tracking_general.append(rostro)
            semaforo.release()

        for persona_seguida in tracking_general:
            # si se detecta procedente de un local sin camara, asignarle la camara actual
            if (
                rostro["nombre"] == persona_seguida["nombre"]
                and persona_seguida["nombre_camara"] == "NINGUNO"
            ):
                semaforo.acquire()
                persona_seguida["nombre_camara"] = nombre_camara
                semaforo.release()

            if (
                get_iou(
                    rostro["coordenadas_rostro"], persona_seguida["coordenadas_rostro"]
                )
                > 0.1
            ):
                semaforo.acquire()
                persona_seguida["coordenadas_rostro"] = rostro["coordenadas_rostro"]
                semaforo.release()

                if rostro["distancia_rostro"] < persona_seguida["distancia_rostro"]:
                    semaforo.acquire()

                    persona_seguida["distancia_rostro"] = rostro["distancia_rostro"]
                    persona_seguida["nombre"] = rostro["nombre"]

                    semaforo.release()

                semaforo.acquire()
                persona_seguida["ttl"] = TTL_MAX
                semaforo.release()

            # historial(persona_seguida["nombre"])


def seguimiento(rostros, cuerpos, nombre_camara):
    seguimiento_cuerpo(cuerpos, nombre_camara)
    seguimiento_rostro(rostros, nombre_camara)


def seguimiento_cuerpo_2(cuerpos, nombre_camara):
    for cuerpo in cuerpos:
        # verifico si no se ha incluido en el tracking para insertarla
        if not coincide_cuerpo_en_tracking(cuerpo, tracking_general):
            cuerpo["ttl"] = TTL_MAX

            semaforo.acquire()
            tracking_general.append(cuerpo)
            semaforo.release()

        for persona_seguida in tracking_general:
            # si se detecta procedente de un local sin camara, asignarle la camara actual
            if (
                cuerpo["nombre"] == persona_seguida["nombre"]
                and persona_seguida["nombre_camara"] == "NINGUNO"
            ):
                semaforo.acquire()
                persona_seguida["nombre_camara"] = nombre_camara
                semaforo.release()

            if (
                get_iou(
                    cuerpo["coordenadas_cuerpo"], persona_seguida["coordenadas_cuerpo"]
                )
                > 0.1
            ):
                semaforo.acquire()
                persona_seguida["coordenadas_cuerpo"] = cuerpo["coordenadas_cuerpo"]
                persona_seguida["coordenadas_rostro"] = cuerpo["coordenadas_rostro"]
                persona_seguida["distancia_rostro"] = cuerpo["distancia_rostro"]
                semaforo.release()

                if cuerpo["confianza_cuerpo"] > persona_seguida["confianza_cuerpo"]:
                    semaforo.acquire()

                    persona_seguida["confianza_cuerpo"] = cuerpo["confianza_cuerpo"]
                    persona_seguida["nombre"] = cuerpo["nombre"]

                    semaforo.release()

                semaforo.acquire()
                persona_seguida["ttl"] = TTL_MAX
                semaforo.release()

            if cuerpo["coordenadas_rostro"] != (0, 0, 0, 0):
                if (
                    get_iou(
                        cuerpo["coordenadas_rostro"],
                        persona_seguida["coordenadas_rostro"],
                    )
                    > 0.1
                ):
                    semaforo.acquire()
                    persona_seguida["coordenadas_rostro"] = cuerpo["coordenadas_rostro"]
                    persona_seguida["coordenadas_cuerpo"] = cuerpo["coordenadas_cuerpo"]
                    persona_seguida["confianza_cuerpo"] = cuerpo["confianza_cuerpo"]
                    semaforo.release()

                    if cuerpo["distancia_rostro"] < persona_seguida["distancia_rostro"]:
                        semaforo.acquire()

                        persona_seguida["distancia_rostro"] = cuerpo["distancia_rostro"]
                        persona_seguida["nombre"] = cuerpo["nombre"]

                        semaforo.release()

                    semaforo.acquire()
                    persona_seguida["ttl"] = TTL_MAX
                    semaforo.release()


def rectangulo_nombre_rostros(coordenadas_local, nombre_camara, frame, camara):
    for persona in tracking_general:
        # Si se encuentra en el local actual y la camara actual
        if (
            persona["coordenadas_local"] == coordenadas_local
            and persona["nombre_camara"] == nombre_camara
        ):
            if persona["coordenadas_cuerpo"] != (0, 0, 0, 0):
                cv2.rectangle(
                    frame,
                    (
                        persona["coordenadas_cuerpo"][0],
                        persona["coordenadas_cuerpo"][1],
                    ),
                    (
                        persona["coordenadas_cuerpo"][2],
                        persona["coordenadas_cuerpo"][3],
                    ),
                    datos.ROJO,
                    2,
                )

                cv2.rectangle(
                    frame,
                    (
                        persona["coordenadas_cuerpo"][0],
                        persona["coordenadas_cuerpo"][3] - 35,
                    ),
                    (
                        persona["coordenadas_cuerpo"][2],
                        persona["coordenadas_cuerpo"][3],
                    ),
                    datos.ROJO,
                    cv2.FILLED,
                )

                # print(persona["coordenadas_rostro"])

                cv2.rectangle(
                    frame,
                    (
                        persona["coordenadas_rostro"][0],
                        persona["coordenadas_rostro"][1],
                    ),
                    (
                        persona["coordenadas_rostro"][2],
                        persona["coordenadas_rostro"][3],
                    ),
                    datos.NEGRO,
                    2,
                )
                cv2.putText(
                    frame,
                    persona["nombre"] + " TTL: " + str(persona["ttl"]),
                    (
                        persona["coordenadas_cuerpo"][0] + 6,
                        persona["coordenadas_cuerpo"][3] - 6,
                    ),
                    datos.font,
                    1.0,
                    datos.BLANCO,
                    1,
                )


def transferencia(coordenadas_local, nombre_camara, camara):
    for persona in tracking_general:
        # Si se encuentra en el local actual y la camara actual
        if (
            persona["coordenadas_local"] == coordenadas_local
            and persona["nombre_camara"] == nombre_camara
        ):
            if persona["ttl"] == 0:
                i = 0
                for left, top, right, bottom in camara["rectangulos"]:
                    if contenido_en(
                        persona["coordenadas_cuerpo"], (left, top, right, bottom)
                    ):
                        semaforo.acquire()
                        persona["coordenadas_cuerpo"] = camara[
                            "rectangulos_relacionados"
                        ][i]
                        persona["coordenadas_local"] = camara["locales_relacionados"][i]
                        persona["nombre_camara"] = camara["camaras_relacionadas"][i]
                        persona["ttl"] = datos.TTL_MAX
                        semaforo.release()
                    i = i + 1


def rectangulos_entrada_salida(camara, frame):
    for left, top, right, bottom in camara["rectangulos"]:
        cv2.rectangle(frame, (left, top), (right, bottom), datos.VERDE, 2)

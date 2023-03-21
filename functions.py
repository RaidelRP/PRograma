import cv2
import os
from random import randint
from datetime import datetime


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (original_height, original_width) = image.shape[:2]

    if width is None and height is None:
        return image

    elif width is None and not (height is None):
        ratio = height / float(original_height)
        width = int(original_width * ratio)
    elif height is None and not (width is None):
        ratio = width / float(original_width)
        height = int(original_height * ratio)
    new_size = (width, height)
    return cv2.resize(image, new_size, interpolation=inter)


def existe_en_tracking(nombre, tracking):
    for t in tracking:
        if t["nombre"] == nombre:
            return True
    return False


def coincide_rostro_en_tracking(persona, tracking):
    for t in tracking:
        # print("IOU (coincide_rostro_en_tracking)", get_iou(persona["coordenadas_rostro"], t["coordenadas_rostro"]))
        if get_iou(persona["coordenadas_rostro"], t["coordenadas_rostro"]) > 0.1:
            return True
    return False

def coincide_cuerpo_en_tracking(persona, tracking):
    for t in tracking:
        # print("IOU (coincide_cuerpo_en_tracking)", get_iou(persona["coordenadas_rostro"], t["coordenadas_rostro"]))
        if get_iou(persona["coordenadas_cuerpo"], t["coordenadas_cuerpo"]) > 0.1:
            return True
    return False


def pos_en_tracking(nombre, tracking):
    for i in range(0, len(tracking)):
        if tracking[i]["nombre"] == nombre:
            return i
    return -1


def get_iou(bb1, bb2):
    assert bb1[0] <= bb1[2]
    assert bb1[1] <= bb1[3]
    assert bb2[0] <= bb2[2]
    assert bb2[1] <= bb2[3]

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def contar_desconocidos():
    counter = 0
    # myPath = os.path.abspath(os.getcwd())
    myPath = "rostros"
    for root, dirs, files in os.walk(myPath):
        for file in files:
            if file.startswith("Desconocido"):
                counter += 1
    return counter


def coordenada_aleatoria(x, y):
    return randint(x, y)


def historial(nombre):
    with open("historial.txt", "a+") as historial:
        fecha_completa = datetime.now()
        fecha = fecha_completa.strftime("%Y-%m-%d")
        hora = fecha_completa.strftime("%H:%M:%S")
        historial.writelines(f"{fecha} {hora} - {nombre}\n")


def coincide_rostro(rostro, cuerpo):
    assert rostro[0] <= rostro[2]
    assert rostro[1] <= rostro[3]
    assert cuerpo[0] <= cuerpo[2]
    assert cuerpo[1] <= cuerpo[3]

    x_left = max(rostro[0], cuerpo[0])
    y_top = max(rostro[1], cuerpo[1])
    x_right = min(rostro[2], cuerpo[2])
    y_bottom = min(rostro[3], cuerpo[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    rostro_area = (rostro[2] - rostro[0]) * (rostro[3] - rostro[1])

    return (intersection_area >= 0.5 * rostro_area)


def unir_rostros_cuerpos(rostros, cuerpos):
    for rostro in rostros:
        for cuerpo in cuerpos:
            if coincide_rostro(rostro["coordenadas_rostro"], cuerpo["coordenadas_cuerpo"]):
                rostro["coordenadas_cuerpo"] = cuerpo["coordenadas_cuerpo"]
                rostro["confianza_cuerpo"] = cuerpo["confianza_cuerpo"]
                cuerpos.remove(cuerpo)

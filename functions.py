import glob
import os
from datetime import datetime
from random import randint

import cv2

import datos


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
        if get_iou(persona["coordenadas_rostro"], t["coordenadas_rostro"]) > 0.1:
            return True
    return False


def coincide_cuerpo_en_tracking(persona, tracking):
    for t in tracking:
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

    return intersection_area >= 0.5 * rostro_area


def unir_rostros_cuerpos(rostros, cuerpos):
    for rostro in rostros:
        for cuerpo in cuerpos:
            if coincide_rostro(
                rostro["coordenadas_rostro"], cuerpo["coordenadas_cuerpo"]
            ):
                rostro["coordenadas_cuerpo"] = cuerpo["coordenadas_cuerpo"]
                rostro["confianza_cuerpo"] = cuerpo["confianza_cuerpo"]
                cuerpos.remove(cuerpo)


def comparar_imagenes(imagen1, imagen2):
    shift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = shift.detectAndCompute(imagen1, None)
    kp_2, desc_2 = shift.detectAndCompute(imagen2, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)

    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    return len(good_points) / number_keypoints


def comparar_imagen_con_varias(imagen, ruta):  # ruta = "rostros\*"
    # Load all the images
    all_images_to_compare = []
    titles = []
    for f in glob.iglob(ruta):
        image = cv2.imread(f)
        titles.append(f)
        all_images_to_compare.append(image)

    for image_to_compare, title in zip(all_images_to_compare, titles):
        if comparar_imagenes(imagen, image_to_compare) > 0.5:
            return True

    return False


def contenido_en(rect1, rect2):
    return (
        rect1[0] >= rect2[0]
        and rect1[1] >= rect2[1]
        and rect1[2] <= rect2[2]
        and rect1[3] <= rect2[3]
    )


def datos_camara(cam):
    for local in datos.LOCALES:
        for camara in local["camaras"]:
            if camara["nombre_camara"] == cam:
                return camara
    return None


def grid(frame, width, height):
    for i in range(0, width, 10):
        cv2.line(frame, (i, 0), (i, height), (0, 0, 0), 1)
        cv2.putText(frame, str(i), (i, 20), datos.font, 0.5, (0, 0, 0))
    for i in range(0, height, 10):
        cv2.line(frame, (0, i), (width, i), (0, 0, 0), 1)
        cv2.putText(frame, str(i), (0, i), datos.font, 0.5, (0, 0, 0))

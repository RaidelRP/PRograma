import os

import cv2
import face_recognition

from functions import resize

global tracking_general
tracking_general = []

global known_face_encodings
global known_face_names

global imagenes
imagenes = [
    resize(cv2.imread("negro.png"), height=840, width=597),
    resize(cv2.imread("negro.png"), height=280, width=373),
    resize(cv2.imread("negro.png"), height=280, width=373),
    resize(cv2.imread("negro.png"), height=280, width=373),
]

TTL_MAX = 100

known_face_encodings = []
known_face_names = []

imageFacesPath = "rostros"
for file_name in os.listdir(imageFacesPath):
    image = cv2.imread(imageFacesPath + "/" + file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if len(face_recognition.face_encodings(image)) > 0:
        f_coding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(f_coding)
        known_face_names.append(file_name.split(".")[0])

mapa = "plano cenpis.png"

font = cv2.FONT_HERSHEY_DUPLEX

ROJO = (0, 0, 255)
NEGRO = (0, 0, 0)
VERDE = (0, 255, 0)
BLANCO = (255, 255, 255)
MARRON = (35, 50, 78)

AULA_COORD = (0, 0, 405, 350)
LOBBY_COORD = (405, 0, 1075, 350)
LOCAL1_COORD = (0, 1130, 405, 1510)
LOCAL2_COORD = (0, 750, 405, 1130)
LOCAL3_COORD = (0, 350, 405, 750)
PASILLO_COORD = (855, 350, 1075, 980)
BANNO_COORD = (405, 350, 855, 545)
PANTRY_COORD = (405, 545, 855, 750)
AULA_PRE_COORD = (405, 750, 860, 1220)
DIRECCION_COORD = (860, 980, 1075, 1275)
LAB_COORD = (405, 1220, 860, 1510)
LAB_COORD_2 = (860, 1275, 1075, 1510)

CAM1_RECTS = [(0, 250, 100, 480), (530, 350, 640, 480), (465, 100, 590, 300)]
CAM2_RECTS = [
    (0, 0, 140, 280),
    (0, 320, 140, 480),
    (500, 80, 640, 320),
    (500, 340, 640, 480),
]
CAM3_RECTS = [(410, 70, 500, 200)]
# CAM4_RECTS = [(450, 150, 530, 400), (0, 0, 50, 100)]
CAM4_RECTS = [(450, 150, 530, 400)]
# CAM4_RECTS = [(0, 0, 50, 100)]
CAM5_RECTS = [(330, 220, 450, 480), (575, 200, 640, 450)]

CAM1_LOCS_REL = [AULA_COORD, PASILLO_COORD]
CAM2_LOCS_REL = [PASILLO_COORD, LOCAL2_COORD, DIRECCION_COORD, LOCAL1_COORD]
CAM3_LOCS_REL = [LOBBY_COORD]
CAM4_LOCS_REL = [LOCAL2_COORD]
# CAM4_LOCS_REL = [LOCAL2_COORD, AULA_COORD]
# CAM4_LOCS_REL = [AULA_COORD]
CAM5_LOCS_REL = [LOCAL3_COORD, AULA_PRE_COORD]

CAM1_RECTS_REL = [
    (410, 70, 500, 200),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
]
CAM2_RECTS_REL = []
CAM3_RECTS_REL = [(0, 250, 100, 480)]
# CAM4_RECTS_REL = [(330, 220, 450, 480), (200, 150, 400, 480)]
CAM4_RECTS_REL = [(330, 220, 450, 480)]  # E/S de camara 5
CAM5_RECTS_REL = [(450, 180, 530, 400), (0, 0, 0, 0)]  # E/S de camara 4


CAM1 = {
    "nombre_camara": "CAMARA 1",
    "rectangulos": CAM1_RECTS,
    "rectangulos_relacionados": CAM1_RECTS_REL,
    "locales_relacionados": CAM1_LOCS_REL,
    "camaras_relacionadas": ["CAMARA 3", "NINGUNO", "NINGUNO"],
}
CAM2 = {
    "nombre_camara": "CAMARA 2",
    "rectangulos": CAM2_RECTS,
    "rectangulos_relacionados": CAM2_RECTS_REL,
    "locales_relacionados": CAM2_LOCS_REL,
    "camaras_relacionadas": ["NINGUNO", "NINGUNO", "NINGUNO", "NINGUNO"],
}
CAM3 = {
    "nombre_camara": "CAMARA 3",
    "rectangulos": CAM3_RECTS,
    "rectangulos_relacionados": CAM3_RECTS_REL,
    "locales_relacionados": CAM3_LOCS_REL,
    "camaras_relacionadas": ["CAMARA 1"],
}
CAM4 = {
    "nombre_camara": "CAMARA 4",
    "rectangulos": CAM4_RECTS,
    "rectangulos_relacionados": CAM4_RECTS_REL,
    "locales_relacionados": CAM4_LOCS_REL,
    # "camaras_relacionadas": ["CAMARA 5", "CAMARA 3"],
    # "camaras_relacionadas": ["CAMARA 3"],
    "camaras_relacionadas": ["CAMARA 5"],
}
CAM5 = {
    "nombre_camara": "CAMARA 5",
    "rectangulos": CAM5_RECTS,
    "rectangulos_relacionados": CAM5_RECTS_REL,
    "locales_relacionados": CAM5_LOCS_REL,
    "camaras_relacionadas": ["CAMARA 4", "NINGUNO"],
}

AULA = {"nombre_local": "AULA", "coordenadas": AULA_COORD, "camaras": [CAM3]}
LOBBY = {"nombre_local": "LOBBY", "coordenadas": LOBBY_COORD, "camaras": [CAM1]}
LOCAL1 = {"nombre_local": "LOCAL 1", "coordenadas": LOCAL1_COORD, "camaras": []}
LOCAL2 = {"nombre_local": "LOCAL 2", "coordenadas": LOCAL2_COORD, "camaras": [CAM5]}
LOCAL3 = {"nombre_local": "LOCAL 3", "coordenadas": LOCAL3_COORD, "camaras": [CAM4]}
PASILLO = {"nombre_local": "PASILLO", "coordenadas": PASILLO_COORD, "camaras": []}
BANNO = {"nombre_local": "BANNO", "coordenadas": BANNO_COORD, "camaras": []}
PANTRY = {"nombre_local": "PANTRY", "coordenadas": PANTRY_COORD, "camaras": []}
AULA_PRE = {
    "nombre_local": "AULA PREGRADO",
    "coordenadas": AULA_PRE_COORD,
    "camaras": [CAM2],
}
DIRECCION = {"nombre_local": "DIRECCION", "coordenadas": DIRECCION_COORD, "camaras": []}
LAB = {"nombre_local": "LABORATORIO", "coordenadas": LAB_COORD, "camaras": []}
LAB2 = {"nombre_local": "LABORATORIO", "coordenadas": LAB_COORD_2, "camaras": []}

CAMARAS = [CAM1, CAM2]
LOCALES = [
    AULA,
    LOBBY,
    LOCAL1,
    LOCAL2,
    LOCAL3,
    PASILLO,
    BANNO,
    PANTRY,
    AULA_PRE,
    DIRECCION,
    LAB,
    LAB2,
]

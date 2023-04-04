import cv2
import os
import face_recognition
from functions import resize

global tracking_general
tracking_general = []

global known_face_encodings
global known_face_names

global imagenes
imagenes = [resize(cv2.imread("negro.png"), height=840, width=597),
            resize(cv2.imread("negro.png"), height=280, width=373),
            resize(cv2.imread("negro.png"), height=280, width=373),
            resize(cv2.imread("negro.png"), height=280, width=373)
            ]

TTL_MAX = 30

known_face_encodings = []
known_face_names = []

imageFacesPath = "rostros"
for file_name in os.listdir(imageFacesPath):
    image = cv2.imread(imageFacesPath + "/" + file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
LAB_COORD = [(405, 1220, 860, 1510), (860, 1275, 1075, 1510)]

AULA = {"nombre_local": "AULA", "coordenadas": AULA_COORD}
LOBBY = {"nombre_local": "LOBBY",
         "coordenadas": LOBBY_COORD}
LOCAL1 = {"nombre_local": "LOCAL 1",
          "coordenadas": LOCAL1_COORD}
LOCAL2 = {"nombre_local": "LOCAL 2",
          "coordenadas": LOCAL2_COORD}
LOCAL3 = {"nombre_local": "LOCAL 3",
          "coordenadas": LOCAL3_COORD}
PASILLO = {"nombre_local": "PASILLO",
           "coordenadas": PASILLO_COORD}
BANNO = {"nombre_local": "BANNO",
         "coordenadas": BANNO_COORD}
PANTRY = {"nombre_local": "PANTRY",
          "coordenadas": PANTRY_COORD}
AULA_PRE = {"nombre_local": "AULA PREGRADO",
            "coordenadas": AULA_PRE_COORD}
DIRECCION = {"nombre_local": "DIRECCION",
             "coordenadas": DIRECCION_COORD}
LAB = {"nombre_local": "LAB", "coordenadas": LAB_COORD}

# (left, top, right, bottom)
# (top, right, bottom, left)

CAM1_RECTS = [(0, 250, 100, 480), (530, 350, 640, 480), (530, 70, 640, 300)]
CAM2_RECTS = [(0, 0, 140, 280), (0, 320, 140, 480),
              (500, 80, 640, 320), (500, 340, 640, 480)]
# CAM3_RECTS = [(100, 50, 300, 250)]

CAM1_LOCS_REL = [AULA_COORD, PASILLO_COORD]
CAM2_LOCS_REL = [PASILLO_COORD, LOCAL2_COORD, DIRECCION_COORD, LOCAL1_COORD]

AULA_LOCS_REL = [LOBBY_COORD]
LOBBY_LOCS_REL = [AULA_PRE_COORD, PASILLO_COORD]
PASILLO_LOCS_REL = [LOBBY_COORD, DIRECCION_COORD,
                    AULA_PRE_COORD, BANNO_COORD, PANTRY_COORD]
AULA_PRE_LOCS_REL = [PASILLO_COORD, DIRECCION_COORD,
                     LAB_COORD, LOCAL1_COORD, LOCAL2_COORD, LOCAL3_COORD]
LOCAL1_LOCS_REL = [AULA_PRE_COORD]
LOCAL2_LOCS_REL = [LOCAL3_COORD, AULA_PRE_COORD, PASILLO_COORD]
LOCAL3_LOCS_REL = [LOCAL2_COORD, AULA_PRE_COORD]

CAM1_RECTS_REL = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]
CAM2_RECTS_REL = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]

CAM1 = {"nombre_camara": "CAMARA 1", "rectangulos": CAM1_RECTS,
        "locales_relacionados": CAM1_LOCS_REL, "rectangulos_relacionados": CAM1_RECTS_REL, "camaras_relacionadas": ["NINGUNO", "NINGUNO", "NINGUNO"]}
CAM2 = {"nombre_camara": "CAMARA 2", "rectangulos": CAM2_RECTS,
        "locales_relacionados": CAM2_LOCS_REL, "rectangulos_relacionados": CAM2_RECTS_REL, "camaras_relacionadas": ["NINGUNO", "NINGUNO", "NINGUNO", "NINGUNO"]}
# CAM3 = {"nombre_camara": "CAMARA 3", "rectangulos": CAM3_RECTS,
#         "locales_relacionados": [], "rectangulos_relacionados": [], "camaras_relacionadas": ["CAMARA 1"]}

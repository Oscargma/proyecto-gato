import cv2
import mediapipe as mp
from collections import deque
import math
import os
import sys

# --- 1. RUTAS Y CARGA ---
directorio_script = os.path.dirname(os.path.abspath(__file__))

ruta_gato = os.path.join(directorio_script, 'gato.mp4')
ruta_happy = os.path.join(directorio_script, 'happy.mp4')
ruta_perro_jpg = os.path.join(directorio_script, 'perro_inteligente.jpg')
ruta_perro_png = os.path.join(directorio_script, 'perro_inteligente.png')

imagen_perro_original = None
if os.path.exists(ruta_perro_jpg):
    imagen_perro_original = cv2.imread(ruta_perro_jpg)
elif os.path.exists(ruta_perro_png):
    imagen_perro_original = cv2.imread(ruta_perro_png)

if imagen_perro_original is None:
    print("❌ ERROR: No se encontró la imagen perro_inteligente (.jpg o .png)")
    sys.exit()

# --- 2. CONFIGURACIÓN DE IA ---
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils 

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, 
    min_detection_confidence=0.35, 
    min_tracking_confidence=0.4
)
face_detect = mp_face.FaceDetection(min_detection_confidence=0.5)

# --- 3. INICIO DE CÁMARA Y AJUSTE DE TAMAÑOS ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
if not cap.isOpened():
    print("❌ ERROR: No se pudo conectar a la cámara.")
    sys.exit()

success, init_frame = cap.read()
if success:
    h_cam, w_cam, _ = init_frame.shape
else:
    w_cam, h_cam = 640, 480 

imagen_perro = cv2.resize(imagen_perro_original, (w_cam, h_cam), interpolation=cv2.INTER_CUBIC)

video_scuba = cv2.VideoCapture(ruta_gato)
video_happy = cv2.VideoCapture(ruta_happy)

ventana_abierta = False

# NUEVO: Historiales dobles (para X y para Y)
historial_m1_x = deque(maxlen=20) 
historial_m2_x = deque(maxlen=20) 
historial_m1_y = deque(maxlen=20) 
historial_m2_y = deque(maxlen=20) 

print("✅ Todo listo. IA Multi-Meme ejecutándose...")
print("-> Scuba Cat: Mano en cara + otra mano a los lados")
print("-> Happy Cat: Ambas manos ARRIBA y ABAJO")
print("-> Perro Nerd: Dedo índice arriba")

# --- 4. CICLO PRINCIPAL ---
while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1) 
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    res_hands = hands.process(image_rgb)
    res_face = face_detect.process(image_rgb)
    
    cara_x, cara_y = None, None
    
    if res_face.detections:
        bbox = res_face.detections[0].location_data.relative_bounding_box
        cara_x = int((bbox.xmin + bbox.width / 2) * w)
        cara_y = int((bbox.ymin + bbox.height / 2) * h)
        cv2.circle(image, (cara_x, cara_y), 6, (0, 255, 0), -1)

    flag_gato_scuba = False
    flag_gato_happy = False
    flag_perro_inteligente = False

    if res_hands.multi_hand_landmarks:
        for hand_landmarks in res_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        hand_principal = res_hands.multi_hand_landmarks[0]
        
        indice_arriba = hand_principal.landmark[8].y < hand_principal.landmark[6].y
        medio_abajo = hand_principal.landmark[12].y > hand_principal.landmark[10].y
        anular_abajo = hand_principal.landmark[16].y > hand_principal.landmark[14].y
        menique_abajo = hand_principal.landmark[20].y > hand_principal.landmark[18].y
        
        if indice_arriba and medio_abajo and anular_abajo and menique_abajo:
            flag_perro_inteligente = True
            
        if len(res_hands.multi_hand_landmarks) == 2:
            m1 = res_hands.multi_hand_landmarks[0]
            m2 = res_hands.multi_hand_landmarks[1]
            
            mx1, my1 = int(m1.landmark[0].x * w), int(m1.landmark[0].y * h)
            mx2, my2 = int(m2.landmark[0].x * w), int(m2.landmark[0].y * h)
            
            # Guardamos posiciones X e Y
            historial_m1_x.append(mx1)
            historial_m2_x.append(mx2)
            historial_m1_y.append(my1)
            historial_m2_y.append(my2)

            if len(historial_m1_x) == 20:
                mov1_x_val = max(historial_m1_x) - min(historial_m1_x)
                mov2_x_val = max(historial_m2_x) - min(historial_m2_x)
                
                mov1_y_val = max(historial_m1_y) - min(historial_m1_y)
                mov2_y_val = max(historial_m2_y) - min(historial_m2_y)

                # NUEVA LÓGICA: Happy Cat (Arriba y abajo)
                # Requiere más de 40px de mov vertical, y que sea más vertical que horizontal
                if (mov1_y_val > 40 and mov2_y_val > 40) and (mov1_y_val > mov1_x_val and mov2_y_val > mov2_x_val):
                    flag_gato_happy = True
                
                # Scuba Cat se queda igual (Movimiento horizontal a los lados)
                elif cara_x is not None:
                    dist1 = math.hypot(mx1 - cara_x, my1 - cara_y)
                    dist2 = math.hypot(mx2 - cara_x, my2 - cara_y)
                    
                    if (dist1 < 240 and mov2_x_val > 35) or (dist2 < 240 and mov1_x_val > 35):
                        flag_gato_scuba = True
        else:
            historial_m1_x.clear()
            historial_m2_x.clear()
            historial_m1_y.clear()
            historial_m2_y.clear()
    else:
        historial_m1_x.clear()
        historial_m2_x.clear()
        historial_m1_y.clear()
        historial_m2_y.clear()

    # --- 5. CONTROL Y POSICIONAMIENTO DE VENTANAS ---
    if flag_gato_happy:
        tipo_meme_actual = 'video'
        fuente_actual = video_happy
        titulo_ventana = "Happy Happy Happy"
    elif flag_gato_scuba:
        tipo_meme_actual = 'video'
        fuente_actual = video_scuba
        titulo_ventana = "Scuba Cat"
    elif flag_perro_inteligente:
        tipo_meme_actual = 'image'
        fuente_actual = imagen_perro
        titulo_ventana = "Perro Intelectual"
    else:
        tipo_meme_actual = None
        fuente_actual = None

    if tipo_meme_actual == 'video' and fuente_actual is not None:
        ret, frame_video = fuente_actual.read()
        if not ret:
            fuente_actual.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame_video = fuente_actual.read()
        
        if ret and frame_video is not None:
            cv2.imshow('Meme Window', frame_video)
            cv2.setWindowTitle('Meme Window', titulo_ventana)
            cv2.moveWindow('Meme Window', 50 + w_cam + 20, 50)
            ventana_abierta = True
            
    elif tipo_meme_actual == 'image' and fuente_actual is not None:
        cv2.imshow('Meme Window', fuente_actual)
        cv2.setWindowTitle('Meme Window', titulo_ventana)
        cv2.moveWindow('Meme Window', 50 + w_cam + 20, 50)
        ventana_abierta = True
        
    else:
        if ventana_abierta:
            cv2.destroyWindow('Meme Window')
            ventana_abierta = False

    cv2.imshow('Cámara - Proyecto Oscar', image)
    cv2.moveWindow('Cámara - Proyecto Oscar', 50, 50)
    
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
video_scuba.release()
video_happy.release()
cv2.destroyAllWindows()
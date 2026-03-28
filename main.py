# =============================================================================
# main.py
# Entry point dell'applicazione.
# Gestisce il loop principale, la webcam e l'orchestrazione dei moduli.
# =============================================================================

import cv2
import mediapipe as mp

from config import (
    WEBCAM_INDEX, WINDOW_NAME, FULLSCREEN,
    MAX_NUM_FACES, FACE_REFINE_LANDMARKS,
    FACE_MIN_DETECTION_CONF, FACE_MIN_TRACKING_CONF,
    MAX_NUM_HANDS, HAND_MIN_DETECTION_CONF, HAND_MIN_TRACKING_CONF,
    GLASSES_FOLDER, HATS_FOLDER, SMOOTH
)

# -----------------------------------------------------------------------
# Import moduli interni (verranno creati nei prossimi step)
# -----------------------------------------------------------------------
from utils.loader   import load_accessories
from core.overlay   import overlay_png, apply_smooth
from core.face      import apply_glasses, apply_hat, compute_face_metrics
from core.hands     import compute_hand_metrics
from ui.sidebar     import draw_sidebar, on_mouse
from ui.panel       import draw_info_panel

# -----------------------------------------------------------------------
# Inizializzazione MediaPipe
# -----------------------------------------------------------------------
mp_face_mesh      = mp.solutions.face_mesh
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=MAX_NUM_FACES,
    refine_landmarks=FACE_REFINE_LANDMARKS,
    min_detection_confidence=FACE_MIN_DETECTION_CONF,
    min_tracking_confidence=FACE_MIN_TRACKING_CONF
)

hands_detector = mp_hands.Hands(
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=HAND_MIN_DETECTION_CONF,
    min_tracking_confidence=HAND_MIN_TRACKING_CONF
)

# -----------------------------------------------------------------------
# Carica accessori
# -----------------------------------------------------------------------
glasses_list    = load_accessories(GLASSES_FOLDER, "glasses")
hats_list       = load_accessories(HATS_FOLDER,    "hats")
all_accessories = glasses_list + hats_list

# -----------------------------------------------------------------------
# Stato selezione accessori
# None = nessun accessorio selezionato per quella categoria
# -----------------------------------------------------------------------
selected = {
    "glasses": None,
    "hats":    None
}

# -----------------------------------------------------------------------
# Stato smoothing per ogni categoria di accessorio
# x = -1 significa "non ancora inizializzato"
# -----------------------------------------------------------------------
smooth = {
    "glasses": {"x": -1, "y": -1, "w": -1, "a": 0},
    "hats":    {"x": -1, "y": -1, "w": -1, "a": 0}
}

# -----------------------------------------------------------------------
# Apri webcam
# -----------------------------------------------------------------------
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("Errore: webcam non trovata!")
    exit()

# -----------------------------------------------------------------------
# Configura finestra
# -----------------------------------------------------------------------
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
if FULLSCREEN:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# -----------------------------------------------------------------------
# Configura callback mouse (passa frame_w come parametro condiviso)
# -----------------------------------------------------------------------
mouse_param = {"frame_w": 1920}
cv2.setMouseCallback(WINDOW_NAME, on_mouse, {
    "frame_w":        mouse_param,
    "selected":       selected,
    "smooth":         smooth,
    "all_accessories": all_accessories,
    "glasses_list":   glasses_list
})

print(f"Premi ESC per uscire, D per attivare/disattivare il debug")
debug_mode = True

# =============================================================================
# Loop principale
# =============================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Specchio orizzontale
    frame    = cv2.flip(frame, 1)
    h, w     = frame.shape[:2]
    mouse_param["frame_w"] = w

    # Converti BGR → RGB per MediaPipe
    rgb              = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    face_results     = face_mesh.process(rgb)
    hands_results    = hands_detector.process(rgb)
    rgb.flags.writeable = True

    # -------------------------------------------------------------------
    # VOLTO + ACCESSORI
    # -------------------------------------------------------------------
    face_metrics = None
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:

            # Debug: disegna mesh facciale
            if debug_mode:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

            # Calcola misure viso (distanza, larghezza, altezza)
            face_metrics = compute_face_metrics(face_landmarks, w, h)

            # Applica occhiali
            if selected["glasses"] is not None:
                frame, smooth["glasses"] = apply_glasses(
                    frame, face_landmarks,
                    glasses_list[selected["glasses"]]["img"],
                    smooth["glasses"], w, h)

            # Applica cappello
            if selected["hats"] is not None:
                frame, smooth["hats"] = apply_hat(
                    frame, face_landmarks,
                    hats_list[selected["hats"]]["img"],
                    smooth["hats"], w, h)
    else:
        # Nessun volto: resetta smoothing
        smooth["glasses"]["x"] = -1
        smooth["hats"]["x"]    = -1

    # -------------------------------------------------------------------
    # MANI
    # -------------------------------------------------------------------
    hand_metrics = []
    if hands_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):

            # Debug: disegna landmark mani
            if debug_mode:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Calcola misure mano
            handedness   = hands_results.multi_handedness[idx].classification[0].label
            metrics      = compute_hand_metrics(hand_landmarks, handedness, w, face_metrics)
            hand_metrics.append(metrics)

            # Etichetta mano
            wrist = hand_landmarks.landmark[0]
            cv2.putText(frame, handedness,
                (int(wrist.x * w) - 20, int(wrist.y * h) + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # -------------------------------------------------------------------
    # UI: sidebar e pannello info
    # -------------------------------------------------------------------
    frame = draw_sidebar(frame, all_accessories, selected, glasses_list, w)
    frame = draw_info_panel(frame, face_metrics, hand_metrics, debug_mode)

    if frame is None or frame.size == 0:
        print("Frame vuoto!")
        continue
    print(f"Frame shape: {frame.shape}")
    
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:        # ESC
        break
    elif key == ord('d'):
        debug_mode = not debug_mode

# =============================================================================
# Cleanup
# =============================================================================
cap.release()
cv2.destroyAllWindows()
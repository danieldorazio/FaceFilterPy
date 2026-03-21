import cv2
import mediapipe as mp

# -----------------------------------------------------------------------
# Inizializzazione MediaPipe
# Face Mesh: rileva 468 landmark facciali
# Hands: rileva fino a 2 mani con 21 landmark ciascuna
# -----------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,      # landmarks più precisi (occhi e labbra)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------------------------------------------------
# Apri la webcam
# -----------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Errore: webcam non trovata!")
    exit()

# -----------------------------------------------------------------------
# Fullscreen
# -----------------------------------------------------------------------
cv2.namedWindow("Face Filter", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Face Filter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Premi ESC per uscire, D per attivare/disattivare il debug")

debug_mode = True  # mostra i landmark

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Ribalta il frame orizzontalmente (effetto specchio)
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # -----------------------------------------------------------------------
    # MediaPipe lavora in RGB, OpenCV in BGR → converti
    # -----------------------------------------------------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False  # ottimizzazione memoria

    # Processa volto e mani
    face_results  = face_mesh.process(rgb)
    hands_results = hands.process(rgb)

    rgb.flags.writeable = True

    # -----------------------------------------------------------------------
    # VOLTO — disegna i 468 landmark facciali
    # -----------------------------------------------------------------------
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            if debug_mode:
                # Disegna la mesh facciale completa
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Disegna i contorni (occhi, labbra, sopracciglia)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

    # -----------------------------------------------------------------------
    # MANI — disegna i 21 landmark per ogni mano rilevata
    # -----------------------------------------------------------------------
    if hands_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):

            if debug_mode:
                # Disegna le connessioni e i landmark della mano
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Mostra quale mano è (sinistra o destra)
            handedness = hands_results.multi_handedness[idx].classification[0].label
            # Prende il landmark 0 (polso) come riferimento per il testo
            wrist = hand_landmarks.landmark[0]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            cv2.putText(frame, handedness,
                (wrist_x - 20, wrist_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2)

    # -----------------------------------------------------------------------
    # Pannello info in alto a sinistra
    # -----------------------------------------------------------------------
    faces_found = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
    hands_found = len(hands_results.multi_hand_landmarks) if hands_results.multi_hand_landmarks else 0

    cv2.rectangle(frame, (10, 10), (280, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (280, 80), (255, 255, 255), 1)
    cv2.putText(frame, f"Volti: {faces_found}  Mani: {hands_found}",
        (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, f"Debug: {'ON' if debug_mode else 'OFF'}  (D per toggle)",
        (18, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Face Filter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:    # ESC — esci
        break
    elif key == ord('d'):  # D — toggle debug
        debug_mode = not debug_mode

cap.release()
cv2.destroyAllWindows()
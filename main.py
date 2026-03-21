import cv2
import mediapipe as mp
import numpy as np
import os

# -----------------------------------------------------------------------
# Inizializzazione MediaPipe
# -----------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------------------------------------------------
# Carica tutti gli accessori dalle cartelle img/glasses e img/hats
# Restituisce una lista di dizionari con immagine, nome e categoria
# -----------------------------------------------------------------------
def load_accessories(folder, category):
    items = []
    if not os.path.exists(folder):
        return items
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)
            if img is not None:
                items.append({
                    "img":      img,
                    "name":     filename.replace(".png", ""),
                    "category": category
                })
    return items

glasses_list = load_accessories("img/glasses", "glasses")
hats_list    = load_accessories("img/hats",    "hats")

# Tutti gli accessori in un'unica lista per la sidebar
all_accessories = glasses_list + hats_list

# -----------------------------------------------------------------------
# Stato corrente: quale accessorio è selezionato per ogni categoria
# None = nessuno selezionato
# -----------------------------------------------------------------------
selected = {
    "glasses": None,
    "hats":    None
}

# -----------------------------------------------------------------------
# Smoothing per occhiali e cappello
# -----------------------------------------------------------------------
smooth = {
    "glasses": {"x": -1, "y": -1, "w": -1, "a": 0},
    "hats":    {"x": -1, "y": -1, "w": -1, "a": 0}
}
SMOOTH = 0.7

# -----------------------------------------------------------------------
# Sidebar: dimensioni e layout
# -----------------------------------------------------------------------
SIDEBAR_W    = 110   # larghezza sidebar in pixel
ITEM_SIZE    = 90    # dimensione di ogni quadrato accessorio
ITEM_PADDING = 10    # spazio tra i quadrati
ITEM_TOTAL   = ITEM_SIZE + ITEM_PADDING

# -----------------------------------------------------------------------
# Overlay PNG con alpha blending vettorizzato (numpy)
# -----------------------------------------------------------------------
def overlay_png(background, overlay, x, y):
    if overlay is None or overlay.shape[2] < 4:
        return background

    ov_h, ov_w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + ov_w), min(bg_h, y + ov_h)
    if x1 >= x2 or y1 >= y2:
        return background

    ov_x1, ov_y1 = x1 - x, y1 - y
    ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)

    alpha = overlay[ov_y1:ov_y2, ov_x1:ov_x2, 3] / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)

    src     = overlay[ov_y1:ov_y2, ov_x1:ov_x2, :3].astype(float)
    dst     = background[y1:y2, x1:x2].astype(float)
    blended = src * alpha + dst * (1 - alpha)
    background[y1:y2, x1:x2] = blended.astype(np.uint8)

    return background

# -----------------------------------------------------------------------
# Smoothing LERP
# -----------------------------------------------------------------------
def apply_smooth(s, new_x, new_y, new_w, new_a):
    if s["x"] < 0:
        s["x"], s["y"], s["w"], s["a"] = new_x, new_y, new_w, new_a
    else:
        s["x"] = s["x"] * SMOOTH + new_x * (1 - SMOOTH)
        s["y"] = s["y"] * SMOOTH + new_y * (1 - SMOOTH)
        s["w"] = s["w"] * SMOOTH + new_w * (1 - SMOOTH)
        s["a"] = s["a"] * SMOOTH + new_a * (1 - SMOOTH)
    return s

# -----------------------------------------------------------------------
# Disegna la sidebar a destra con tutti gli accessori
# Evidenzia in verde l'accessorio selezionato
# -----------------------------------------------------------------------
def draw_sidebar(frame, accessories, selected, frame_w):
    sidebar_x = frame_w - SIDEBAR_W

    # Sfondo sidebar semitrasparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (sidebar_x, 0), (frame_w, frame.shape[0]), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Separatore verticale
    cv2.line(frame, (sidebar_x, 0), (sidebar_x, frame.shape[0]), (255, 255, 255), 1)

    for i, acc in enumerate(accessories):
        item_y = ITEM_PADDING + i * ITEM_TOTAL
        item_x = sidebar_x + (SIDEBAR_W - ITEM_SIZE) // 2

        # Sfondo del quadrato
        is_selected = selected.get(acc["category"]) == i if acc["category"] == "glasses" else \
                      selected.get(acc["category"]) == i - len(glasses_list)
        bg_color = (0, 180, 0) if is_selected else (60, 60, 60)
        cv2.rectangle(frame, (item_x, item_y), (item_x + ITEM_SIZE, item_y + ITEM_SIZE), bg_color, -1)
        cv2.rectangle(frame, (item_x, item_y), (item_x + ITEM_SIZE, item_y + ITEM_SIZE), (200, 200, 200), 1)

        # Miniatura dell'accessorio dentro il quadrato
        thumb = cv2.resize(acc["img"], (ITEM_SIZE - 10, ITEM_SIZE - 10))
        frame = overlay_png(frame, thumb, item_x + 5, item_y + 5)

        # Etichetta categoria sopra il primo elemento di ogni gruppo
        if i == 0:
            cv2.putText(frame, "GLASSES", (sidebar_x + 5, item_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 220, 255), 1)
        if i == len(glasses_list):
            cv2.putText(frame, "HATS", (sidebar_x + 5, item_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 220, 255), 1)

    return frame

# -----------------------------------------------------------------------
# Gestione click del mouse sulla sidebar
# -----------------------------------------------------------------------
def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    frame_w = param["frame_w"]
    sidebar_x = frame_w - SIDEBAR_W

    # Clicck fuori dalla sidebar → ignora
    if x < sidebar_x:
        return

    for i, acc in enumerate(all_accessories):
        item_y = ITEM_PADDING + i * ITEM_TOTAL
        item_x = sidebar_x + (SIDEBAR_W - ITEM_SIZE) // 2

        if item_x <= x <= item_x + ITEM_SIZE and item_y <= y <= item_y + ITEM_SIZE:
            cat = acc["category"]
            # Calcola l'indice relativo alla categoria
            if cat == "glasses":
                idx = i
            else:
                idx = i - len(glasses_list)

            # Toggle: se già selezionato → deseleziona, altrimenti seleziona
            if selected[cat] == idx:
                selected[cat] = None
                smooth[cat]["x"] = -1
            else:
                selected[cat] = idx
            break

# -----------------------------------------------------------------------
# Posiziona gli occhiali sui landmark degli occhi
# -----------------------------------------------------------------------
def apply_glasses(frame, face_landmarks, img, s, w, h):
    lm        = face_landmarks.landmark
    left_eye  = lm[33]
    right_eye = lm[263]

    lx = int(left_eye.x  * w)
    ly = int(left_eye.y  * h)
    rx = int(right_eye.x * w)
    ry = int(right_eye.y * h)

    eye_width = rx - lx
    center_x  = (lx + rx) // 2
    center_y  = (ly + ry) // 2
    angle     = np.degrees(np.arctan2(ry - ly, rx - lx))

    new_w = int(eye_width * 1.8)
    new_h = int(img.shape[0] * (new_w / img.shape[1]))
    new_x = center_x - new_w // 2
    new_y = center_y - new_h // 2

    s = apply_smooth(s, new_x, new_y, new_w, angle)

    sw = max(1, int(s["w"]))
    sh = max(1, int(img.shape[0] * (sw / img.shape[1])))
    resized = cv2.resize(img, (sw, sh))

    M       = cv2.getRotationMatrix2D((sw // 2, sh // 2), s["a"], 1.0)
    rotated = cv2.warpAffine(resized, M, (sw, sh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0))

    return overlay_png(frame, rotated, int(s["x"]), int(s["y"])), s

# -----------------------------------------------------------------------
# Posiziona il cappello sopra la testa
# Usa il landmark 10 (fronte) e 151 (sommità testa)
# -----------------------------------------------------------------------
def apply_hat(frame, face_landmarks, img, s, w, h):
    lm         = face_landmarks.landmark

    # Landmark per la larghezza della testa
    left_head  = lm[234]   # zigomo sinistro
    right_head = lm[454]   # zigomo destro
    top_head   = lm[10]    # fronte

    lx = int(left_head.x  * w)
    rx = int(right_head.x * w)
    ty = int(top_head.y   * h)

    head_width = rx - lx
    center_x   = (lx + rx) // 2
    angle      = np.degrees(np.arctan2(
        right_head.y - left_head.y,
        right_head.x - left_head.x))

    # Larghezza cappello = 1.4x la larghezza della testa
    new_w = int(head_width * 1.4)
    new_h = int(img.shape[0] * (new_w / img.shape[1]))

    # Posiziona sopra la fronte
    new_x = center_x - new_w // 2
    new_y = ty - new_h

    s = apply_smooth(s, new_x, new_y, new_w, angle)

    sw = max(1, int(s["w"]))
    sh = max(1, int(img.shape[0] * (sw / img.shape[1])))
    resized = cv2.resize(img, (sw, sh))

    M       = cv2.getRotationMatrix2D((sw // 2, sh // 2), s["a"], 1.0)
    rotated = cv2.warpAffine(resized, M, (sw, sh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0))

    return overlay_png(frame, rotated, int(s["x"]), int(s["y"])), s

# -----------------------------------------------------------------------
# Apri webcam e finestra
# -----------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Errore: webcam non trovata!")
    exit()

cv2.namedWindow("Face Filter", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Face Filter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Passa frame_w al callback del mouse tramite param
mouse_param = {"frame_w": 1920}
cv2.setMouseCallback("Face Filter", on_mouse, mouse_param)

print("Premi ESC per uscire, D per attivare/disattivare il debug")
debug_mode = True

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Aggiorna frame_w reale per il mouse callback
    mouse_param["frame_w"] = w

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    face_results  = face_mesh.process(rgb)
    hands_results = hands.process(rgb)

    rgb.flags.writeable = True

    # -----------------------------------------------------------------------
    # VOLTO + ACCESSORI
    # -----------------------------------------------------------------------
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:

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

            # Applica occhiali se selezionati
            if selected["glasses"] is not None:
                idx = selected["glasses"]
                frame, smooth["glasses"] = apply_glasses(
                    frame, face_landmarks,
                    glasses_list[idx]["img"],
                    smooth["glasses"], w, h)

            # Applica cappello se selezionato
            if selected["hats"] is not None:
                idx = selected["hats"]
                frame, smooth["hats"] = apply_hat(
                    frame, face_landmarks,
                    hats_list[idx]["img"],
                    smooth["hats"], w, h)

    else:
        smooth["glasses"]["x"] = -1
        smooth["hats"]["x"]    = -1

    # -----------------------------------------------------------------------
    # MANI
    # -----------------------------------------------------------------------
    if hands_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            if debug_mode:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            handedness = hands_results.multi_handedness[idx].classification[0].label
            wrist      = hand_landmarks.landmark[0]
            cv2.putText(frame, handedness,
                (int(wrist.x * w) - 20, int(wrist.y * h) + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # -----------------------------------------------------------------------
    # Sidebar accessori
    # -----------------------------------------------------------------------
    frame = draw_sidebar(frame, all_accessories, selected, w)

    # -----------------------------------------------------------------------
    # Pannello info
    # -----------------------------------------------------------------------
    faces_found = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
    hands_found = len(hands_results.multi_hand_landmarks) if hands_results.multi_hand_landmarks else 0

    cv2.rectangle(frame, (10, 10), (300, 90), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, 90), (255, 255, 255), 1)
    cv2.putText(frame, f"Volti: {faces_found}  Mani: {hands_found}",
        (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, f"Debug: {'ON' if debug_mode else 'OFF'}  (D per toggle)",
        (18, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Face Filter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('d'):
        debug_mode = not debug_mode

cap.release()
cv2.destroyAllWindows()
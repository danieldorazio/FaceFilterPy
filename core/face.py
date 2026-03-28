# =============================================================================
# core/face.py
# Logica per il posizionamento degli accessori sul volto
# e calcolo delle misure facciali.
# =============================================================================

import numpy as np
from config import (
    LANDMARK_LEFT_EYE, LANDMARK_RIGHT_EYE,
    LANDMARK_LEFT_CHEEK, LANDMARK_RIGHT_CHEEK,
    LANDMARK_FOREHEAD, LANDMARK_CHIN,
    GLASSES_SCALE, HAT_SCALE,
    FACE_REAL_CM, FOCAL_LENGTH
)
from core.overlay import overlay_png, apply_smooth, resize_and_rotate


def compute_face_metrics(face_landmarks, w: int, h: int) -> dict:
    """
    Calcola le misure del volto: larghezza, altezza e distanza dalla webcam.

    Args:
        face_landmarks: landmark facciali MediaPipe
        w, h:           larghezza e altezza del frame in pixel

    Returns:
        Dizionario con: w_px, h_px, dist_cm, w_cm, h_cm
    """
    lm          = face_landmarks.landmark
    left_cheek  = lm[LANDMARK_LEFT_CHEEK]
    right_cheek = lm[LANDMARK_RIGHT_CHEEK]
    forehead    = lm[LANDMARK_FOREHEAD]
    chin        = lm[LANDMARK_CHIN]

    face_w_px = int(abs(right_cheek.x - left_cheek.x) * w)
    face_h_px = int(abs(chin.y - forehead.y) * h)

    face_dist = 0
    face_w_cm = 0.0
    face_h_cm = 0.0

    if face_w_px > 0:
        face_dist = int((FACE_REAL_CM * FOCAL_LENGTH) / face_w_px)
        face_w_cm = round((face_w_px * face_dist) / FOCAL_LENGTH, 1)
        face_h_cm = round((face_h_px * face_dist) / FOCAL_LENGTH, 1)

    return {
        "w_px":   face_w_px,
        "h_px":   face_h_px,
        "dist_cm": face_dist,
        "w_cm":   face_w_cm,
        "h_cm":   face_h_cm
    }


def apply_glasses(frame, face_landmarks, img, s: dict, w: int, h: int):
    """
    Posiziona gli occhiali sul volto usando i landmark degli occhi.
    Landmark 33 = occhio sinistro, 263 = occhio destro.

    Args:
        frame:          frame corrente della webcam
        face_landmarks: landmark facciali MediaPipe
        img:            immagine degli occhiali (BGRA)
        s:              stato smoothing
        w, h:           dimensioni frame

    Returns:
        Tuple (frame aggiornato, stato smoothing aggiornato)
    """
    lm        = face_landmarks.landmark
    left_eye  = lm[LANDMARK_LEFT_EYE]
    right_eye = lm[LANDMARK_RIGHT_EYE]

    lx = int(left_eye.x  * w)
    ly = int(left_eye.y  * h)
    rx = int(right_eye.x * w)
    ry = int(right_eye.y * h)

    eye_width = rx - lx
    center_x  = (lx + rx) // 2
    center_y  = (ly + ry) // 2
    angle     = np.degrees(np.arctan2(ry - ly, rx - lx))

    new_w = int(eye_width * GLASSES_SCALE)
    new_h = int(img.shape[0] * (new_w / img.shape[1]))
    new_x = center_x - new_w // 2
    new_y = center_y - new_h // 2

    s       = apply_smooth(s, new_x, new_y, new_w, angle)
    rotated = resize_and_rotate(img, int(s["w"]), s["a"])

    return overlay_png(frame, rotated, int(s["x"]), int(s["y"])), s


def apply_hat(frame, face_landmarks, img, s: dict, w: int, h: int):
    """
    Posiziona il cappello sopra la testa usando i landmark della testa.
    Landmark 234 = zigomo sinistro, 454 = zigomo destro, 10 = fronte.

    Args:
        frame:          frame corrente della webcam
        face_landmarks: landmark facciali MediaPipe
        img:            immagine del cappello (BGRA)
        s:              stato smoothing
        w, h:           dimensioni frame

    Returns:
        Tuple (frame aggiornato, stato smoothing aggiornato)
    """
    lm         = face_landmarks.landmark
    left_head  = lm[LANDMARK_LEFT_CHEEK]
    right_head = lm[LANDMARK_RIGHT_CHEEK]
    top_head   = lm[LANDMARK_FOREHEAD]

    lx = int(left_head.x  * w)
    rx = int(right_head.x * w)
    ty = int(top_head.y   * h)

    head_width = rx - lx
    center_x   = (lx + rx) // 2
    angle      = np.degrees(np.arctan2(
        right_head.y - left_head.y,
        right_head.x - left_head.x))

    new_w = int(head_width * HAT_SCALE)
    new_h = int(img.shape[0] * (new_w / img.shape[1]))
    new_x = center_x - new_w // 2
    new_y = ty - new_h

    s       = apply_smooth(s, new_x, new_y, new_w, angle)
    rotated = resize_and_rotate(img, int(s["w"]), s["a"])

    return overlay_png(frame, rotated, int(s["x"]), int(s["y"])), s
# =============================================================================
# core/accessories/glasses.py
# Posizionamento degli occhiali sul volto.
# Le misure reali dell'occhiale sono definite in config.py
# =============================================================================

import cv2
import numpy as np
from config import (
    LANDMARK_LEFT_CHEEK, LANDMARK_RIGHT_CHEEK,
    FACE_REAL_CM, FOCAL_LENGTH,
    GLASSES_REAL_W_CM, GLASSES_REAL_H_CM
)
from core.overlay import overlay_png, apply_smooth, resize_and_rotate


def apply_glasses(frame, face_landmarks, img, s: dict, w: int, h: int):
    """
    Posiziona gli occhiali sul volto usando le misure reali dell'occhiale
    e la distanza stimata dalla webcam per calcolare la scala esatta.

    Landmark usati:
        33  = angolo esterno occhio sinistro
        263 = angolo esterno occhio destro
        234 = zigomo sinistro (larghezza viso)
        454 = zigomo destro (larghezza viso)

    Args:
        frame:          frame corrente della webcam
        face_landmarks: landmark facciali MediaPipe
        img:            immagine degli occhiali (BGRA)
        s:              stato smoothing
        w, h:           dimensioni frame in pixel

    Returns:
        Tuple (frame aggiornato, stato smoothing aggiornato)
    """
    lm = face_landmarks.landmark

    left_eye_outer  = lm[33]
    right_eye_outer = lm[263]
    left_cheek      = lm[LANDMARK_LEFT_CHEEK]
    right_cheek     = lm[LANDMARK_RIGHT_CHEEK]

    # Calcola distanza dalla webcam tramite larghezza del viso
    face_w_px = abs(right_cheek.x - left_cheek.x) * w
    if face_w_px <= 0:
        return frame, s

    dist_cm      = (FACE_REAL_CM * FOCAL_LENGTH) / face_w_px

    # Calcola dimensioni occhiali in pixel dalle misure reali
    # Formula pinhole inversa: px = (cm * FOCAL_LENGTH) / dist_cm
    glasses_w_px = int((GLASSES_REAL_W_CM * FOCAL_LENGTH) / dist_cm)
    glasses_h_px = int((GLASSES_REAL_H_CM * FOCAL_LENGTH) / dist_cm)

    if glasses_w_px <= 0 or glasses_h_px <= 0:
        return frame, s

    # Coordinate pixel degli occhi
    lx    = left_eye_outer.x  * w
    ly    = left_eye_outer.y  * h
    rx    = right_eye_outer.x * w
    ry    = right_eye_outer.y * h

    # Angolo di inclinazione della testa
    angle = np.degrees(np.arctan2(ry - ly, rx - lx))

    # Centro orizzontale tra gli occhi
    # Offset verticale negativo per alzare leggermente gli occhiali
    center_x = int((lx + rx) / 2)
    center_y = int((ly + ry) / 2) - int(glasses_h_px * 0.1)
    new_x    = center_x - glasses_w_px // 2
    new_y    = center_y - glasses_h_px // 2

    # Smoothing LERP
    s       = apply_smooth(s, new_x, new_y, glasses_w_px, angle)
    sw      = max(1, int(s["w"]))
    rotated = resize_and_rotate(img, sw, s["a"])

    return overlay_png(frame, rotated, int(s["x"]), int(s["y"])), s
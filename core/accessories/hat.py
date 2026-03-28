# =============================================================================
# core/accessories/hat.py
# Posizionamento del cappello sopra la testa.
# =============================================================================

import numpy as np
from config import (
    LANDMARK_LEFT_CHEEK, LANDMARK_RIGHT_CHEEK,
    LANDMARK_FOREHEAD, HAT_SCALE
)
from core.overlay import overlay_png, apply_smooth, resize_and_rotate


def apply_hat(frame, face_landmarks, img, s: dict, w: int, h: int):
    """
    Posiziona il cappello sopra la testa usando i landmark della testa.

    Landmark usati:
        234 = zigomo sinistro (larghezza testa)
        454 = zigomo destro (larghezza testa)
        10  = fronte (punto più alto rilevato)

    Args:
        frame:          frame corrente della webcam
        face_landmarks: landmark facciali MediaPipe
        img:            immagine del cappello (BGRA)
        s:              stato smoothing
        w, h:           dimensioni frame in pixel

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

    # Angolo di inclinazione della testa
    angle = np.degrees(np.arctan2(
        right_head.y - left_head.y,
        right_head.x - left_head.x))

    # Larghezza cappello = HAT_SCALE x larghezza testa
    new_w = int(head_width * HAT_SCALE)
    new_h = int(img.shape[0] * (new_w / img.shape[1]))

    # Posiziona sopra la fronte
    new_x = center_x - new_w // 2
    new_y = ty - new_h

    # Smoothing LERP
    s       = apply_smooth(s, new_x, new_y, new_w, angle)
    rotated = resize_and_rotate(img, int(s["w"]), s["a"])

    return overlay_png(frame, rotated, int(s["x"]), int(s["y"])), s
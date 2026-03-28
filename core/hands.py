# =============================================================================
# core/hands.py
# Calcolo delle misure delle mani.
# =============================================================================

from config import (
    FOCAL_LENGTH,
    LANDMARK_PALM_LEFT, LANDMARK_PALM_RIGHT,
    LANDMARK_WRIST, LANDMARK_MIDDLE_TIP
)


def compute_hand_metrics(hand_landmarks, handedness: str, w: int, face_metrics: dict) -> dict:
    """
    Calcola le misure di una mano: larghezza palmo e lunghezza.
    Se disponibile la distanza del viso, converte i pixel in cm.

    Args:
        hand_landmarks: landmark della mano MediaPipe
        handedness:     "Left" o "Right"
        w:              larghezza del frame in pixel
        face_metrics:   dizionario con le misure del viso (può essere None)

    Returns:
        Dizionario con: handedness, palm_w_px, hand_len_px, palm_w_cm, hand_len_cm
    """
    lm = hand_landmarks.landmark

    # Larghezza palmo: base indice → base mignolo
    palm_w_px = int(abs(lm[LANDMARK_PALM_LEFT].x - lm[LANDMARK_PALM_RIGHT].x) * w)

    # Lunghezza mano: polso → punta del medio
    hand_len_px = int((
        (lm[LANDMARK_WRIST].x - lm[LANDMARK_MIDDLE_TIP].x) ** 2 +
        (lm[LANDMARK_WRIST].y - lm[LANDMARK_MIDDLE_TIP].y) ** 2
    ) ** 0.5 * w)

    # Converti in cm se abbiamo la distanza del viso
    palm_w_cm   = 0.0
    hand_len_cm = 0.0

    if face_metrics and face_metrics["dist_cm"] > 0:
        dist = face_metrics["dist_cm"]
        palm_w_cm   = round((palm_w_px   * dist) / FOCAL_LENGTH, 1)
        hand_len_cm = round((hand_len_px * dist) / FOCAL_LENGTH, 1)

    return {
        "handedness":   handedness,
        "palm_w_px":    palm_w_px,
        "hand_len_px":  hand_len_px,
        "palm_w_cm":    palm_w_cm,
        "hand_len_cm":  hand_len_cm
    }
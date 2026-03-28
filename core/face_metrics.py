# =============================================================================
# core/face_metrics.py
# Calcolo delle misure facciali: larghezza, altezza e distanza dalla webcam.
# =============================================================================

from config import (
    LANDMARK_LEFT_CHEEK, LANDMARK_RIGHT_CHEEK,
    LANDMARK_FOREHEAD, LANDMARK_CHIN,
    FACE_REAL_CM, FOCAL_LENGTH
)


def compute_face_metrics(face_landmarks, w: int, h: int) -> dict:
    """
    Calcola le misure del volto: larghezza, altezza e distanza dalla webcam.

    Landmark usati:
        234 = zigomo sinistro (larghezza)
        454 = zigomo destro (larghezza)
        10  = fronte (altezza)
        152 = mento (altezza)

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
        "w_px":    face_w_px,
        "h_px":    face_h_px,
        "dist_cm": face_dist,
        "w_cm":    face_w_cm,
        "h_cm":    face_h_cm
    }
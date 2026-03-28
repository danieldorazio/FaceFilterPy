# =============================================================================
# core/overlay.py
# Funzioni per sovrapporre immagini PNG con trasparenza e smoothing LERP.
# =============================================================================

import cv2
import numpy as np
from config import SMOOTH


def overlay_png(background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Sovrappone un'immagine PNG con canale alpha su un'altra immagine.
    Usa alpha blending vettorizzato con numpy per le performance.

    Args:
        background: immagine di sfondo (BGR)
        overlay:    immagine da sovrapporre (BGRA)
        x, y:       posizione in pixel dell'angolo in alto a sinistra

    Returns:
        Immagine con overlay applicato
    """
    if overlay is None or overlay.shape[2] < 4:
        return background

    ov_h, ov_w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    # Calcola area di sovrapposizione gestendo i bordi
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + ov_w), min(bg_h, y + ov_h)
    if x1 >= x2 or y1 >= y2:
        return background

    ov_x1, ov_y1 = x1 - x, y1 - y
    ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)

    # Estrae canale alpha e normalizza tra 0 e 1
    alpha   = overlay[ov_y1:ov_y2, ov_x1:ov_x2, 3] / 255.0
    alpha   = np.stack([alpha] * 3, axis=-1)

    src     = overlay[ov_y1:ov_y2, ov_x1:ov_x2, :3].astype(float)
    dst     = background[y1:y2, x1:x2].astype(float)
    blended = src * alpha + dst * (1 - alpha)
    background[y1:y2, x1:x2] = blended.astype(np.uint8)

    return background


def apply_smooth(s: dict, new_x: float, new_y: float, new_w: float, new_a: float) -> dict:
    """
    Applica interpolazione lineare (LERP) sulla posizione di un accessorio.
    Elimina i movimenti bruschi (flickering) tra un frame e l'altro.

    Args:
        s:     dizionario con i valori precedenti {x, y, w, a}
        new_x: nuova posizione X
        new_y: nuova posizione Y
        new_w: nuova larghezza
        new_a: nuovo angolo

    Returns:
        Dizionario aggiornato con i valori interpolati
    """
    if s["x"] < 0:
        # Prima rilevazione: inizializza direttamente
        s["x"], s["y"], s["w"], s["a"] = new_x, new_y, new_w, new_a
    else:
        # LERP: valore = vecchio * SMOOTH + nuovo * (1 - SMOOTH)
        s["x"] = s["x"] * SMOOTH + new_x * (1 - SMOOTH)
        s["y"] = s["y"] * SMOOTH + new_y * (1 - SMOOTH)
        s["w"] = s["w"] * SMOOTH + new_w * (1 - SMOOTH)
        s["a"] = s["a"] * SMOOTH + new_a * (1 - SMOOTH)
    return s


def resize_and_rotate(img: np.ndarray, width: int, angle: float) -> np.ndarray:
    """
    Ridimensiona un'immagine mantenendo le proporzioni e la ruota.

    Args:
        img:   immagine sorgente (BGRA)
        width: larghezza target in pixel
        angle: angolo di rotazione in gradi

    Returns:
        Immagine ridimensionata e ruotata
    """
    width  = max(1, width)
    height = max(1, int(img.shape[0] * (width / img.shape[1])))

    resized   = cv2.resize(img, (width, height))
    cx, cy    = width // 2, height // 2
    M         = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated   = cv2.warpAffine(resized, M, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0))

    return rotated
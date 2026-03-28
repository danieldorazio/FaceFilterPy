# =============================================================================
# utils/loader.py
# Caricamento degli accessori dalle cartelle img/
# =============================================================================

import cv2
import os

def load_accessories(folder: str, category: str) -> list:
    """
    Carica tutte le immagini PNG da una cartella come accessori.
    
    Args:
        folder:   percorso della cartella (es. "img/glasses")
        category: categoria dell'accessorio (es. "glasses", "hats")
    
    Returns:
        Lista di dizionari con chiavi: img, name, category
    """
    items = []

    if not os.path.exists(folder):
        print(f"Attenzione: cartella '{folder}' non trovata.")
        return items

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".png"):
            continue

        path = os.path.join(folder, filename)
        img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Attenzione: impossibile caricare '{path}'.")
            continue

        items.append({
            "img":      img,
            "name":     filename.replace(".png", ""),
            "category": category
        })

    print(f"Caricati {len(items)} accessori da '{folder}'.")
    return items
# =============================================================================
# ui/sidebar.py
# Sidebar degli accessori: disegno e gestione click del mouse.
# =============================================================================

import cv2
from config import SIDEBAR_W, ITEM_SIZE, ITEM_PADDING, ITEM_TOTAL
from core.overlay import overlay_png


def draw_sidebar(frame, accessories: list, selected: dict, glasses_list: list, frame_w: int):
    sidebar_x = frame_w - SIDEBAR_W

    # Sfondo semitrasparente — metodo corretto senza sovrascrivere frame
    sub = frame[0:frame.shape[0], sidebar_x:frame_w]
    black = cv2.addWeighted(sub, 0.3, 
        cv2.rectangle(sub.copy(), (0, 0), (sub.shape[1], sub.shape[0]), (30, 30, 30), -1),
        0.7, 0)
    frame[0:frame.shape[0], sidebar_x:frame_w] = black

    # Separatore verticale
    cv2.line(frame, (sidebar_x, 0), (sidebar_x, frame.shape[0]), (255, 255, 255), 1)

    for i, acc in enumerate(accessories):
        item_y = ITEM_PADDING + i * ITEM_TOTAL
        item_x = sidebar_x + (SIDEBAR_W - ITEM_SIZE) // 2

        if acc["category"] == "glasses":
            is_selected = selected.get("glasses") == i
        else:
            is_selected = selected.get("hats") == i - len(glasses_list)

        bg_color = (0, 180, 0) if is_selected else (60, 60, 60)
        cv2.rectangle(frame,
            (item_x, item_y),
            (item_x + ITEM_SIZE, item_y + ITEM_SIZE),
            bg_color, -1)
        cv2.rectangle(frame,
            (item_x, item_y),
            (item_x + ITEM_SIZE, item_y + ITEM_SIZE),
            (200, 200, 200), 1)

        thumb = cv2.resize(acc["img"], (ITEM_SIZE - 10, ITEM_SIZE - 10))
        frame = overlay_png(frame, thumb, item_x + 5, item_y + 5)

        if i == 0:
            cv2.putText(frame, "GLASSES", (sidebar_x + 5, item_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 220, 255), 1)
        if i == len(glasses_list):
            cv2.putText(frame, "HATS", (sidebar_x + 5, item_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 220, 255), 1)

    return frame


def on_mouse(event, x: int, y: int, flags, param: dict):
    """
    Callback del mouse per la selezione degli accessori nella sidebar.
    Toggle: click su un accessorio selezionato lo deseleziona.

    Args:
        event:  tipo di evento mouse
        x, y:   coordinate del click
        flags:  flag aggiuntivi OpenCV
        param:  dizionario con frame_w, selected, smooth, all_accessories, glasses_list
    """
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    frame_w      = param["frame_w"]["frame_w"]
    selected     = param["selected"]
    smooth       = param["smooth"]
    accessories  = param["all_accessories"]
    glasses_list = param["glasses_list"]
    sidebar_x    = frame_w - SIDEBAR_W

    # Click fuori dalla sidebar → ignora
    if x < sidebar_x:
        return

    for i, acc in enumerate(accessories):
        item_y = ITEM_PADDING + i * ITEM_TOTAL
        item_x = sidebar_x + (SIDEBAR_W - ITEM_SIZE) // 2

        if item_x <= x <= item_x + ITEM_SIZE and item_y <= y <= item_y + ITEM_SIZE:
            cat = acc["category"]
            idx = i if cat == "glasses" else i - len(glasses_list)

            # Toggle selezione
            if selected[cat] == idx:
                selected[cat]    = None
                smooth[cat]["x"] = -1
            else:
                selected[cat] = idx
            break
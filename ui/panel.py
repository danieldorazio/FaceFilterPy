# =============================================================================
# ui/panel.py
# Pannello informativo in alto a sinistra con misure di viso e mani.
# =============================================================================

import cv2


def draw_info_panel(frame, face_metrics: dict, hand_metrics: list, debug_mode: bool):
    faces_found = 1 if face_metrics else 0
    hands_found = len(hand_metrics)

    panel_h = 95 + hands_found * 18
    cv2.rectangle(frame, (10, 10), (260, 10 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (260, 10 + panel_h), (255, 255, 255), 1)

    cv2.putText(frame, f"Volti: {faces_found}  Mani: {hands_found}",
        (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    if face_metrics:
        cv2.putText(frame, f"Viso: {face_metrics['w_cm']} x {face_metrics['h_cm']} cm",
            (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
        cv2.putText(frame, f"Dist: ~{face_metrics['dist_cm']} cm",
            (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 150), 1)
    else:
        cv2.putText(frame, "Viso: non rilevato",
            (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 200), 1)

    cv2.putText(frame, f"Debug: {'ON' if debug_mode else 'OFF'}  (D toggle)",
        (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    for i, m in enumerate(hand_metrics):
        if m["palm_w_cm"] > 0:
            text = f"{m['handedness']}: {m['palm_w_cm']} x {m['hand_len_cm']} cm"
        else:
            text = f"{m['handedness']}: {m['palm_w_px']} x {m['hand_len_px']} px"

        cv2.putText(frame, text,
            (18, 100 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 255, 100), 1)

    return frame
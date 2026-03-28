# =============================================================================
# config.py
# Costanti globali del progetto.
# Modifica qui per cambiare il comportamento dell'applicazione.
# =============================================================================

# -----------------------------------------------------------------------
# Webcam
# -----------------------------------------------------------------------
WEBCAM_INDEX = 0          # indice webcam (0 = predefinita)

# -----------------------------------------------------------------------
# Finestra
# -----------------------------------------------------------------------
WINDOW_NAME  = "Face Filter"
FULLSCREEN   = True

# -----------------------------------------------------------------------
# MediaPipe - Face Mesh
# -----------------------------------------------------------------------
MAX_NUM_FACES            = 1
FACE_REFINE_LANDMARKS    = True
FACE_MIN_DETECTION_CONF  = 0.5
FACE_MIN_TRACKING_CONF   = 0.5

# -----------------------------------------------------------------------
# MediaPipe - Hands
# -----------------------------------------------------------------------
MAX_NUM_HANDS            = 2
HAND_MIN_DETECTION_CONF  = 0.5
HAND_MIN_TRACKING_CONF   = 0.5

# -----------------------------------------------------------------------
# Smoothing LERP
# Controlla quanto gli accessori seguono fluidamente il volto.
# 0.0 = nessuno smoothing (scattoso)
# 1.0 = massimo smoothing (lentissimo)
# -----------------------------------------------------------------------
SMOOTH = 0.7

# -----------------------------------------------------------------------
# Stima distanza (formula pinhole camera)
# FACE_REAL_CM = larghezza media reale di un viso adulto in cm
# FOCAL_LENGTH = valore calibrato empiricamente per webcam standard
# -----------------------------------------------------------------------
FACE_REAL_CM = 16.0
FOCAL_LENGTH = 600.0

# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
SIDEBAR_W    = 110    # larghezza sidebar in pixel
ITEM_SIZE    = 90     # dimensione quadrato accessorio
ITEM_PADDING = 10     # spazio tra i quadrati
ITEM_TOTAL   = ITEM_SIZE + ITEM_PADDING

# -----------------------------------------------------------------------
# Cartelle accessori
# -----------------------------------------------------------------------
GLASSES_FOLDER = "img/glasses"
HATS_FOLDER    = "img/hats"

# -----------------------------------------------------------------------
# Landmark facciali utilizzati
# -----------------------------------------------------------------------
LANDMARK_LEFT_EYE    = 33     # angolo esterno occhio sinistro
LANDMARK_RIGHT_EYE   = 263    # angolo esterno occhio destro
LANDMARK_LEFT_CHEEK  = 234    # zigomo sinistro (larghezza testa)
LANDMARK_RIGHT_CHEEK = 454    # zigomo destro (larghezza testa)
LANDMARK_FOREHEAD    = 10     # fronte
LANDMARK_CHIN        = 152    # mento
LANDMARK_PALM_LEFT   = 5      # base indice (larghezza palmo)
LANDMARK_PALM_RIGHT  = 17     # base mignolo (larghezza palmo)
LANDMARK_WRIST       = 0      # polso
LANDMARK_MIDDLE_TIP  = 12     # punta del medio (lunghezza mano)

# -----------------------------------------------------------------------
# Scala accessori
# -----------------------------------------------------------------------
GLASSES_SCALE = 1.8   # larghezza occhiali rispetto alla distanza tra gli occhi
HAT_SCALE     = 1.4   # larghezza cappello rispetto alla larghezza della testa

# Misure reali occhiale (aggiornale per ogni nuovo occhiale)
GLASSES_REAL_W_CM = 15.22
GLASSES_REAL_H_CM =  4.93
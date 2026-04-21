import cv2
import mediapipe as mp
import numpy as np
import random
import math

# ─────────────────────────────────────────────
# Camera setup  ← buffer=1 kills latency
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # ← single-frame buffer → zero stale frames

# ─────────────────────────────────────────────
# MediaPipe  — lighter model for lower latency
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
    model_complexity=0              # ← 0 = fast/lite model (was default 1)
)
mp_draw = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
# Canvas
# ─────────────────────────────────────────────
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

prev_x, prev_y         = 0, 0
smooth_x, smooth_y     = 0.0, 0.0
SMOOTH_FACTOR          = 2.5       # lower = snappier tracking
SMOOTH_PINCH           = 2.0

# ─────────────────────────────────────────────
# Pinch-to-move
# ─────────────────────────────────────────────
prev_pinch_x, prev_pinch_y = 0, 0
smooth_px, smooth_py        = 0.0, 0.0
PINCH_THRESHOLD             = 48   # px between thumb tip & index tip

# ─────────────────────────────────────────────
# Sparkle particle system
# ─────────────────────────────────────────────
sparkles = []   # {x, y, vx, vy, life, max_life, color, size, angle, spin}

SPARKLE_COLORS = [
    (255, 230,  80),   # gold
    (255, 180, 255),   # pink
    (180, 255, 255),   # cyan
    (255, 255, 255),   # white
    (200, 255, 150),   # lime
    (255, 150, 100),   # orange
]
MAX_SPARKLES = 120   # cap so it never gets expensive

def spawn_sparkles(x, y):
    if len(sparkles) >= MAX_SPARKLES:
        return
    slots = min(MAX_SPARKLES - len(sparkles), random.randint(3, 6))
    for _ in range(slots):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 4.5)
        life  = random.randint(10, 20)
        sparkles.append({
            "x":        float(x),  "y":      float(y),
            "vx":       math.cos(angle) * speed,
            "vy":       math.sin(angle) * speed,
            "life":     life,      "max_life": life,
            "color":    random.choice(SPARKLE_COLORS),
            "size":     random.uniform(2.5, 5.0),
            "angle":    random.uniform(0, math.pi),
            "spin":     random.uniform(-0.35, 0.35),
        })


# Reusable sparkle overlay (drawn once per frame, not per particle)
_sparkle_layer = np.zeros((720, 1280, 3), dtype=np.uint8)

def update_and_draw_sparkles(output):
    """Physics update + batch-render onto a reusable layer, then alpha-blend once."""
    _sparkle_layer[:] = 0          # clear once

    alive = []
    for sp in sparkles:
        sp["life"] -= 1
        if sp["life"] <= 0:
            continue
        sp["x"]     += sp["vx"]
        sp["y"]     += sp["vy"]
        sp["vy"]    += 0.18        # gravity
        sp["vx"]    *= 0.94        # drag
        sp["angle"] += sp["spin"]

        alpha  = sp["life"] / sp["max_life"]
        size   = max(1.0, sp["size"] * alpha)
        cx, cy = int(sp["x"]), int(sp["y"])

        if not (0 <= cx < 1280 and 0 <= cy < 720):
            continue

        # Draw a 4-pointed star directly on _sparkle_layer (no frame copy!)
        c = sp["color"]
        bright = (
            min(255, int(c[0] * alpha)),
            min(255, int(c[1] * alpha)),
            min(255, int(c[2] * alpha)),
        )
        pts = []
        for i in range(8):
            a = sp["angle"] + i * math.pi / 4
            r = size if i % 2 == 0 else size * 0.35
            pts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
        cv2.fillPoly(_sparkle_layer, [np.array(pts, dtype=np.int32)], bright)
        alive.append(sp)

    sparkles.clear()
    sparkles.extend(alive)

    # Single addWeighted for all sparkles combined
    if alive:
        cv2.add(output, _sparkle_layer, output)


# ─────────────────────────────────────────────
# Brush / Tool settings
# ─────────────────────────────────────────────
current_tool   = "blue"
toolbar_height = 90

TOOLS = [
    {"name": "blue",   "color": (220, 130,  30), "icon": "B", "label": "Blue"},
    {"name": "red",    "color": ( 50,  50, 220), "icon": "R", "label": "Red"},
    {"name": "green",  "color": ( 50, 200,  80), "icon": "G", "label": "Green"},
    {"name": "yellow", "color": ( 30, 220, 220), "icon": "Y", "label": "Yellow"},
    {"name": "eraser", "color": (100, 100, 100), "icon": "E", "label": "Erase"},
    {"name": "clear",  "color": ( 60,  60,  60), "icon": "C", "label": "Clear"},
]

DRAW_COLORS = {
    "blue":   (255, 100,  30),
    "red":    ( 50,  50, 255),
    "green":  ( 60, 220,  60),
    "yellow": ( 30, 230, 230),
}

BOX_W, BOX_H   = 150, 68
BOX_MARGIN     = 18
TOOLBAR_START_X = 30


def draw_toolbar(frame, cur_tool):
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (1280, toolbar_height + 6), (20, 20, 30), -1)
    cv2.addWeighted(bar, 0.72, frame, 0.28, 0, frame)
    cv2.line(frame, (0, toolbar_height + 5), (1280, toolbar_height + 5), (80, 80, 120), 1)

    for i, tool in enumerate(TOOLS):
        x1 = TOOLBAR_START_X + i * (BOX_W + BOX_MARGIN)
        x2 = x1 + BOX_W
        y1, y2 = 11, 11 + BOX_H
        cx = (x1 + x2) // 2
        is_active  = (tool["name"] == cur_tool)
        tool_color = tool["color"]

        if is_active:
            cv2.rectangle(frame, (x1, y1), (x2, y2), tool_color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        else:
            btn = frame.copy()
            cv2.rectangle(btn, (x1, y1), (x2, y2), (35, 35, 50), -1)
            cv2.addWeighted(btn, 0.8, frame, 0.2, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          tuple(int(c * 0.6) for c in tool_color), 1)

        icon_col  = (10, 10, 10)   if is_active else tool_color
        label_col = (230, 230, 230) if is_active else (160, 160, 170)
        cv2.putText(frame, tool["icon"], (cx - 10, y1 + 38),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, icon_col, 2, cv2.LINE_AA)
        cv2.putText(frame, tool["label"],
                    (cx - len(tool["label"]) * 6, y1 + 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_col, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Tool: {cur_tool.upper()}", (1060, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 220), 1, cv2.LINE_AA)
    cv2.putText(frame,
                "☝ draw  |  ☝+ring: idle  |  🖐 all-up: erase  |  pinch: move canvas  |  Q quit  C clear",
                (6, toolbar_height + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 150), 1, cv2.LINE_AA)


def get_tool_x_ranges():
    return [(TOOLBAR_START_X + i * (BOX_W + BOX_MARGIN),
             TOOLBAR_START_X + i * (BOX_W + BOX_MARGIN) + BOX_W)
            for i in range(len(TOOLS))]

TOOL_X_RANGES = get_tool_x_ranges()


def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
while True:
    success, frame = cap.read()
    if not success:
        continue          # skip bad frames instead of breaking

    frame = cv2.flip(frame, 1)
    # Process MediaPipe at half-res for speed, display at full res
    small  = cv2.resize(frame, (640, 360))
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False          # avoid an internal copy in MediaPipe
    result = hands.process(rgb)
    rgb.flags.writeable = True

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            h, w = 720, 1280   # full-res coords (landmarks are 0-1 normalised)

            # ── Raw fingertip (index = lm 8) ──
            raw_x = int(lm[8].x * w)
            raw_y = int(lm[8].y * h)

            # ── Exponential smoothing (EMA) ──
            smooth_x += (raw_x - smooth_x) / SMOOTH_FACTOR
            smooth_y += (raw_y - smooth_y) / SMOOTH_FACTOR
            x, y = int(smooth_x), int(smooth_y)

            # ── Thumb tip (use RAW index tip for pinch distance — not smoothed) ──
            thumb_x = int(lm[4].x * w)
            thumb_y = int(lm[4].y * h)
            pinch_dist = math.hypot(raw_x - thumb_x, raw_y - thumb_y)

            # ── Finger states ──
            index_up  = finger_up(lm,  8,  6)
            middle_up = finger_up(lm, 12, 10)
            ring_up   = finger_up(lm, 16, 14)
            pinky_up  = finger_up(lm, 20, 18)

            # ── Gesture classification ──
            # Pinch = thumb & index close, middle & ring NOT both up
            is_pinch      = (pinch_dist < PINCH_THRESHOLD) and not (middle_up and ring_up)

            all_fingers   = index_up and middle_up and ring_up and pinky_up
            index_ring    = index_up and ring_up and not middle_up and not pinky_up
            only_index    = index_up and not middle_up and not ring_up and not pinky_up and not is_pinch

            # ── Toolbar hit-test ──
            if y < toolbar_height:
                for idx, (tx1, tx2) in enumerate(TOOL_X_RANGES):
                    if tx1 < x < tx2:
                        tn = TOOLS[idx]["name"]
                        if tn == "clear":
                            canvas[:] = 0
                            sparkles.clear()
                        else:
                            current_tool = tn
                        break

            # ═══════════════════════════════════
            #  PINCH → MOVE canvas
            # ═══════════════════════════════════
            if is_pinch and y >= toolbar_height:
                # Use raw (unsmoothed) midpoint so the grab point matches fingertips exactly
                pinch_cx = (raw_x + thumb_x) // 2
                pinch_cy = (raw_y + thumb_y) // 2

                # On the very first pinch frame: SNAP smooth vars to real position
                # (avoids giant dx/dy from EMA crawling up from 0)
                if prev_pinch_x == 0 and prev_pinch_y == 0:
                    smooth_px, smooth_py = float(pinch_cx), float(pinch_cy)
                else:
                    smooth_px += (pinch_cx - smooth_px) / SMOOTH_PINCH
                    smooth_py += (pinch_cy - smooth_py) / SMOOTH_PINCH

                scx, scy = int(smooth_px), int(smooth_py)

                # Visual
                cv2.line(frame, (raw_x, raw_y), (thumb_x, thumb_y), (255, 200, 50), 2)
                cv2.circle(frame, (scx, scy), 16, (255, 200, 50), cv2.FILLED)
                cv2.circle(frame, (scx, scy), 16, (255, 255, 255), 2)
                cv2.putText(frame, "MOVE", (scx - 24, scy - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 200), 1, cv2.LINE_AA)

                # Apply delta only when we have a valid previous pinch point
                if prev_pinch_x != 0 and prev_pinch_y != 0:
                    dx = scx - prev_pinch_x
                    dy = scy - prev_pinch_y
                    # Clamp delta to avoid runaway shifts on glitchy frames
                    dx = max(-80, min(80, dx))
                    dy = max(-80, min(80, dy))
                    M  = np.float32([[1, 0, dx], [0, 1, dy]])
                    canvas = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]))

                prev_pinch_x, prev_pinch_y = scx, scy
                prev_x, prev_y = 0, 0

            # ═══════════════════════════════════
            #  DRAW — only index finger ✍️
            # ═══════════════════════════════════
            elif only_index and y >= toolbar_height:
                prev_pinch_x, prev_pinch_y = 0, 0
                smooth_px, smooth_py = 0.0, 0.0

                # Glowing cursor dot (drawn at raw pos for zero-lag feel)
                cv2.circle(frame, (raw_x, raw_y), 12, (0, 255, 180), cv2.FILLED)
                cv2.circle(frame, (raw_x, raw_y), 12, (255, 255, 255), 2)
                cv2.circle(frame, (raw_x, raw_y),  4, (255, 255, 255), cv2.FILLED)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                if current_tool == "eraser":
                    draw_color = (0, 0, 0)
                    thickness  = 40
                else:
                    draw_color = DRAW_COLORS.get(current_tool, (255, 100, 30))
                    thickness  = 10

                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, thickness, cv2.LINE_AA)
                prev_x, prev_y = x, y

                # ✨ Sparkles only when painting (not erasing)
                if current_tool != "eraser":
                    spawn_sparkles(x, y)

            # ═══════════════════════════════════
            #  ERASE — all 4 fingers up 🖐️
            # ═══════════════════════════════════
            elif all_fingers and y >= toolbar_height:
                prev_pinch_x, prev_pinch_y = 0, 0
                smooth_px, smooth_py = 0.0, 0.0

                cv2.circle(frame, (raw_x, raw_y), 30, (80, 80, 255), cv2.FILLED)
                cv2.circle(frame, (raw_x, raw_y), 30, (255, 255, 255), 3)
                cv2.putText(frame, "ERASE", (raw_x - 32, raw_y - 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 60, cv2.LINE_AA)
                prev_x, prev_y = x, y

            # ═══════════════════════════════════
            #  IDLE — index + ring  (do nothing)
            # ═══════════════════════════════════
            elif index_ring and y >= toolbar_height:
                prev_pinch_x, prev_pinch_y = 0, 0
                smooth_px, smooth_py = 0.0, 0.0
                prev_x, prev_y = 0, 0

                cv2.circle(frame, (raw_x, raw_y), 10, (180, 180, 180), cv2.FILLED)
                cv2.circle(frame, (raw_x, raw_y), 10, (255, 255, 255), 1)
                cv2.putText(frame, "IDLE", (raw_x - 18, raw_y - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            else:
                prev_pinch_x, prev_pinch_y = 0, 0
                smooth_px, smooth_py = 0.0, 0.0
                prev_x, prev_y = 0, 0

            # Draw skeleton
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 220, 150), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 150, 255), thickness=2)
            )

    else:
        prev_x, prev_y = 0, 0
        prev_pinch_x, prev_pinch_y = 0, 0
        smooth_px, smooth_py = 0.0, 0.0

    # ─────────────────────────────────────────
    # Composite canvas → frame (fully opaque)
    # ─────────────────────────────────────────
    gray  = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, m  = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    inv_m = cv2.bitwise_not(m)

    output = cv2.add(
        cv2.bitwise_and(frame,  frame,  mask=inv_m),  # camera where no ink
        cv2.bitwise_and(canvas, canvas, mask=m)        # ink pixels fully opaque
    )

    # ✨ Batch-render sparkles (single addWeighted for all)
    update_and_draw_sparkles(output)

    # Toolbar last (always on top)
    draw_toolbar(output, current_tool)

    cv2.imshow("AR Smart Board", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas[:] = 0
        sparkles.clear()

cap.release()
cv2.destroyAllWindows()
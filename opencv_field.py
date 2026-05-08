"""
SAOT Phase 2 - OpenCV Interactive Field
Draws a 2D football field, lets you drag 3 players,
and displays the offside line + ML verdict in real time.
"""

import cv2
import numpy as np
from detector_bridge import CoordinateBridge, MLOffsideJudge

WIN_W, WIN_H = 1100, 730
PANEL_H = 110
FIELD_X0, FIELD_Y0 = 60, PANEL_H
FIELD_X1, FIELD_Y1 = 1040, WIN_H - 10

C_GRASS_DARK  = (34,  100,  34)
C_GRASS_LIGHT = (42,  120,  42)
C_LINE        = (255, 255, 255)
C_OFFSIDE     = (0,    30, 220)   # red
C_ONSIDE      = (50,  200,  50)   # green
C_SHADOW      = (0,     0,   0)

C_PASSER    = (50,  200, 255)   # yellow
C_TEAMMATE  = (0,   180, 255)   # orange
C_DEFENDER  = (220,  50,  50)   # blue

PLAYER_RADIUS = 18
DRAG_THRESHOLD = PLAYER_RADIUS + 6

DEFAULT_POSITIONS = {
    "passer":   (50.0, 50.0),
    "teammate": (68.0, 35.0),
    "defender": (65.0, 50.0),
}


def draw_football_field(canvas: np.ndarray, bridge: CoordinateBridge):
    x0, y0, x1, y1 = bridge.x0, bridge.y0, bridge.x1, bridge.y1
    fw, fh = x1 - x0, y1 - y0

    stripe_count = 10
    stripe_w = fw // stripe_count
    for i in range(stripe_count):
        sx = x0 + i * stripe_w
        ex = min(sx + stripe_w, x1)
        color = C_GRASS_DARK if i % 2 == 0 else C_GRASS_LIGHT
        cv2.rectangle(canvas, (sx, y0), (ex, y1), color, -1)

    def fp(fx, fy):
        return bridge.field_to_pixel(fx, fy)

    cv2.rectangle(canvas, (x0, y0), (x1, y1), C_LINE, 2)
    cv2.line(canvas, fp(50, 0), fp(50, 100), C_LINE, 2)
    cx, cy = fp(50, 50)
    r = int(fw * 0.09)
    cv2.circle(canvas, (cx, cy), r, C_LINE, 2)
    cv2.circle(canvas, (cx, cy), 3, C_LINE, -1)

    for side_x, gx0, gx1 in [(0, 16.5, 83.5), (100, 16.5, 83.5)]:
        pa_depth = 16.5 if side_x == 0 else -16.5
        pts = [fp(side_x, gx0), fp(side_x + pa_depth, gx0),
               fp(side_x + pa_depth, gx1), fp(side_x, gx1)]
        for i in range(4):
            cv2.line(canvas, pts[i], pts[(i+1) % 4], C_LINE, 1)

    for side_x in [0, 100]:
        ga_depth = 5.5 if side_x == 0 else -5.5
        pts = [fp(side_x, 36.8), fp(side_x + ga_depth, 36.8),
               fp(side_x + ga_depth, 63.2), fp(side_x, 63.2)]
        for i in range(4):
            cv2.line(canvas, pts[i], pts[(i+1) % 4], C_LINE, 1)

    for side_x in [0, 100]:
        gd = -3 if side_x == 0 else 3
        pts = [fp(side_x, 44.8), fp(side_x + gd, 44.8),
               fp(side_x + gd, 55.2), fp(side_x, 55.2)]
        for i in range(4):
            cv2.line(canvas, pts[i], pts[(i+1) % 4], C_LINE, 2)

    for fx, fy in [(0, 0), (100, 0), (0, 100), (100, 100)]:
        cv2.circle(canvas, fp(fx, fy), int(fw * 0.01), C_LINE, 1)

    mid_y = y0 + fh // 2
    arr_x = x0 + 15
    cv2.arrowedLine(canvas, (arr_x, mid_y + 20), (arr_x, mid_y - 20),
                    (180, 180, 180), 1, tipLength=0.4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for val in [0, 25, 50, 75, 100]:
        px, _ = bridge.field_to_pixel(val, 0)
        cv2.putText(canvas, str(val), (px - 8, y0 - 5),
                    font, 0.35, (180, 180, 180), 1)


def draw_player(canvas, px, py, color, label, role, is_dragging=False):
    cv2.circle(canvas, (px, py), PLAYER_RADIUS, color, -1)
    border_color = (255, 255, 255) if is_dragging else (200, 200, 200)
    border_w = 3 if is_dragging else 1
    cv2.circle(canvas, (px, py), PLAYER_RADIUS, border_color, border_w)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, label, (px - 6, py + 5), font, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)

    tw, _ = cv2.getTextSize(role, font, 0.38, 1)[0], 0
    cv2.putText(canvas, role, (px - tw[0] // 2 - 2, py + PLAYER_RADIUS + 14),
                font, 0.38, color, 1, cv2.LINE_AA)


def draw_offside_line(canvas, bridge, defender_fx, is_offside):
    lx0, ly0, lx1, ly1 = bridge.offside_line_pixels(defender_fx)
    color = C_OFFSIDE if is_offside else C_ONSIDE
    alpha = 0.55

    overlay = canvas.copy()
    cv2.line(overlay, (lx0, ly0), (lx1, ly1), color, 3)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    dash_h = 12
    y = ly0
    while y < ly1:
        cv2.line(canvas, (lx0, y), (lx1, min(y + dash_h, ly1)), color, 2)
        y += dash_h * 2

    arrow_size = 8
    for ay in [ly0 + 2, ly1 - 2]:
        pts = np.array([[lx0, ay], [lx0 - arrow_size, ay - arrow_size],
                        [lx0 + arrow_size, ay - arrow_size]], np.int32)
        if ay > ly0 + 5:
            pts = np.array([[lx0, ay], [lx0 - arrow_size, ay + arrow_size],
                            [lx0 + arrow_size, ay + arrow_size]], np.int32)
        cv2.fillPoly(canvas, [pts], color)


def draw_verdict_panel(canvas, verdict: dict, positions: dict, bridge: CoordinateBridge):
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX

    is_offside = verdict["is_offside"]
    color      = C_OFFSIDE if is_offside else C_ONSIDE

    cv2.rectangle(canvas, (0, 0), (WIN_W, PANEL_H), (18, 20, 28), -1)
    # Thin colored accent line at bottom of panel
    cv2.line(canvas, (0, PANEL_H - 2), (WIN_W, PANEL_H - 2), color, 2)

    SEP1 = 310
    SEP2 = 720
    sep_color = (50, 55, 65)
    cv2.line(canvas, (SEP1, 10), (SEP1, PANEL_H - 10), sep_color, 1)
    cv2.line(canvas, (SEP2, 10), (SEP2, PANEL_H - 10), sep_color, 1)

    label = "OFFSIDE" if is_offside else "ONSIDE"
    cv2.putText(canvas, label, (22, 72), font_bold, 1.6, color, 2, cv2.LINE_AA)
    sub = "Illegal position" if is_offside else "Legal position"
    cv2.putText(canvas, sub, (22, 95), font, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    cx = SEP1 + 14

    cv2.putText(canvas, "CONFIDENCE", (cx, 28), font, 0.40, (130, 130, 130), 1)
    conf   = verdict["confidence"]
    bar_x  = cx
    bar_y  = 36
    bar_w  = SEP2 - SEP1 - 90
    bar_h  = 20
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (45, 45, 50), -1)
    cv2.rectangle(canvas, (bar_x, bar_y),
                  (bar_x + int(bar_w * conf), bar_y + bar_h), color, -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (90, 90, 90), 1)
    cv2.putText(canvas, f"{conf*100:.1f}%",
                (bar_x + bar_w + 8, bar_y + 15),
                font, 0.55, color, 1, cv2.LINE_AA)

    xd = verdict["x_diff"]
    xd_color = C_OFFSIDE if xd > 0 else C_ONSIDE
    cv2.putText(canvas, "X DIFF", (cx, 77), font, 0.40, (130, 130, 130), 1)
    xd_str = f"{xd:+.2f} m"
    cv2.putText(canvas, xd_str, (cx, 98), font_bold, 0.65, xd_color, 1, cv2.LINE_AA)
    arrow = ">> attacker ahead of defender" if xd > 0 else "<< attacker behind defender"
    cv2.putText(canvas, arrow, (cx + 110, 98), font, 0.34, (110, 110, 110), 1)

    rx = SEP2 + 14

    tm = positions["teammate"]
    df = positions["defender"]

    players = [
        ("TM", tm, C_TEAMMATE),
        ("DEF", df, C_DEFENDER),
    ]
    row_y = 26
    for tag, pos, pcol in players:
        cv2.putText(canvas, f"{tag}", (rx, row_y), font_bold, 0.45, pcol, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"x={pos[0]:5.1f}  y={pos[1]:5.1f}",
                    (rx + 38, row_y), font, 0.40, (170, 170, 170), 1)
        row_y += 22

    controls = "[R] Reset    [Q / ESC] Exit"
    cv2.putText(canvas, controls, (rx, 97), font, 0.37, (85, 85, 95), 1)


def draw_offside_zone(canvas, bridge, defender_fx, teammate_pixel, is_offside):
    """Shade the offside zone (area beyond defender toward opponent goal)."""
    if not is_offside:
        return
    lx, _ = bridge.field_to_pixel(defender_fx, 0)
    rx = bridge.x1
    overlay = canvas.copy()
    cv2.rectangle(overlay, (lx, bridge.y0), (rx, bridge.y1),
                  (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)


def draw_pass_arrow(canvas, passer_px, teammate_px, is_offside):
    color = (0, 80, 200) if is_offside else (0, 180, 80)
    cv2.arrowedLine(canvas, passer_px, teammate_px, (30, 30, 30), 4,
                    tipLength=0.15)
    cv2.arrowedLine(canvas, passer_px, teammate_px, color, 2,
                    tipLength=0.15)


class SAOTApp:

    def __init__(self, judge: MLOffsideJudge):
        self.judge = judge
        self.bridge = CoordinateBridge((FIELD_X0, FIELD_Y0, FIELD_X1, FIELD_Y1))

        # Field positions (0-100 scale)
        self.positions = dict(DEFAULT_POSITIONS)

        # Drag state
        self.dragging = None   # "passer" | "teammate" | "defender" | None
        self.drag_offset = (0, 0)

        # Latest verdict
        self.verdict = self._compute_verdict()

        cv2.namedWindow("SAOT - Offside Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SAOT - Offside Detection", WIN_W, WIN_H)
        cv2.setMouseCallback("SAOT - Offside Detection", self._mouse_callback)

    def _compute_verdict(self) -> dict:
        return self.judge.judge(
            self.positions["teammate"],
            self.positions["defender"],
        )

    def _pixel_of(self, name: str) -> tuple[int, int]:
        return self.bridge.field_to_pixel(*self.positions[name])

    def _nearest_player(self, mx, my):
        best, best_dist = None, DRAG_THRESHOLD
        for name in ["teammate", "defender"]:
            px, py = self._pixel_of(name)
            d = ((mx - px) ** 2 + (my - py) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best = d, name
        return best

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._nearest_player(x, y)
            if hit:
                self.dragging = hit
                px, py = self._pixel_of(hit)
                self.drag_offset = (px - x, py - y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            nx, ny = x + self.drag_offset[0], y + self.drag_offset[1]
            fx, fy = self.bridge.pixel_to_field(nx, ny)
            self.positions[self.dragging] = (fx, fy)
            self.verdict = self._compute_verdict()

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    def _render(self) -> np.ndarray:
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 20)

        draw_football_field(canvas, self.bridge)

        tm_px = self._pixel_of("teammate")
        df_px = self._pixel_of("defender")
        ps_px = self._pixel_of("passer")

        # Offside zone shading
        draw_offside_zone(canvas, self.bridge,
                          self.positions["defender"][0],
                          tm_px, self.verdict["is_offside"])

        # Offside line
        draw_offside_line(canvas, self.bridge,
                          self.positions["defender"][0],
                          self.verdict["is_offside"])

        # Pass arrow (visual only, not used for prediction)
        draw_pass_arrow(canvas, ps_px, tm_px, self.verdict["is_offside"])

        # Players
        draw_player(canvas, *df_px, C_DEFENDER, "DEF", "Defender",
                    self.dragging == "defender")
        draw_player(canvas, *ps_px, C_PASSER, "PS", "Passer",
                    self.dragging == "passer")
        draw_player(canvas, *tm_px, C_TEAMMATE, "TM", "Teammate",
                    self.dragging == "teammate")

        # Verdict panel
        draw_verdict_panel(canvas, self.verdict, self.positions, self.bridge)

        return canvas

    def run(self):
        print("\n[SAOT Phase 2] OpenCV field running.")
        print("  Drag players to move them. R=Reset, Q/ESC=Quit\n")
        print(f"  {'Player':<12} {'Field X':>8} {'Field Y':>8} {'Verdict':>12} {'Conf':>8}")
        print("  " + "-" * 52)

        prev_verdict = None
        while True:
            frame = self._render()
            cv2.imshow("SAOT - Offside Detection", frame)

            # Print to console on verdict change
            v = self.verdict
            if prev_verdict != v["is_offside"]:
                tm = self.positions["teammate"]
                df = self.positions["defender"]
                arrow = "🚩" if v["is_offside"] else "✅"
                print(f"  {arrow} {v['label']:<10}  "
                      f"TM({tm[0]:5.1f},{tm[1]:5.1f})  "
                      f"DEF({df[0]:5.1f},{df[1]:5.1f})  "
                      f"x_diff={v['x_diff']:+.2f}  "
                      f"conf={v['confidence']*100:.1f}%")
                prev_verdict = v["is_offside"]

            key = cv2.waitKey(16) & 0xFF
            if key in [ord('q'), 27]:
                break
            elif key == ord('r'):
                self.positions = dict(DEFAULT_POSITIONS)
                self.verdict = self._compute_verdict()
                prev_verdict = None
                print("  [R] Positions reset to default.")

        cv2.destroyAllWindows()
        print("\n[SAOT] Session ended.")
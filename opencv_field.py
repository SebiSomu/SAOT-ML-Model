"""
SAOT Phase 3 - Multi-Camera 3D Field with Stickmen
=====================================================
3 cameras, one active at a time, switch with arrow keys or on-screen buttons:
  - Camera 0: Bird View  (top-down)   → drag & drop enabled
  - Camera 1: Left Side  (left sideline looking right)
  - Camera 2: Right Side (right sideline looking left)

Players are rendered as stickmen. Offside line drawn only when offside exists.
Field coords: x in [0,100] (depth), y in [0,100] (width), z=0 (ground).
"""

import cv2
import numpy as np
import math
from detector_bridge import CoordinateBridge, MLOffsideJudge

# ── Window ────────────────────────────────────────────────
WIN_W, WIN_H = 1200, 750
PANEL_H = 100         # top HUD panel
FIELD_Y0 = PANEL_H
FIELD_Y1 = WIN_H - 50  # bottom nav bar height = 50
NAV_H = WIN_H - FIELD_Y1

# ── Colors (BGR) ──────────────────────────────────────────
C_BG          = (12,  14,  18)
C_PANEL_BG    = (18,  20,  28)
C_NAV_BG      = (14,  16,  22)
C_GRASS_D     = (30,  90,  30)
C_GRASS_L     = (38, 110,  38)
C_LINE        = (230, 230, 230)
C_OFFSIDE     = (0,   40, 220)   # red
C_ONSIDE      = (50, 200,  50)   # green
C_SEP         = (50,  55,  65)

# Stickman colors
C_TEAMMATE  = (0,  170, 255)   # orange
C_DEFENDER  = (220, 50,  50)   # blue
C_PASSER    = (50, 200, 255)   # yellow (visual only in bird view)

# ── Field 3D dimensions (meters, mapped to 0-100 coords) ──
# x: 0=own goal → 100=opponent goal  (105m real)
# y: 0=left → 100=right              (68m real)
# z: always 0 (ground level)

CAMERA_NAMES = ["Bird View", "Left Camera", "Right Camera"]
N_CAMERAS = 3

# Default player positions (field coords)
DEFAULT_POSITIONS = {
    "teammate": (68.0, 35.0),
    "defender": (65.0, 50.0),
    "passer":   (50.0, 50.0),
}

PLAYER_RADIUS_BV = 14   # bird view circle radius (px)
DRAG_THRESHOLD   = PLAYER_RADIUS_BV + 8


# ══════════════════════════════════════════════════════════
# 3D → 2D Projection helpers
# ══════════════════════════════════════════════════════════

def project_bird_view(fx, fy, rect):
    """Top-down orthographic projection onto rect (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = rect
    px = int(x0 + fx / 100 * (x1 - x0))
    py = int(y0 + fy / 100 * (y1 - y0))
    return px, py


def project_side_camera(fx, fy, fz_head, rect, look_from_left=True):
    """
    Side camera: looks along the Y axis.
    - Horizontal axis: x (field depth 0-100)
    - Vertical axis:   z (height, 0=ground)
    - Depth cue:       y (distance from camera → scale)

    look_from_left=True  → camera on left sideline (y=0), looks right (+y)
    look_from_left=False → camera on right sideline (y=100), looks left (-y)
    """
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0

    # Distance from camera (y axis)
    dist = fy if look_from_left else (100 - fy)
    dist = max(dist, 1)

    # Perspective scale: closer = bigger
    scale = 80 / (dist + 10)

    # Horizontal: map fx to screen x (flip if right camera)
    if look_from_left:
        sx = x0 + int(fx / 100 * w)
    else:
        sx = x0 + int((1 - fx / 100) * w)

    # Vertical: z=0 → bottom of field area, z=fz_head → up
    ground_y = y1 - int(h * 0.08)
    sy = ground_y - int(fz_head * scale * h * 0.012)

    return sx, sy, scale


# ══════════════════════════════════════════════════════════
# Stickman drawing
# ══════════════════════════════════════════════════════════

def draw_stickman_bird_view(canvas, px, py, color, label, is_dragging=False):
    """
    Bird view: realistic top-down stickman.
    Seen from directly above — head (circle), torso (rectangle),
    arms and legs spread outward like a star.
    """
    S = PLAYER_RADIUS_BV   # base scale ~14px

    # ── Shadow (ellipse slightly offset) ──────────────────
    cv2.ellipse(canvas, (px + 3, py + 3), (S, int(S * 0.6)),
                0, 0, 360, (0, 0, 0), -1)

    # ── Legs (bottom of body, spread down-left / down-right) ──
    leg = int(S * 0.95)
    lw  = max(int(S * 0.18), 2)
    cv2.line(canvas, (px - int(S*0.25), py + int(S*0.3)),
             (px - int(S*0.45), py + leg), color, lw)
    cv2.line(canvas, (px + int(S*0.25), py + int(S*0.3)),
             (px + int(S*0.45), py + leg), color, lw)

    # ── Arms (spread left and right from torso middle) ────
    arm = int(S * 0.9)
    aw  = max(int(S * 0.15), 2)
    cv2.line(canvas, (px - int(S*0.35), py),
             (px - arm, py - int(S*0.1)), color, aw)
    cv2.line(canvas, (px + int(S*0.35), py),
             (px + arm, py - int(S*0.1)), color, aw)

    # ── Torso (filled rounded rectangle) ──────────────────
    tw, th = int(S * 0.55), int(S * 0.75)
    cv2.ellipse(canvas, (px, py), (tw, th), 0, 0, 360, color, -1)

    # ── Head (circle at top of body) ──────────────────────
    head_r = max(int(S * 0.42), 4)
    head_cy = py - int(S * 0.72)
    skin = (min(color[0] + 60, 255),
            min(color[1] + 50, 255),
            min(color[2] + 40, 255))
    cv2.circle(canvas, (px, head_cy), head_r, skin, -1)
    cv2.circle(canvas, (px, head_cy), head_r, color, 1)

    # ── Jersey number dot (tiny circle on torso) ──────────
    cv2.circle(canvas, (px, py), max(int(S * 0.15), 2), (255,255,255), -1)

    # ── Drag highlight ring ────────────────────────────────
    if is_dragging:
        cv2.circle(canvas, (px, py), S + 5, (255, 255, 255), 2)

    # ── Label above head ──────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    tw_px = cv2.getTextSize(label, font, 0.40, 1)[0][0]
    lbl_y = head_cy - head_r - 4
    # small background pill
    cv2.rectangle(canvas,
                  (px - tw_px // 2 - 3, lbl_y - 11),
                  (px + tw_px // 2 + 3, lbl_y + 2),
                  (20, 20, 20), -1)
    cv2.putText(canvas, label, (px - tw_px // 2, lbl_y),
                font, 0.40, color, 1, cv2.LINE_AA)


def draw_stickman_side_view(canvas, sx, sy, scale, color, label):
    """
    Side-view stickman with perspective scale.
    Similar style to bird view: filled body parts, detailed limbs.
    """
    s = max(scale, 0.5)
    head_r  = max(int(8  * s), 4)
    torso_w = max(int(10 * s), 4)
    torso_h = max(int(22 * s), 8)
    leg     = max(int(20 * s), 7)
    arm_len = max(int(14 * s), 5)
    neck    = max(int(4  * s), 2)

    # Key points (from ground up)
    foot_y   = sy
    hip_y    = sy - leg
    torso_y  = hip_y - torso_h
    neck_y   = torso_y - neck
    head_cy  = neck_y - head_r

    cx = sx   # center x

    # Shadow on ground
    cv2.ellipse(canvas, (cx, foot_y), (head_r + 2, max(3, head_r // 3)),
                0, 0, 360, (0, 0, 0), -1)

    # Legs (two lines from hip to feet, slightly spread)
    spread = max(int(5 * s), 3)
    cv2.line(canvas, (cx, hip_y), (cx - spread, foot_y), color, max(int(2*s), 1))
    cv2.line(canvas, (cx, hip_y), (cx + spread, foot_y), color, max(int(2*s), 1))

    # Torso (filled rounded rectangle)
    cv2.ellipse(canvas, (cx, torso_y + torso_h//2), (torso_w, torso_h//2),
                0, 0, 360, color, -1)

    # Arms (from upper torso, angled down slightly)
    arm_y = torso_y + max(int(6 * s), 3)
    cv2.line(canvas, (cx, arm_y), (cx - arm_len, arm_y + max(int(4*s),2)),
             color, max(int(2*s), 1))
    cv2.line(canvas, (cx, arm_y), (cx + arm_len, arm_y + max(int(4*s),2)),
             color, max(int(2*s), 1))

    # Head (circle)
    cv2.circle(canvas, (cx, head_cy), head_r, color, -1)
    cv2.circle(canvas, (cx, head_cy), head_r, (200, 200, 200), 1)

    # Jersey number dot (tiny circle on torso)
    cv2.circle(canvas, (cx, torso_y + torso_h//2), max(int(2 * s), 2), (255, 255, 255), -1)

    # Label above head
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.28, min(0.5, 0.32 * s))
    tw = cv2.getTextSize(label, font, fs, 1)[0][0]
    cv2.putText(canvas, label, (cx - tw // 2, head_cy - head_r - 3),
                font, fs, color, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════
# Field drawing per camera
# ══════════════════════════════════════════════════════════

def draw_field_bird_view(canvas, rect):
    x0, y0, x1, y1 = rect
    fw, fh = x1 - x0, y1 - y0

    # Grass stripes
    stripes = 10
    sw = fw // stripes
    for i in range(stripes):
        sx = x0 + i * sw
        ex = min(sx + sw, x1)
        cv2.rectangle(canvas, (sx, y0), (ex, y1),
                      C_GRASS_D if i % 2 == 0 else C_GRASS_L, -1)

    def fp(fx, fy):
        return project_bird_view(fx, fy, rect)

    # Boundary
    cv2.rectangle(canvas, (x0, y0), (x1, y1), C_LINE, 2)
    # Halfway line
    cv2.line(canvas, fp(50, 0), fp(50, 100), C_LINE, 2)
    # Center circle
    cx, cy = fp(50, 50)
    cv2.circle(canvas, (cx, cy), int(fw * 0.09), C_LINE, 2)
    cv2.circle(canvas, (cx, cy), 3, C_LINE, -1)
    # Penalty areas
    for sx_f, depth in [(0, 16.5), (100, -16.5)]:
        pts = [fp(sx_f, 16.5), fp(sx_f + depth, 16.5),
               fp(sx_f + depth, 83.5), fp(sx_f, 83.5)]
        for i in range(4):
            cv2.line(canvas, pts[i], pts[(i+1)%4], C_LINE, 1)
    # Goals
    for sx_f, gd in [(0, -3), (100, 3)]:
        pts = [fp(sx_f, 44), fp(sx_f + gd, 44),
               fp(sx_f + gd, 56), fp(sx_f, 56)]
        for i in range(4):
            cv2.line(canvas, pts[i], pts[(i+1)%4], C_LINE, 2)

    # Axis labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for v in [0, 25, 50, 75, 100]:
        px, _ = fp(v, 0)
        cv2.putText(canvas, str(v), (px - 8, y0 - 4),
                    font, 0.32, (150, 150, 150), 1)


def draw_field_side_view(canvas, rect, look_from_left=True,
                         focus_x=65.0, focus_y=50.0, is_offside=False):
    """
    VAR-style side camera zoomed on the offside zone.
    When is_offside=True: entire view goes blue, players drawn in white.
    """
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0

    X_WINDOW = 18.0
    x_min = focus_x - X_WINDOW
    x_max = focus_x + X_WINDOW

    def field_to_sx(fx):
        t = (fx - x_min) / (x_max - x_min)
        if not look_from_left:
            t = 1.0 - t
        return x0 + int(np.clip(t, 0, 1) * w)

    ground_frac = 0.72
    ground_y = y0 + int(h * ground_frac)

    if is_offside:
        # ── OFFSIDE MODE: full blue field ─────────────────
        # Sky: deep blue gradient
        sky_top    = (120, 20, 10)
        sky_bottom = (160, 40, 20)
        for row in range(y0, ground_y):
            t = (row - y0) / max(ground_y - y0, 1)
            b = int(sky_top[0] + (sky_bottom[0] - sky_top[0]) * t)
            g = int(sky_top[1] + (sky_bottom[1] - sky_top[1]) * t)
            r = int(sky_top[2] + (sky_bottom[2] - sky_top[2]) * t)
            cv2.line(canvas, (x0, row), (x1, row), (b, g, r), 1)

        # Grass: dark blue stripes
        n_stripes = 8
        for i in range(n_stripes):
            sy_top = ground_y + int((y1 - ground_y) * i / n_stripes)
            sy_bot = ground_y + int((y1 - ground_y) * (i + 1) / n_stripes)
            t = i / n_stripes
            dark  = (int(100 + 30 * t), int(15 + 5 * t), int(10 + 5 * t))
            light = (int(130 + 30 * t), int(25 + 5 * t), int(15 + 5 * t))
            cv2.rectangle(canvas, (x0, sy_top), (x1, sy_bot),
                          dark if i % 2 == 0 else light, -1)
        # Horizon line white
        cv2.line(canvas, (x0, ground_y), (x1, ground_y), (255, 255, 255), 2)
        cv2.line(canvas, (x0, y1 - 6),   (x1, y1 - 6),   (255, 255, 255), 2)

    else:
        # ── NORMAL MODE: green field ───────────────────────
        sky_top    = (30, 50, 80)
        sky_bottom = (55, 80, 55)
        for row in range(y0, ground_y):
            t = (row - y0) / max(ground_y - y0, 1)
            r = int(sky_top[0] + (sky_bottom[0] - sky_top[0]) * t)
            g = int(sky_top[1] + (sky_bottom[1] - sky_top[1]) * t)
            b = int(sky_top[2] + (sky_bottom[2] - sky_top[2]) * t)
            cv2.line(canvas, (x0, row), (x1, row), (b, g, r), 1)

        n_stripes = 8
        for i in range(n_stripes):
            sy_top = ground_y + int((y1 - ground_y) * i / n_stripes)
            sy_bot = ground_y + int((y1 - ground_y) * (i + 1) / n_stripes)
            t = i / n_stripes
            dark  = tuple(int(c * (0.65 + 0.35 * t)) for c in C_GRASS_D)
            light = tuple(int(c * (0.65 + 0.35 * t)) for c in C_GRASS_L)
            cv2.rectangle(canvas, (x0, sy_top), (x1, sy_bot),
                          dark if i % 2 == 0 else light, -1)
        cv2.line(canvas, (x0, ground_y), (x1, ground_y), C_LINE, 2)
        cv2.line(canvas, (x0, y1 - 6),   (x1, y1 - 6),   C_LINE, 2)

    # ── Camera label (INVERTED: look_from_left=True → RIGHT cam label) ──
    font = cv2.FONT_HERSHEY_SIMPLEX
    lbl_color = (220, 220, 255) if is_offside else (160, 160, 170)
    cam_lbl = "RIGHT SIDELINE CAM" if look_from_left else "LEFT SIDELINE CAM"
    cv2.putText(canvas, cam_lbl, (x0 + 10, y0 + 22),
                font, 0.46, lbl_color, 1, cv2.LINE_AA)

    vp_x = (x0 + x1) // 2
    vp_y = ground_y - int(h * 0.18)

    return ground_y, vp_y, vp_x, field_to_sx, x_min, x_max


# ══════════════════════════════════════════════════════════
# Offside line per camera
# ══════════════════════════════════════════════════════════

def draw_offside_line_bird(canvas, rect, defender_fx, is_offside):
    if not is_offside:
        return
    x0, y0, x1, y1 = rect
    lx = x0 + int(defender_fx / 100 * (x1 - x0))
    # Shaded zone
    overlay = canvas.copy()
    cv2.rectangle(overlay, (lx, y0), (x1, y1), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.13, canvas, 0.87, 0, canvas)
    # Dashed line
    color = C_OFFSIDE
    dash, gap = 14, 10
    y = y0
    while y < y1:
        cv2.line(canvas, (lx, y), (lx, min(y + dash, y1)), color, 2)
        y += dash + gap
    # "OFFSIDE LINE" label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "OFFSIDE LINE", (lx + 4, y0 + 18),
                font, 0.38, color, 1, cv2.LINE_AA)


def draw_offside_line_side(canvas, rect, defender_fx, is_offside,
                           look_from_left=True, field_to_sx=None,
                           ground_y=None, vp_y=None, vp_x=None):
    if not is_offside or field_to_sx is None:
        return
    x0, y0, x1, y1 = rect

    lx = field_to_sx(defender_fx)
    # VAR-style blue translucent
    color_main = (255, 120, 40)   # blue-ish (BGR)
    color_edge = (255, 255, 255)  # white edges
    color_glow = (255, 180, 80)   # lighter blue highlight

    # ── Full blue overlay on entire canvas area (sky + grass) ─
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (160, 30, 15), -1)
    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

    # ── 3D Wall (VAR style blue translucent) ───────────────────
    wall_w_ground = 8
    wall_w_vp     = 2

    pts_wall = np.array([
        [lx - wall_w_ground, y1],
        [lx + wall_w_ground, y1],
        [vp_x + wall_w_vp,   vp_y],
        [vp_x - wall_w_vp,   vp_y],
    ], np.int32)

    # Transparent blue fill
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts_wall], color_main)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

    # Inner highlight (fake lighting for 3D)
    inner_pts = np.array([
        [lx - wall_w_ground//2, y1],
        [lx + wall_w_ground//2, y1],
        [vp_x + 1, vp_y],
        [vp_x - 1, vp_y],
    ], np.int32)

    overlay2 = canvas.copy()
    cv2.fillPoly(overlay2, [inner_pts], color_glow)
    cv2.addWeighted(overlay2, 0.25, canvas, 0.75, 0, canvas)

    # Edges (white, sharp)
    cv2.line(canvas, (lx - wall_w_ground, y1), (vp_x - wall_w_vp, vp_y), color_edge, 2)
    cv2.line(canvas, (lx + wall_w_ground, y1), (vp_x + wall_w_vp, vp_y), color_edge, 2)

    # Vertical center line (like VAR)
    cv2.line(canvas, (lx, ground_y), (lx, y1), color_edge, 2)

    # Ground contact (strong white)
    cv2.line(canvas, (lx - wall_w_ground, ground_y),
                    (lx + wall_w_ground, ground_y), color_edge, 3)

    # Top cap (thin white)
    cv2.line(canvas, (vp_x - wall_w_vp, vp_y),
                    (vp_x + wall_w_vp, vp_y), color_edge, 2)

    # ── Label ─────────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_DUPLEX
    lbl_x = lx + wall_w_ground + 6 if look_from_left else lx - 105
    cv2.putText(canvas, "OFFSIDE LINE", (lbl_x, ground_y - 14),
                font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════
# HUD Panel (top)
# ══════════════════════════════════════════════════════════

def draw_hud(canvas, verdict, cam_idx, positions):
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX

    is_offside = verdict["is_offside"]
    color      = C_OFFSIDE if is_offside else C_ONSIDE

    # Background
    cv2.rectangle(canvas, (0, 0), (WIN_W, PANEL_H), C_PANEL_BG, -1)
    cv2.line(canvas, (0, PANEL_H - 2), (WIN_W, PANEL_H - 2), color, 2)

    SEP1, SEP2 = 290, 700
    cv2.line(canvas, (SEP1, 8), (SEP1, PANEL_H - 8), C_SEP, 1)
    cv2.line(canvas, (SEP2, 8), (SEP2, PANEL_H - 8), C_SEP, 1)

    # ── Zone 1: Verdict ──────────────────────────────────
    label = "OFFSIDE" if is_offside else "ONSIDE"
    cv2.putText(canvas, label, (18, 68), font_bold, 1.55, color, 2, cv2.LINE_AA)
    sub = "Illegal position" if is_offside else "Legal position"
    cv2.putText(canvas, sub, (18, 90), font, 0.40, (150, 150, 150), 1, cv2.LINE_AA)

    # ── Zone 2: Confidence + x_diff ──────────────────────
    cx = SEP1 + 14
    conf = verdict["confidence"]
    cv2.putText(canvas, "CONFIDENCE", (cx, 26), font, 0.38, (120, 120, 120), 1)
    bx, by, bw, bh = cx, 33, SEP2 - SEP1 - 100, 18
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (40, 40, 48), -1)
    cv2.rectangle(canvas, (bx, by), (bx + int(bw * conf), by + bh), color, -1)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1)
    cv2.putText(canvas, f"{conf*100:.1f}%", (bx + bw + 8, by + 14),
                font, 0.52, color, 1, cv2.LINE_AA)

    xd = verdict["x_diff"]
    xd_color = C_OFFSIDE if xd > 0 else C_ONSIDE
    cv2.putText(canvas, "X DIFF", (cx, 74), font, 0.38, (120, 120, 120), 1)
    cv2.putText(canvas, f"{xd:+.2f} m", (cx, 93), font_bold, 0.62, xd_color, 1, cv2.LINE_AA)
    hint = ">> ahead of defender" if xd > 0 else "<< behind defender"
    cv2.putText(canvas, hint, (cx + 95, 93), font, 0.32, (100, 100, 100), 1)

    # ── Zone 3: Positions + camera name ──────────────────
    rx = SEP2 + 14
    tm = positions["teammate"]
    df = positions["defender"]
    for tag, pos, pcol, y_off in [("TM",  tm, C_TEAMMATE, 28),
                                   ("DEF", df, C_DEFENDER, 52)]:
        cv2.putText(canvas, tag, (rx, y_off), font_bold, 0.44, pcol, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"x={pos[0]:5.1f}  y={pos[1]:5.1f}",
                    (rx + 38, y_off), font, 0.38, (160, 160, 160), 1)

    # Camera name badge
    cam_name = CAMERA_NAMES[cam_idx]
    cv2.putText(canvas, f"CAM: {cam_name}", (rx, 82), font, 0.40, (130, 130, 140), 1)
    cv2.putText(canvas, "[R] Reset  [Q/ESC] Quit", (rx, PANEL_H - 8),
                font, 0.33, (70, 70, 80), 1)


# ══════════════════════════════════════════════════════════
# Navigation bar (bottom)
# ══════════════════════════════════════════════════════════

def draw_nav_bar(canvas, cam_idx):
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX
    ny = FIELD_Y1
    cv2.rectangle(canvas, (0, ny), (WIN_W, WIN_H), C_NAV_BG, -1)
    cv2.line(canvas, (0, ny), (WIN_W, ny), (45, 48, 58), 1)

    # Camera dots
    n = N_CAMERAS
    dot_spacing = WIN_W // (n + 1)
    dot_y = ny + NAV_H // 2

    # Left arrow button
    arr_l = (50, dot_y)
    cv2.putText(canvas, "< PREV", (arr_l[0] - 20, arr_l[1] + 5),
                font_bold, 0.50, (180, 180, 200), 1, cv2.LINE_AA)

    # Right arrow button
    cv2.putText(canvas, "NEXT >", (WIN_W - 100, dot_y + 5),
                font_bold, 0.50, (180, 180, 200), 1, cv2.LINE_AA)

    # Camera indicator dots
    center_x = WIN_W // 2
    for i in range(n):
        dot_x = center_x + (i - n // 2) * 90
        if i == cam_idx:
            cv2.circle(canvas, (dot_x, dot_y), 8, (200, 200, 220), -1)
            cv2.putText(canvas, CAMERA_NAMES[i],
                        (dot_x - 35, dot_y + 22), font, 0.35, (200, 200, 220), 1)
        else:
            cv2.circle(canvas, (dot_x, dot_y), 5, (70, 70, 85), -1)
            cv2.putText(canvas, CAMERA_NAMES[i],
                        (dot_x - 35, dot_y + 22), font, 0.32, (80, 80, 90), 1)

    # Store button rects for click detection (returned)
    btn_prev = (10, ny + 4, 105, WIN_H - 4)
    btn_next = (WIN_W - 110, ny + 4, WIN_W - 5, WIN_H - 4)
    return btn_prev, btn_next


# ══════════════════════════════════════════════════════════
# Pass arrow
# ══════════════════════════════════════════════════════════

def draw_pass_arrow_bird(canvas, passer_px, teammate_px, is_offside):
    color = C_OFFSIDE if is_offside else C_ONSIDE
    cv2.arrowedLine(canvas, passer_px, teammate_px, (20, 20, 20), 4, tipLength=0.18)
    cv2.arrowedLine(canvas, passer_px, teammate_px, color,       2, tipLength=0.18)


# ══════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════

class SAOTApp3:

    def __init__(self, judge: MLOffsideJudge):
        self.judge   = judge
        self.cam_idx = 0   # active camera index

        self.positions = dict(DEFAULT_POSITIONS)
        self.verdict   = self._compute_verdict()

        # Bird-view field rect
        margin = 55
        self.bv_rect = (margin, FIELD_Y0 + 5,
                        WIN_W - margin, FIELD_Y1 - 5)

        # Side-view field rect (same area, different renderer)
        self.sv_rect = self.bv_rect

        # Drag state (bird view only)
        self.dragging     = None
        self.drag_offset  = (0, 0)
        self.btn_prev     = None
        self.btn_next     = None

        cv2.namedWindow("SAOT - Phase 3", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SAOT - Phase 3", WIN_W, WIN_H)
        cv2.setMouseCallback("SAOT - Phase 3", self._mouse_cb)

    # ── Verdict ──────────────────────────────────────────
    def _compute_verdict(self):
        return self.judge.judge(
            self.positions["teammate"],
            self.positions["defender"],
        )

    # ── Pixel helpers (bird view) ─────────────────────────
    def _bv_pixel(self, name):
        fx, fy = self.positions[name]
        return project_bird_view(fx, fy, self.bv_rect)

    def _bv_field(self, px, py):
        x0, y0, x1, y1 = self.bv_rect
        fx = np.clip((px - x0) / (x1 - x0) * 100, 0, 100)
        fy = np.clip((py - y0) / (y1 - y0) * 100, 0, 100)
        return round(float(fx), 2), round(float(fy), 2)

    def _nearest_bv(self, mx, my):
        best, bd = None, DRAG_THRESHOLD
        for name in ["teammate", "defender", "passer"]:
            px, py = self._bv_pixel(name)
            d = math.hypot(mx - px, my - py)
            if d < bd:
                bd, best = d, name
        return best

    # ── Mouse callback ────────────────────────────────────
    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Nav bar buttons
            if self.btn_prev and self._in_rect(x, y, self.btn_prev):
                self.cam_idx = (self.cam_idx - 1) % N_CAMERAS
                return
            if self.btn_next and self._in_rect(x, y, self.btn_next):
                self.cam_idx = (self.cam_idx + 1) % N_CAMERAS
                return
            # Drag only in bird view
            if self.cam_idx == 0:
                hit = self._nearest_bv(x, y)
                if hit:
                    self.dragging = hit
                    px, py = self._bv_pixel(hit)
                    self.drag_offset = (px - x, py - y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            nx = x + self.drag_offset[0]
            ny = y + self.drag_offset[1]
            self.positions[self.dragging] = self._bv_field(nx, ny)
            self.verdict = self._compute_verdict()

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    @staticmethod
    def _in_rect(x, y, rect):
        x0, y0, x1, y1 = rect
        return x0 <= x <= x1 and y0 <= y <= y1

    # ── Render ────────────────────────────────────────────
    def _render(self):
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        canvas[:] = C_BG

        is_offside = self.verdict["is_offside"]
        tm_pos = self.positions["teammate"]
        df_pos = self.positions["defender"]
        ps_pos = self.positions["passer"]

        if self.cam_idx == 0:
            self._render_bird_view(canvas, is_offside, tm_pos, df_pos, ps_pos)
        elif self.cam_idx == 1:
            # Camera 1 = Right sideline (look_from_left=True renders from right perspective)
            self._render_side_view(canvas, is_offside, tm_pos, df_pos, ps_pos,
                                   look_from_left=False)
        else:
            # Camera 2 = Left sideline
            self._render_side_view(canvas, is_offside, tm_pos, df_pos, ps_pos,
                                   look_from_left=True)

        draw_hud(canvas, self.verdict, self.cam_idx, self.positions)
        self.btn_prev, self.btn_next = draw_nav_bar(canvas, self.cam_idx)
        return canvas

    def _render_bird_view(self, canvas, is_offside, tm_pos, df_pos, ps_pos):
        rect = self.bv_rect
        draw_field_bird_view(canvas, rect)
        draw_offside_line_bird(canvas, rect, df_pos[0], is_offside)

        ps_px = project_bird_view(*ps_pos, rect)
        tm_px = project_bird_view(*tm_pos, rect)
        df_px = project_bird_view(*df_pos, rect)

        draw_pass_arrow_bird(canvas, ps_px, tm_px, is_offside)

        draw_stickman_bird_view(canvas, *df_px, C_DEFENDER, "DEF",
                                self.dragging == "defender")
        draw_stickman_bird_view(canvas, *ps_px, C_PASSER,   "PS",
                                self.dragging == "passer")
        draw_stickman_bird_view(canvas, *tm_px, C_TEAMMATE, "TM",
                                self.dragging == "teammate")

        # Drag hint
        if self.cam_idx == 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            x0, y0, _, y1 = rect
            cv2.putText(canvas, "Drag players with mouse (Bird View only)",
                        (x0 + 4, y1 + 16), font, 0.36, (80, 80, 92), 1)

    def _render_side_view(self, canvas, is_offside, tm_pos, df_pos, ps_pos,
                          look_from_left):
        rect = self.sv_rect
        x0, y0, x1, y1 = rect

        # Focus center = midpoint of TM and DEF
        focus_x = (tm_pos[0] + df_pos[0]) / 2

        # Draw focused field (blue if offside)
        ground_y, vp_y, vp_x, field_to_sx, x_min, x_max = draw_field_side_view(
            canvas, rect, look_from_left, focus_x=focus_x, is_offside=is_offside)

        # 3D offside wall
        draw_offside_line_side(canvas, rect, df_pos[0], is_offside,
                               look_from_left, field_to_sx,
                               ground_y, vp_y, vp_x)

        # ── Stickmen ─────────────────────────────────────────
        ground_h = y1 - ground_y

        # Use team colors: blue for DEF, yellow for PS, orange for TM
        col_tm  = C_TEAMMATE
        col_def = C_DEFENDER
        col_ps  = C_PASSER

        players = [
            (tm_pos,  col_tm,  "TM"),
            (df_pos,  col_def, "DEF"),
            (ps_pos,  col_ps,  "PS"),
        ]

        # Painter's algorithm: farthest first
        if look_from_left:
            players.sort(key=lambda p: -p[0][1])
        else:
            players.sort(key=lambda p:  p[0][1])

        for pos, color, label in players:
            fx, fy = pos
            if fx < x_min - 5 or fx > x_max + 5:
                continue

            sx = field_to_sx(fx)
            dist = fy if look_from_left else (100.0 - fy)
            dist = max(dist, 1.0)
            scale = np.clip(12.0 / (dist + 2.0), 0.30, 1.80)
            depth_frac = np.clip(dist / 100.0, 0, 1)
            sy_ground = int(ground_y + ground_h * 0.85
                            - ground_h * 0.75 * depth_frac)

            draw_stickman_side_view(canvas, sx, sy_ground, scale, color, label)

        # Hint
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Switch to Bird View to drag players  (← / → arrow keys)",
                    (x0 + 4, y1 + 16), font, 0.36, (80, 80, 92), 1)

    # ── Event loop ────────────────────────────────────────
    def run(self):
        print("\n[SAOT Phase 3] Multi-camera field running.")
        print("  ← → arrow keys  or  click PREV/NEXT  to switch camera")
        print("  Drag players only in Bird View | R=Reset | Q/ESC=Quit\n")
        print(f"  {'Camera':<14} {'Verdict':<10} {'Conf':>6}  Positions")
        print("  " + "─" * 60)

        prev_verdict = None
        while True:
            frame = self._render()
            cv2.imshow("SAOT - Phase 3", frame)

            v = self.verdict
            if prev_verdict != v["is_offside"]:
                tm = self.positions["teammate"]
                df = self.positions["defender"]
                icon = "🚩" if v["is_offside"] else "✅"
                print(f"  {icon} {v['label']:<9}  "
                      f"conf={v['confidence']*100:.1f}%  "
                      f"TM({tm[0]:.1f},{tm[1]:.1f})  "
                      f"DEF({df[0]:.1f},{df[1]:.1f})  "
                      f"x_diff={v['x_diff']:+.2f}")
                prev_verdict = v["is_offside"]

            key = cv2.waitKey(16) & 0xFF
            if key in [ord('q'), 27]:
                break
            elif key == ord('r'):
                self.positions = dict(DEFAULT_POSITIONS)
                self.verdict   = self._compute_verdict()
                prev_verdict   = None
                print("  [R] Positions reset.")
            elif key == 81 or key == ord('a'):   # left arrow
                self.cam_idx = (self.cam_idx - 1) % N_CAMERAS
                print(f"  Camera → {CAMERA_NAMES[self.cam_idx]}")
            elif key == 83 or key == ord('d'):   # right arrow
                self.cam_idx = (self.cam_idx + 1) % N_CAMERAS
                print(f"  Camera → {CAMERA_NAMES[self.cam_idx]}")

        cv2.destroyAllWindows()
        print("\n[SAOT Phase 3] Session ended.")
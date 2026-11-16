#!/usr/bin/env python3
"""Recover a photo-booth frame template from a finished strip.

The strip holds the same four photos twice (left column / right column), each
column inside a different frame design. Because the photo pixels are identical
in both columns, anything that differs between a left slot and its right twin
must be a frame decoration (title swash, flower, vine, glow) lying on top of
the photo. That gives a clean separation without guessing by colour:

  1. slot rectangles are fixed (measured from the edge gradients);
  2. left/right difference -> where decorations overlap the photo;
  3. the twin slot supplies the clean photo pixel behind each decoration, so
     the decoration's alpha and un-mixed colour can be solved (difference
     matting) instead of keeping red photo fringes;
  4. everything else inside a slot becomes fully transparent.

Outputs: transparent PNG, chroma-green PNG, and the slot mask.

Usage: extract_template.py SOURCE.jpg OUT_DIR
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

# Slot geometry in source pixels (source is 1920x2860).
SLOT_W, SLOT_H = 825, 604
COL_X = (90, 1005)
ROW_Y = (135, 774, 1414, 2053)
MARGIN = 3          # slot-edge strip polluted by JPEG ringing
REACH = 60          # how far a soft glow may extend from solid decoration
GREEN = (0, 255, 0)  # BGR
# Where a frame casts a soft halo onto the photo: (column, row_y) -> boxes
# (slot-relative x0, y0, x1, y1). Only the two swashes of the left title do.
GLOW = {(0, 135): [(110, 0, 330, 140), (530, 0, 725, 140)]}


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def drop_small(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def interior(mask):
    m = mask.copy()
    m[:MARGIN], m[-MARGIN:], m[:, :MARGIN], m[:, -MARGIN:] = False, False, False, False
    return m


def nearest_values(valid, values):
    """For every pixel, the value at the nearest pixel where `valid` is set."""
    _, labels = cv2.distanceTransformWithLabels(
        (~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(valid)
    lut = np.zeros((labels.max() + 1,) + values.shape[2:], values.dtype)
    lut[labels[ys, xs]] = values[ys, xs]
    return lut[labels]


def solidify(alpha):
    """Mask of a decoration made solid, for a frame whose decorations are red.

    Red roses on the red wall: parts of a flower equal the photo colour and
    would be cut out with it. Decorations only sit along the slot edge as
    compact blobs, so close their outline and fill every small pocket.
    """
    solid = (alpha > 0.5).astype(np.uint8)
    filled = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, disk(3)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats((~filled).astype(np.uint8), connectivity=4)
    pocket = np.zeros(n, bool)
    pocket[1:] = st[1:, cv2.CC_STAT_AREA] < 300
    return filled | pocket[lab]


def matte(side, P, support, uncertain, glow, red_frame):
    """Alpha + un-mixed colour of the decoration lying over `side`.

    P is the clean photo behind it, `support` where this side may be decorated,
    `uncertain` where P had to be inpainted (both frames decorated), `glow`
    where the frame has a soft halo (smooth, mid-strength, never solid).
    `red_frame`: this frame's decorations are red like the photo's wall, so a
    pixel matching the photo proves nothing - keep decorations solid instead.
    """
    S = side.astype(np.float32)
    e = S - P
    mag = np.abs(e).max(axis=2)
    # Solid decoration is either very different from the photo or textured;
    # a soft glow is neither, and must stay semi-transparent.
    gray = S.mean(axis=2)
    tex = cv2.dilate(gray, np.ones((5, 5), np.uint8)) - cv2.erode(gray, np.ones((5, 5), np.uint8))
    core = drop_small(interior(mag > 40) & support & ((mag > 110) | ((tex > 14) & ~glow)), 12)
    if not core.any():
        return np.zeros(mag.shape, np.float32), S
    # Enclosed flat areas are decoration too - unless they show the photo.
    core |= ndimage.binary_fill_holes(core) & (mag > 25)
    # A decoration close to the photo's own colour (red rose on the red wall)
    # leaves ragged gaps; bridge them wherever the pixel is not plain photo.
    closed = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, disk(7)) > 0
    core |= closed & (mag > 8) & support & ~glow
    near = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    inner = cv2.erode(core.astype(np.uint8), disk(1)) > 0
    Dn = nearest_values(inner if inner.any() else core, S)

    # Model the pixel as photo + alpha * (X - photo) and keep whichever X fits
    # best: the neighbouring solid decoration, a white glow, or a black shadow.
    candidates = [Dn, np.full_like(P, 255.0), np.zeros_like(P)]
    halo = glow & support & ~uncertain & (mag > 15) & (mag < 110) & (tex < 10)
    if halo.sum() > 50:
        # One halo colour: on the ray photo -> observed, where it is most neutral.
        t = np.linspace(1, 8, 300)[:, None]
        ray = P[halo].mean(axis=0) + t * e[halo].mean(axis=0)
        ray = ray[(ray.min(axis=1) >= 0) & (ray.max(axis=1) <= 255)]
        halo_colour = ray[np.argmin(ray.max(axis=1) - ray.min(axis=1))]
        candidates.append(np.broadcast_to(halo_colour.astype(np.float32), P.shape).copy())
    best = None
    for X in candidates:
        v = X - P
        ap = np.clip((e * v).sum(axis=2) / np.maximum((v * v).sum(axis=2), 1.0), 0, 1)
        rs = np.abs(e - ap[..., None] * v).max(axis=2)
        if best is None:
            a_proj, resid, Dfit = ap, rs, X.copy()
            best = True
        else:
            better = rs < resid
            a_proj = np.where(better, ap, a_proj)
            resid = np.where(better, rs, resid)
            Dfit = np.where(better[..., None], X, Dfit)
    # Lowest alpha that keeps the un-mixed colour a real colour.
    gamut = np.where(e > 0, e / np.maximum(255 - P, 1), -e / np.maximum(P, 1)).max(axis=2)
    alpha = np.maximum(a_proj, np.clip(gamut, 0, 1))
    # Right next to solid decoration, a pixel the mix model cannot explain is
    # solid too. Where P is only an estimate, demand a larger deviation.
    lo = np.where(uncertain, 22.0, 10.0) + np.where(glow, 40.0, 0.0)
    alpha = np.where(near <= 3, np.maximum(alpha, smoothstep(resid, lo, lo + 16)), alpha)
    alpha[core] = 1
    alpha *= smoothstep(mag, np.where(uncertain, 14.0, 4.0), np.where(uncertain, 28.0, 10.0))
    alpha[(near > REACH) | ~support] = 0
    alpha[alpha > 0.97] = 1
    # Keep only what hangs together with solid decoration (drops noise specks).
    n, lab = cv2.connectedComponents((alpha > 0.05).astype(np.uint8), connectivity=8)
    if n > 1:
        touches = ndimage.maximum(cv2.dilate(core.astype(np.uint8), disk(2)), lab, np.arange(n)) > 0
        touches[0] = False
        alpha[~touches[lab]] = 0

    # Pinholes inside solid decoration (decoration colour == photo colour).
    solid = alpha > 0.5
    holes = ~solid
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
    small = np.zeros(n, bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] < 12
    alpha[small[lab] & holes] = 1

    if red_frame:
        alpha[solidify(alpha) & support] = 1

    # Un-mix the colour so no photo red survives in soft edges / glow.
    safe = np.maximum(alpha, 1e-3)[..., None]
    unmixed = np.clip(P + e / safe, 0, 255)
    w = smoothstep(alpha, 0.15, 0.4)[..., None]
    D = w * unmixed + (1 - w) * Dfit
    D = np.where((glow & (alpha < 1))[..., None], Dfit, D)
    D = np.where((alpha >= 1)[..., None], S, D)

    # Slot-edge strip: both alpha and colour are unreliable there (ringing
    # against two different frames), so continue them from just inside.
    h, w_ = alpha.shape
    yy = np.clip(np.arange(h), MARGIN, h - 1 - MARGIN)
    xx = np.clip(np.arange(w_), MARGIN, w_ - 1 - MARGIN)
    a_in, d_in = alpha[yy][:, xx], D[yy][:, xx]
    strip = ~interior(np.ones_like(solid))
    # ...but a strip pixel that still shows the photo is not decoration.
    a_strip = np.where((a_in > 0.9) & (mag < 8), 0, a_in)
    D = np.where(strip[..., None], np.where((a_strip > 0.9)[..., None], S, d_in), D)
    alpha = np.where(strip, a_strip, alpha)
    if red_frame:  # once more, now including the edge strip
        newly = solidify(alpha) & support & (alpha < 1)
        alpha[newly] = 1
        D = np.where(newly[..., None], S, D)
    return alpha.astype(np.float32), D


def solve_plate(L, R):
    """Clean photo plate for a slot pair plus each side's decoration support."""
    Lf, Rf = L.astype(np.float32), R.astype(np.float32)
    d = np.abs(Lf - Rf).max(axis=2)
    strong = drop_small(interior(d > 24), 20)
    zone = cv2.dilate(strong.astype(np.uint8), disk(REACH)) > 0
    active = cv2.dilate(((d > 4) & zone).astype(np.uint8), disk(2)) > 0
    # Side-neutral photo estimate: fill everything that differs from around it.
    P0 = cv2.inpaint(L, cv2.dilate(active.astype(np.uint8), disk(2)), 6,
                     cv2.INPAINT_TELEA).astype(np.float32)
    dL = cv2.GaussianBlur(np.abs(Lf - P0).max(axis=2), (0, 0), 2.5)
    dR = cv2.GaussianBlur(np.abs(Rf - P0).max(axis=2), (0, 0), 2.5)

    # Which frame(s) own each blob of difference? A side owns it only if a real
    # share of the blob is farther from the photo estimate on that side.
    n, lab = cv2.connectedComponents(cv2.dilate(strong.astype(np.uint8), disk(4)), connectivity=8)
    allowed = [np.zeros(d.shape, bool), np.zeros(d.shape, bool)]
    for i in range(1, n):
        blob = lab == i
        area = int((blob & strong).sum())
        for k, (mine, theirs) in enumerate(((dL, dR), (dR, dL))):
            votes = int((blob & strong & (mine > theirs) & (mine > 30)).sum())
            if votes >= max(15, 0.01 * area):
                allowed[k] |= cv2.dilate(blob.astype(np.uint8), disk(REACH)) > 0

    both = active & allowed[0] & allowed[1] & (np.minimum(dL, dR) > 15)
    # Photo detail that merely defeats the inpainting looks "decorated on both
    # sides" too, but there the two slots agree; real overlaps mostly differ.
    n, lab = cv2.connectedComponents(both.astype(np.uint8), connectivity=8)
    if n > 1:
        differs = ndimage.mean(d > 12, lab, np.arange(n))
        differs[0] = 0
        both = (differs >= 0.3)[lab]
    both = cv2.dilate(both.astype(np.uint8), disk(2)) > 0
    left_owns = active & allowed[0] & ((dL >= dR) | ~allowed[1])
    right_owns = active & allowed[1] & ~left_owns
    sup = [left_owns | both, right_owns | both]
    # Behind a decoration the twin slot shows the photo; where both frames are
    # decorated, fill in from the surrounding known photo.
    comp = np.where(left_owns[..., None], R, L)
    P = comp.astype(np.float32)
    if both.any():
        P = cv2.inpaint(comp, both.astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32)
    uncertain = cv2.dilate(both.astype(np.uint8), disk(2)) > 0
    return P, sup, uncertain


def clean_ring(img, slot_alpha, x0, y0):
    """Repair the thin ring of frame just outside a slot.

    The outermost frame pixels are blended with the photo (anti-aliasing plus
    JPEG ringing), which would leave a red fringe around the hole. Rebuild them
    from the frame pixels directly behind them: the first two rows always, the
    third only where the frame is flat. Nothing is made transparent, and spots
    where a decoration crosses the slot edge are left untouched.
    `slot_alpha` is the alpha map with everything outside the slots zeroed.
    """
    H, W = img.shape[:2]
    x1, y1 = x0 + SLOT_W - 1, y0 + SLOT_H - 1
    src = img.copy()
    for y in range(y0 - MARGIN, y1 + MARGIN + 1):
        inside_y = y0 <= y <= y1
        xs = (list(range(x0 - MARGIN, x0)) + list(range(x1 + 1, x1 + MARGIN + 1))
              if inside_y else range(x0 - MARGIN, x1 + MARGIN + 1))
        for x in xs:
            dx = -1 if x < x0 else (1 if x > x1 else 0)
            dy = -1 if y < y0 else (1 if y > y1 else 0)
            bx = x0 if dx < 0 else (x1 if dx > 0 else x)   # nearest slot pixel
            by = y0 if dy < 0 else (y1 if dy > 0 else y)
            if slot_alpha[max(by - 4, 0):by + 5, max(bx - 4, 0):bx + 5].max() > 0:
                continue  # a decoration crosses the edge here
            depth = max(abs(x - bx), abs(y - by))
            pts = [(by + dy * k, bx + dx * k) for k in (3, 4, 5, 6, 7)]
            if any(not (0 <= py < H and 0 <= px < W) for py, px in pts):
                continue
            if depth <= 2:
                img[y, x] = src[pts[0]]
            else:
                cols = np.array([src[q] for q in pts[1:]], np.int32)
                if (cols.max(axis=0) - cols.min(axis=0)).max() <= 8:
                    img[y, x] = src[pts[1]]


def main(src_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src_path))
    assert img is not None and img.shape[:2] == (2860, 1920), "unexpected source image"
    H, W = img.shape[:2]
    color = img.astype(np.float32)
    alpha = np.ones((H, W), np.float32)
    plate = img.copy()          # original with decorations removed from slots (for checking)
    slot_mask = np.zeros((H, W), np.uint8)

    for y in ROW_Y:
        L = img[y:y + SLOT_H, COL_X[0]:COL_X[0] + SLOT_W]
        R = img[y:y + SLOT_H, COL_X[1]:COL_X[1] + SLOT_W]
        P, sup, uncertain = solve_plate(L, R)
        for k, (x, side) in enumerate(((COL_X[0], L), (COL_X[1], R))):
            glow = np.zeros(P.shape[:2], bool)
            for gx0, gy0, gx1, gy1 in GLOW.get((k, y), ()):
                glow[gy0:gy1, gx0:gx1] = True
            a, D = matte(side, P, sup[k], uncertain, glow, red_frame=(k == 1))
            if k == 0:
                # The pale frame has no saturated red of its own, so any such
                # pixel still standing over a slot is leftover photo wall.
                b_, g_, r_ = (side[..., i].astype(np.int32) for i in range(3))
                wall = (r_ > 80) & (g_ < 70) & (b_ < 80) & (r_ > 2 * np.maximum(g_, b_))
                a[wall & ~glow] = 0
            alpha[y:y + SLOT_H, x:x + SLOT_W] = a
            color[y:y + SLOT_H, x:x + SLOT_W] = D
            plate[y:y + SLOT_H, x:x + SLOT_W] = np.round(P).astype(np.uint8)
            slot_mask[y:y + SLOT_H, x:x + SLOT_W] = 255

    frame = np.clip(color, 0, 255).astype(np.uint8)
    slot_alpha = np.where(slot_mask > 0, alpha, 0)
    for y in ROW_Y:
        for x in COL_X:
            clean_ring(frame, slot_alpha, x, y)

    a8 = np.round(alpha * 255).astype(np.uint8)
    frame[a8 == 0] = 0
    rgba = np.dstack([frame, a8])
    cv2.imwrite(str(out_dir / "template_transparent.png"), rgba)

    af = (a8.astype(np.float32) / 255)[..., None]
    green = np.empty_like(frame)
    green[:] = GREEN
    comp = np.round(frame * af + green * (1 - af)).astype(np.uint8)
    cv2.imwrite(str(out_dir / "template_green.png"), comp)
    cv2.imwrite(str(out_dir / "slot_mask.png"), slot_mask)

    # Self-check 1: no pixel outside the slots lost opacity (no frame cropping).
    assert (a8[slot_mask == 0] == 255).all()
    # Self-check 2: template over the clean photo plate must reproduce the strip.
    recon = np.round(frame * af + plate * (1 - af)).astype(np.int32)
    err = np.abs(recon - img.astype(np.int32)).max(axis=2)
    print("opaque decoration px inside slots:", int(((a8 == 255) & (slot_mask > 0)).sum()))
    print("semi-transparent px inside slots:", int(((a8 > 0) & (a8 < 255)).sum()))
    print("cleared photo px:", int((a8 == 0).sum()))
    print("reconstruction error p99 / max:", int(np.percentile(err, 99)), int(err.max()))
    return rgba, plate


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

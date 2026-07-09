"""
Renders a structured receipt dict (see layout.build_receipt) into a clean,
'fresh off the till' PIL image. The photographed look is applied afterwards
by augment.py.
"""
import os
import random
from PIL import Image, ImageDraw, ImageFont

_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts")
MONO_REGULAR = os.path.join(_FONT_DIR, "dejavu-sans.condensed.ttf")
MONO_BOLD = os.path.join(_FONT_DIR, "dejavu-sans-mono.bold.ttf")
SANS_BOLD = os.path.join(_FONT_DIR, "dejavu-sans-mono.bold.ttf")

PAPER_WIDTH = 420
MARGIN = 22
PAPER_BG = (250, 249, 245)
INK = (25, 25, 25)
FAINT_INK = (90, 90, 90)


def _font(path, size):
    return ImageFont.truetype(path, size)


def _text_w(draw, text, font):
    return draw.textlength(text, font=font)


def _wrap_to_width(draw, text, font, max_width):
    """Greedy word-wrap text to fit max_width, returns list of lines."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if _text_w(draw, trial, font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _money(value):
    return f"{value:,.2f}"


def _dashed_line(draw, y, width=PAPER_WIDTH, margin=MARGIN, color=FAINT_INK):
    x = margin
    dash, gap = 5, 4
    while x < width - margin:
        draw.line([(x, y), (min(x + dash, width - margin), y)], fill=color, width=1)
        x += dash + gap


def _draw_barcode(draw, x, y, w, h, color=INK):
    """Draws a decorative (non-decodable) barcode pattern."""
    cx = x
    rng = random.Random()
    while cx < x + w:
        bar_w = rng.choice([1, 1, 2, 3])
        if rng.random() < 0.55:
            draw.rectangle([cx, y, cx + bar_w, y + h], fill=color)
        cx += bar_w + rng.choice([1, 2])


def render_receipt(receipt):
    style = receipt["_style"]
    accent = tuple(style["accent_color"])

    # Generous height estimate; we crop to actual content afterwards.
    est_height = 700 + len(receipt["items"]) * 34
    img = Image.new("RGB", (PAPER_WIDTH, est_height), PAPER_BG)
    draw = ImageDraw.Draw(img)

    f_title = _font(SANS_BOLD, int(26 * style.get("header_font_scale", 1.0) / 1.4))
    f_subtitle = _font(MONO_BOLD, 13)
    f_body = _font(MONO_REGULAR, 13)
    f_body_bold = _font(MONO_BOLD, 13)
    f_small = _font(MONO_REGULAR, 11)
    f_total = _font(MONO_BOLD, 16)

    y = 20
    store = receipt["store"]

    # --- Header -----------------------------------------------------
    title = store["name"]
    tw = _text_w(draw, title, f_title)
    draw.text(((PAPER_WIDTH - tw) / 2, y), title, font=f_title, fill=accent)
    y += f_title.size + 10

    for line in (store["legal_name"], store["branch"], store["address"], store["phone"]):
        lw = _text_w(draw, line, f_small)
        draw.text(((PAPER_WIDTH - lw) / 2, y), line, font=f_small, fill=FAINT_INK)
        y += 15

    y += 8
    _dashed_line(draw, y)
    y += 12

    # --- Transaction meta --------------------------------------------
    tx = receipt["transaction"]
    meta_lines = [
        f"DATE: {tx['date']}      TIME: {tx['time']}",
        f"TILL: {tx['till_number']}   OP: {tx['cashier']}   RCPT#: {tx['receipt_number']}",
    ]
    for line in meta_lines:
        draw.text((MARGIN, y), line, font=f_small, fill=FAINT_INK)
        y += 15

    y += 6
    _dashed_line(draw, y)
    y += 12

    # --- Items ----------------------------------------------------------
    content_w = PAPER_WIDTH - 2 * MARGIN
    for item in receipt["items"]:
        name = item["name"].upper()
        lines = _wrap_to_width(draw, name, f_body, content_w)
        for ln in lines:
            draw.text((MARGIN, y), ln, font=f_body, fill=INK)
            y += 17

        if item["quantity"] > 1:
            qty_str = f"  {item['quantity']} @ {_money(item['unit_price'])}"
            draw.text((MARGIN, y), qty_str, font=f_small, fill=FAINT_INK)

        right_str = f"{item['vat_code']}    {_money(item['total_price'])}"
        rw = _text_w(draw, right_str, f_body)
        draw.text((PAPER_WIDTH - MARGIN - rw, y), right_str, font=f_body, fill=INK)
        y += 19

    y += 6
    _dashed_line(draw, y)
    y += 12

    # --- Totals ----------------------------------------------------------
    totals = receipt["totals"]

    def _row(label, value_str, font_label=f_body, font_value=None, color=INK):
        nonlocal y
        font_value = font_value or font_label
        vw = _text_w(draw, value_str, font_value)
        draw.text((MARGIN, y), label, font=font_label, fill=color)
        draw.text((PAPER_WIDTH - MARGIN - vw, y), value_str, font=font_value, fill=color)
        y += font_label.size + 7

    _row(f"ITEMS SOLD: {totals['item_count']}", "")
    _row("SUBTOTAL", _money(totals["subtotal"]))
    for vb in totals["vat_breakdown"]:
        _row(f"  VAT {vb['code']} ({vb['label']})", _money(vb["vat_amount"]), f_small, f_small, FAINT_INK)

    y += 4
    _dashed_line(draw, y)
    y += 10
    _row("TOTAL", f"GBP {_money(totals['total'])}", f_total, f_total)
    y += 6

    # --- Payment ----------------------------------------------------------
    pay = receipt["payment"]
    if pay["method"] == "CARD":
        _row(pay["card_type"], _money(pay["amount_authorised"]), f_body)
        _row("CARD ENDING", f"**** {pay['card_last4']}", f_small, f_small, FAINT_INK)
        draw.text((MARGIN, y), "AID: A0000000031010  APPROVED", font=f_small, fill=FAINT_INK)
        y += 16
    else:
        _row("CASH TENDERED", _money(pay["amount_tendered"]), f_body)
        _row("CHANGE", _money(pay["change"]), f_body)

    y += 8
    _dashed_line(draw, y)
    y += 14

    # --- Footer ----------------------------------------------------------
    for line in style.get("footer_lines", []):
        lw = _text_w(draw, line, f_small)
        draw.text(((PAPER_WIDTH - lw) / 2, y), line, font=f_small, fill=FAINT_INK)
        y += 15

    y += 14
    _draw_barcode(draw, MARGIN, y, content_w, 36)
    y += 40
    code_line = tx["receipt_number"]
    cw = _text_w(draw, code_line, f_small)
    draw.text(((PAPER_WIDTH - cw) / 2, y), code_line, font=f_small, fill=INK)
    y += 24

    # Crop to actual content height with a little bottom margin.
    final = img.crop((0, 0, PAPER_WIDTH, y + 16))
    return final

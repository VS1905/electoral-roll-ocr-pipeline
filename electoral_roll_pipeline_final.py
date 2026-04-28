#!/usr/bin/env python3
"""OCR pipeline for extracting voter records from Hindi electoral roll PDFs."""


import os, re, sys, csv, time, logging, argparse, tempfile, traceback
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from pathlib       import Path
from typing        import Dict, List, Optional, Tuple
from collections   import Counter

try:
    import numpy as np
    from PIL import Image
    import fitz
except ImportError as e:
    sys.exit(f"[FATAL] {e}\nRun: pip install pymupdf Pillow numpy")

try:
    from paddleocr import PaddleOCR
except ImportError:
    sys.exit("[FATAL] Run: pip install paddleocr")


HEADER_H_FRAC   = 0.12

LINE_DARK       = 150
LINE_FRAC_MIN   = 0.88
PAGE_TOP_FRAC   = 0.02
PAGE_BOT_FRAC   = 0.96
MIN_BOX_H_PX    = 150
COL_DARK        = 150
COL_FRAC_MIN    = 0.70
BLANK_DARK_RATIO = 0.010
BLANK_TEXT_RATIO = 0.004
EPIC_CONF_MIN   = 0.65
SKIP_PAGES      = {1, 2}

CSV_COLUMNS = [
    "Serial_Number", "EPIC_Number",
    "Voter_Name", "Relative_Name", "Relation_Type",
    "House_Number", "Age", "Gender",
]

_DEVA = str.maketrans("०१२३४५६७८९", "0123456789")
_PHOTO_TOKENS = {"फोटो", "उपलब्ध", "है"}


_EPIC_FULL   = re.compile(r"^([A-Z]{3}\d{7}|UP/\d{1,3}/\d{1,3}/\d{7,10})$", re.I)
_EPIC_SEARCH = re.compile(r"([A-Z]{3}\d{7}|UP/\d{1,3}/\d{1,3}/\d{7,10})", re.I)

_EPIC_LOOSE  = re.compile(r"([A-Z]{2,4}\d{5,10}|UP/\d{1,3}/\d{1,3}/\d{5,10})", re.I)

_SERIAL_RE   = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")

_RE_NAME    = re.compile(r"नाम\s*[:।]?\s*(.+)")
_RE_FATHER  = re.compile(r"पिता\s*(?:का\s*)?नाम\s*[:।]?\s*(.*?)(?=मकान\s*संख्या|आयु|लिंग|$)")
_RE_HUSBAND = re.compile(r"पति\s*(?:का\s*)?नाम\s*[:।]?\s*(.*?)(?=मकान\s*संख्या|आयु|लिंग|$)")
_RE_MOTHER  = re.compile(r"माता\s*(?:का\s*)?नाम\s*[:।]?\s*(.*?)(?=मकान\s*संख्या|आयु|लिंग|$)")
_RE_HOUSE   = re.compile(r"मकान\s*संख्या\s*[:।]?\s*([^\s]+)")
_RE_AGE     = re.compile(r"आयु[^\d०-९]{0,10}([\d०-९]{1,3})")
_RE_GENDER  = re.compile(r"लिंग\s*[:।]?\s*(पुरुष|महिला|तृतीय\s*लिंग)")

_FIELD_WORDS = ["मकान", "संख्या", "आयु", "लिंग", "पिता", "पति", "माता", "फोटो", "उपलब्ध"]
_HINDI_RE    = re.compile(r"[\u0900-\u097F]")


_FIELD_STOP_RE = re.compile(
    r"\s*(?:मकान\s*संख्या|मकान|संख्या|आयु|लिंग|फोटो|उपलब्ध|पृष्ठ|कुल\s*पृष्ठ)\s*[:।]?.*$",
    re.U
)


def nd(s: str) -> str:
    return str(s or "").translate(_DEVA)

def normalise_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def has_hindi(text: str) -> bool:
    return bool(_HINDI_RE.search(str(text or "")))

def strip_token(t: str) -> str:
    return str(t or "").strip("[](){} =|-.,;:'\"<>").upper()

def is_epic(t: str) -> bool:
    return bool(_EPIC_FULL.match(strip_token(t)))

def clean(v: str) -> str:
    """Strip field-bleed, photo noise, and trailing punctuation."""
    v = str(v or "")
    v = re.sub(r"\s*फोटो.*$",    "", v, flags=re.U)
    v = re.sub(r"\s*उपलब्ध.*$", "", v, flags=re.U)
    v = _FIELD_STOP_RE.sub("", v)
    v = re.sub(r"\s+", " ", v)
    v = re.sub(r"[\s\-–|।,:]+$", "", v)
    return v.strip(" -–|\"'[](){}")

def norm_age(raw: str) -> str:
    """Return plausible voter age, fixing OCR digit-joins like 261 → 26."""
    s = nd(str(raw or ""))
    m = re.search(r"\d{1,3}", s)
    if not m:
        return ""
    age = m.group(0)
    try:
        v = int(age)
    except ValueError:
        return ""
    if 18 <= v <= 120:
        return str(v)
    if len(age) == 3:
        for cand in (age[:2], age[-2:]):
            try:
                cv = int(cand)
            except ValueError:
                continue
            if 18 <= cv <= 120:
                return str(cv)
    return ""

def norm_gender(raw: str) -> str:
    if "पुरुष"  in raw: return "पुरुष"
    if "महिला"  in raw: return "महिला"
    if "तृतीय" in raw: return "तृतीय लिंग"
    return ""

def is_photo_noise(text: str) -> bool:
    tokens = set(text.split())
    return bool(tokens & _PHOTO_TOKENS) and not _HINDI_RE.search(
        re.sub(r"[फोटोउपलब्धहै\s]", "", text)
    )

def strip_garbage_prefix(v: str) -> str:
    """
    Strip non-Hindi garbage from the start of an extracted name.
    Finds the first run of 2+ consecutive Hindi chars and starts from there.
    Example: "He P संतराम" → "संतराम"
             "h : h कुमार" → "कुमार"
             "hद2 : h देयी" → "देयी"
    """
    m = re.search(r"[\u0900-\u097F]{2,}", v)
    if m and m.start() > 0:
        return v[m.start():]
    return v

def text_quality_ok(v: str) -> bool:
    """
    Reject OCR garbage; accept clean Hindi names.
    Rules (applied after strip_garbage_prefix):
      - No colon characters (field separator bleed)
      - No digit characters (digits = OCR artifact in person names)
      - At least 3 Hindi Unicode characters
      - Hindi chars make up >= 70% of visible non-space characters
    """
    v = v.strip()
    if len(v) < 2:
        return False

    if ":" in v or "।" in v:
        return False
    if re.search(r"[\d०-९]", v):
        return False
    hindi = len(re.findall(r"[\u0900-\u097F]", v))
    if hindi < 3:
        return False
    visible = re.sub(r"[\s\-–|,;!?]", "", v)
    if not visible:
        return False
    return (hindi / len(visible)) >= 0.70

def setup_logging(debug=False, log_file=None):
    level    = logging.DEBUG if debug else logging.INFO
    fmt      = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    return logging.getLogger("electoral_roll")


_OCR: Optional[PaddleOCR] = None

def get_ocr(device="gpu") -> PaddleOCR:
    global _OCR
    if _OCR is None:
        _OCR = PaddleOCR(lang="hi", device=device)
    return _OCR

def ocr_pil(img: Image.Image, device="gpu") -> List[Tuple[float, float, str, float]]:
    """OCR a PIL image → [(top_y, left_x, text, conf)] sorted reading-order."""
    if img.width < 12 or img.height < 8:
        return []
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        img.save(tmp, format="PNG")
        results = list(get_ocr(device).predict(tmp))
    finally:
        try: os.remove(tmp)
        except OSError: pass
    if not results:
        return []
    res    = results[0]
    texts  = res.get("rec_texts",  []) or []
    scores = res.get("rec_scores", []) or []
    polys  = res.get("rec_polys",  None)
    items  = []
    for i, text in enumerate(texts):
        text = str(text).strip()
        if not text: continue
        conf   = float(scores[i]) if i < len(scores) else 1.0
        top_y  = float(min(p[1] for p in polys[i])) if polys is not None and i < len(polys) else float(i*30)
        left_x = float(min(p[0] for p in polys[i])) if polys is not None and i < len(polys) else 0.0
        items.append((top_y, left_x, text, conf))
    items.sort(key=lambda x: (round(x[0]/20)*20, x[1]))
    return items


def get_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path); n = len(doc); doc.close(); return n

def rasterize_page(pdf_path: str, page_num: int, dpi=300) -> Optional[Image.Image]:
    try:
        doc  = fitz.open(pdf_path)
        page = doc[page_num - 1]
        mat  = fitz.Matrix(dpi/72.0, dpi/72.0)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close(); return img
    except Exception:
        return None


def detect_columns(img: Image.Image) -> List[Tuple[int, int]]:
    W, H = img.size
    def thirds(): t = W//3; return [(0,t),(t,2*t),(2*t,W)]
    arr  = np.array(img.convert("L"))
    y0, y1 = int(H*0.15), int(H*0.85)
    col_dark = (arr[y0:y1,:] < COL_DARK).mean(axis=0)
    dark = np.where(col_dark > COL_FRAC_MIN)[0]
    if len(dark) == 0: return thirds()
    groups, s, p = [], int(dark[0]), int(dark[0])
    for c in dark[1:]:
        c = int(c)
        if c-p > 5: groups.append((s,p)); s=c
        p = c
    groups.append((s,p))
    dividers = [(g[0]+g[1])//2 for g in groups
                if W*0.05 < g[0] and g[1] < W*0.95 and (g[1]-g[0]) < 40]
    if not dividers: return thirds()
    target1, target2 = W/3.0, 2*W/3.0
    c1 = [d for d in dividers if W*0.25 <= d <= W*0.45]
    c2 = [d for d in dividers if W*0.55 <= d <= W*0.75]
    if c1 and c2:
        d1 = min(c1, key=lambda d: abs(d-target1))
        d2 = min(c2, key=lambda d: abs(d-target2))
        widths = [d1, d2-d1, W-d2]
        if d1 < d2 and min(widths) > W*0.24 and max(widths) < W*0.42:
            return [(0,d1),(d1,d2),(d2,W)]
    return thirds()


def detect_box_rows(col_img: Image.Image) -> List[Tuple[int, int]]:
    W, H  = col_img.size
    arr   = np.array(col_img.convert("L"))
    y0, y1 = int(H*PAGE_TOP_FRAC), int(H*PAGE_BOT_FRAC)
    x_pad = max(5, int(W*0.03))
    dark_frac = (arr[y0:y1, x_pad:W-x_pad] < LINE_DARK).mean(axis=1)
    dark = np.where(dark_frac > LINE_FRAC_MIN)[0] + y0
    if len(dark) == 0: return []
    segs, s, p = [], int(dark[0]), int(dark[0])
    for r in dark[1:]:
        r = int(r)
        if r-p > 4: segs.append((s,p)); s=r
        p = r
    segs.append((s,p))
    pairs, i = [], 0
    while i < len(segs):
        s1 = segs[i]
        if i+1 < len(segs) and segs[i+1][0]-s1[1] <= 60:
            pairs.append((s1[0], segs[i+1][1])); i += 2
        else:
            pairs.append((s1[0], s1[1])); i += 1
    filtered = []
    for k, (ht, hb) in enumerate(pairs):
        next_ht = pairs[k+1][0] if k+1 < len(pairs) else y1
        if (next_ht-hb) >= MIN_BOX_H_PX:
            filtered.append((ht, hb))
    return filtered


def is_blank_box(box_img: Image.Image) -> bool:
    arr  = np.array(box_img.convert("L"))
    h, w = arr.shape
    if h < 30 or w < 30: return True
    core = arr[max(3,int(h*0.05)):min(h,int(h*0.95)),
               max(3,int(w*0.05)):min(w,int(w*0.95))]
    if core.size == 0: return True
    dark_ratio = (core < 100).mean()
    left_core  = core[:, :min(int(w*0.35), core.shape[1])]
    text_ratio = (left_core < 200).mean() if left_core.size else 0.0
    return dark_ratio < BLANK_DARK_RATIO or text_ratio < BLANK_TEXT_RATIO


def build_logical_lines(
    items:          List[Tuple[float, float, str, float]],
    box_width:      Optional[int] = None,
    left_text_frac: float = 0.90,
    y_tol:          float = 18.0,
) -> List[Tuple[float, float, str, float]]:
    """
    Convert raw PaddleOCR detections into logical left-side text lines.

    v8 change:
    - keeps the v7 body-crop idea, but avoids over-filtering useful tokens;
    - drops serial/header noise if it leaks into body crop;
    - filters EPIC fragments/photo text;
    - groups split Hindi tokens on the same visual line.
    """
    x_limit = (box_width * left_text_frac) if box_width else None
    filtered = []
    for y, x, t, c in items:
        t = normalise_spaces(t)
        if not t:
            continue
        if is_photo_noise(t) or "फोटो" in t or "उपलब्ध" in t:
            continue
        if x_limit is not None and x > x_limit:
            continue
        t2 = strip_token(t)
        if is_epic(t2) or _EPIC_LOOSE.search(t2):
            continue
        if float(y) < 25 and re.fullmatch(r"\d{1,4}", nd(t.strip())):
            continue
        filtered.append((float(y), float(x), t, float(c)))

    filtered.sort(key=lambda z: (z[0], z[1]))

    rows = []
    for item in filtered:
        y, x, t, c = item
        placed = False
        for row in rows:
            if abs(row["y"] - y) <= y_tol:
                row["items"].append(item)
                row["y"] = min(row["y"], y)
                row["conf"] = max(row["conf"], c)
                placed = True
                break
        if not placed:
            rows.append({"y": y, "conf": c, "items": [item]})

    out = []
    for row in rows:
        row_items = sorted(row["items"], key=lambda z: z[1])
        text = normalise_spaces(" ".join(t for _, _, t, _ in row_items))
        x_min = min(x for _, x, _, _ in row_items)
        conf  = max(c for _, _, _, c in row_items)
        if text:
            out.append((row["y"], x_min, text, conf))

    out.sort(key=lambda z: (z[0], z[1]))
    return out


def extract_serial_and_epic(
    header_items: List[Tuple[float, float, str, float]],
    all_items:    List[Tuple[float, float, str, float]],
) -> Tuple[str, str]:
    """
    Extract serial and EPIC from header-zone (and full-box fallback) OCR lines.
    Returns (serial, epic).
    """
    def joined(items): return " ".join(t for _,_,t,_ in items)


    epic = raw_epic = ""
    for items in (header_items, all_items):
        for _, _, t, c in items:
            t2 = strip_token(t)
            if not raw_epic:
                m0 = _EPIC_LOOSE.search(t2)
                if m0: raw_epic = m0.group(1).upper()
            if is_epic(t2) and c >= EPIC_CONF_MIN:
                epic = t2; raw_epic = raw_epic or t2; break
        if epic: break

    if not epic:
        txt = joined(all_items).upper()
        m = _EPIC_SEARCH.search(txt)
        if m:
            epic = m.group(1).upper()
            raw_epic = raw_epic or epic


    serial = ""
    cands  = []
    for y, x, t, c in header_items:
        token = nd(strip_token(t))
        if re.fullmatch(r"\d{1,4}", token):
            v = int(token)
            if 1 <= v <= 999 and x > 8:
                cands.append((x, y, -c, v))
    if cands:
        cands.sort(key=lambda z: (z[0], z[1], z[2]))
        serial = str(cands[0][3])


    if not serial:
        hdr = nd(joined(header_items))
        for tok in {epic, raw_epic}:
            if tok:
                hdr = hdr.replace(tok, " ")
                m_d = re.search(r"\d{5,}", tok)
                if m_d: hdr = hdr.replace(m_d.group(), " ")
        for m in _SERIAL_RE.finditer(hdr):
            v = int(m.group(1))
            if 1 <= v <= 999:
                serial = str(v); break

    return serial, epic


def fill_missing_serials(records: List[dict], n_cols: int, page_num: int,
                         log: logging.Logger) -> None:
    """
    Infer missing serial numbers from the page's column/box grid layout.
    Electoral roll pages are laid out row-wise: row-bi, col-ci → serial = base + bi*n_cols + ci.
    """
    bases = []
    for r in records:
        s = r.get("Serial_Number", "")
        if not s or not str(s).isdigit(): continue
        ci, bi = r.get("_ci"), r.get("_bi")
        if ci is None or bi is None: continue
        expected_offset = int(bi) * n_cols + int(ci) + 1
        bases.append(int(s) - expected_offset)
    if not bases: return
    base, count = Counter(bases).most_common(1)[0]
    if count < 3: return
    filled = 0
    for r in records:
        if r.get("Serial_Number"): continue
        ci, bi = r.get("_ci"), r.get("_bi")
        if ci is None or bi is None: continue
        expected = base + int(bi) * n_cols + int(ci) + 1
        if 1 <= expected <= 9999:
            r["Serial_Number"] = str(expected)
            filled += 1
    if filled:
        log.debug(f"  Page {page_num}: filled {filled} missing serial(s) from grid pattern")


def clean_candidate_text(v: str) -> str:
    """
    Normalise a possible Hindi person name without allowing OCR garbage to leak in.
    """
    v = normalise_hindi_ocr_text(normalise_spaces(str(v or "")))

    v = clean(v)


    parts = re.split(r"(?:नाम|पिता|पति|माता)\s*[:ः।]?", v)
    cleaned_parts = []
    for part in parts:
        toks = re.findall(r"[\u0900-\u097F]+", part)
        toks = [t for t in toks if t not in {"नाम", "पिता", "पति", "माता", "का", "मकान", "संख्या", "आयु", "लिंग", "फोटो", "उपलब्ध", "है"}]
        if toks:
            cleaned_parts.append(normalise_spaces(" ".join(toks)))
    if cleaned_parts:

        for part in cleaned_parts:
            if len(part.split()) >= 2:
                return part
        return cleaned_parts[0]

    toks = re.findall(r"[\u0900-\u097F]+", v)
    toks = [t for t in toks if t not in {"नाम", "पिता", "पति", "माता", "का", "मकान", "संख्या", "आयु", "लिंग", "फोटो", "उपलब्ध", "है"}]
    return normalise_spaces(" ".join(toks))

def candidate_quality_score(v: str, explicit_label: bool = False) -> int:
    """
    Score a Hindi person-name candidate. Negative means reject.
    Valid names: 1-4 words, all Devanagari, at least half words must be 3+ chars.
    """
    v = clean_candidate_text(v)
    if len(v) < 2:
        return -1
    if ":" in v or "।" in v:
        return -1
    if re.search(r"[\d०-९A-Za-z<>≥=+_\\/]", v):
        return -1
    hindi = len(re.findall(r"[\u0900-\u097F]", v))
    visible = re.sub(r"[\s\-–|,;!?]", "", v)
    if not visible or hindi < 3:
        return -1
    if hindi / len(visible) < 0.90:
        return -1
    words = [w for w in v.split() if w]
    if len(words) > 4:
        return -1

    good_words = [w for w in words if len(w) >= 3 and re.fullmatch(r"[\u0900-\u097F]+", w)]
    if not good_words:
        return -1
    if len(words) >= 2 and len(good_words) < max(1, len(words) // 2):
        return -1
    if len(words) == 1 and words[0] in _LOW_INFO_SINGLE_TOKENS and not explicit_label:
        return -1
    score = hindi + 4 * len(words)
    if explicit_label:
        score += 10
    if len(words) >= 2:
        score += 4
    return score

def choose_best_text(*values: str) -> str:
    best = ""
    best_score = -1
    for v in values:
        vv = clean_candidate_text(v)
        sc = candidate_quality_score(vv)
        if sc > best_score or (sc == best_score and len(vv) > len(best)):
            best, best_score = vv, sc
    return best if best_score >= 0 else ""

def ocr_pil_scaled(img: Image.Image, device="gpu", scale: float = 2.5) -> List[Tuple[float, float, str, float]]:
    """OCR a small line/field crop after autocontrast + scaling.
    This is only used as a fallback for missing name/relative fields.
    """
    if img.width < 10 or img.height < 10:
        return []
    try:
        from PIL import ImageOps, ImageFilter
        work = img.convert("L")
        work = ImageOps.autocontrast(work)
        work = work.filter(ImageFilter.SHARPEN)
        if scale and scale != 1:
            work = work.resize((max(1, int(work.width * scale)), max(1, int(work.height * scale))), Image.Resampling.LANCZOS)
        work = work.convert("RGB")
        raw = ocr_pil(work, device)
        return [(y / scale, x / scale, t, c) for y, x, t, c in raw]
    except Exception:
        return ocr_pil(img, device)

def ocr_name_rel_zones(body_crop: Image.Image, device: str, debug_path: Optional[Path] = None) -> List[Tuple[float, float, str, float]]:
    """Run OCR on only the name and relation strips.
    """
    W, H = body_crop.size
    zones = [
        ("name", int(H * 0.10), int(H * 0.35)),
        ("rel",  int(H * 0.24), int(H * 0.52)),
    ]
    out: List[Tuple[float, float, str, float]] = []
    for zname, y0, y1 in zones:
        y0 = max(0, min(H - 1, y0))
        y1 = max(y0 + 12, min(H, y1))
        crop = body_crop.crop((0, y0, W, y1))
        if debug_path is not None:
            try:
                crop.save(debug_path.with_name(debug_path.stem + f"_{zname}.png"))
            except Exception:
                pass
        for y, x, t, c in ocr_pil_scaled(crop, device=device, scale=2.5):
            out.append((y + y0, x, t, c))
    out.sort(key=lambda z: (z[0], z[1]))
    return out

def relation_fallback_from_sources(
    sources: List[Tuple[List[Tuple[float, float, str, float]], Optional[int]]],
    voter_name: str = "",
) -> Tuple[str, str]:
    """Cross-source relative extraction.
    """
    all_lines: List[Tuple[float, float, str, float]] = []
    for items, width in sources:
        all_lines.extend(build_logical_lines(items, box_width=width))
    all_lines.sort(key=lambda z: (z[0], z[1]))
    texts = [t for _, _, t, _ in all_lines]
    all_text = normalise_spaces(" ".join(texts))

    rel_type = ""
    for lbl in ("पिता", "पति", "माता"):
        if lbl in all_text:
            rel_type = lbl
            break
    if not rel_type:
        return "", ""

    candidates = []

    m = re.search(rel_type + r".*?नाम\s*[:ः।]?\s*(.*?)(?=\s*(?:मकान|संख्या|आयु|लिंग)|$)", all_text)
    if m:
        val = clean_candidate_text(m.group(1))
        sc = candidate_quality_score(val, explicit_label=True)
        if sc >= 0:
            candidates.append((sc, val))


    for y, x, line, conf in all_lines:
        if not (70 <= y <= 145):
            continue
        if any(k in line for k in ("मकान", "संख्या", "आयु", "लिंग", "फोटो", "उपलब्ध")):
            continue
        if rel_type in line or "नाम" in line or "का" in line:

            line2 = re.sub(rel_type, " ", line)
            line2 = re.sub(r"(?:का\s*)?नाम\s*[:ः।]?", " ", line2)
        else:
            line2 = line
        val = clean_candidate_text(line2)
        sc = candidate_quality_score(val, explicit_label=(rel_type in line or "नाम" in line))
        if sc >= 0:
            candidates.append((sc, val))

    if not candidates:
        return "", rel_type
    candidates.sort(key=lambda z: (z[0], len(z[1])), reverse=True)
    rel = candidates[0][1]
    if rel and voter_name and clean(rel) == clean(voter_name):
        rel = ""
    return rel, rel_type

def needs_name_rel_zone_fallback(fields: Dict[str, str]) -> bool:
    """Only fallback for the fields that are still missing.
    """
    return not fields.get("Voter_Name") or not fields.get("Relative_Name") or not fields.get("Relation_Type")

def safe_apply_name_rel_fallback(
    fields: Dict[str, str],
    zone_items: List[Tuple[float, float, str, float]],
    body_items: List[Tuple[float, float, str, float]],
    full_body_items: List[Tuple[float, float, str, float]],
    body_width: Optional[int],
    full_width: Optional[int],
) -> Dict[str, str]:
    """Fill only blank/suspect name-relative fields from the zone OCR.
    """
    out = dict(fields)

    z = finalise_fields(extract_fields_single(zone_items, box_width=body_width)) if zone_items else {}
    if not out.get("Voter_Name") and z.get("Voter_Name"):
        out["Voter_Name"] = z["Voter_Name"]

    rel, rtype = relation_fallback_from_sources(
        [(body_items, body_width), (full_body_items, full_width), (zone_items, body_width)],
        voter_name=out.get("Voter_Name", ""),
    )
    if not out.get("Relation_Type") and (rtype or z.get("Relation_Type")):
        out["Relation_Type"] = rtype or z.get("Relation_Type", "")
    if not out.get("Relative_Name") and (rel or z.get("Relative_Name")):
        out["Relative_Name"] = rel or z.get("Relative_Name", "")

    return finalise_fields(out)


_LOW_INFO_SINGLE_TOKENS = {
    "यादव", "कुमार", "भवन", "देयी", "दास", "शंकर",
    "नाम", "पिता", "पति", "माता", "मकान", "संख्या",
}
_REPEATED_WORD_RE = re.compile(r"\b([\u0900-\u097F]+)(?:\s+\1\b)+")

def normalise_hindi_ocr_text(v: str) -> str:
    v = normalise_spaces(v or "")
    fixes = {
        "कमार":    "कुमार",
        "यदव":     "यादव",
        "े देवी":  " देवी",
        "रामशंकर": "राम शंकर",
        "कृषण":    "कृष्ण",
        "क्रमार":  "कुमार",
        "फमार":    "कुमार",
        "नामः":    "नाम:",
        "नाम ः":   "नाम:",
        "दोढे":    "ढोढे",
        "पाला":    "पाल",
    }
    for a, b in fixes.items():
        v = v.replace(a, b)
    v = _REPEATED_WORD_RE.sub(r"\1", v)


    words = v.split()
    if len(words) >= 2:
        last = words[-1]


        _NOISE_SUFFIXES = {"ता","ते","ा","जा","ाखा","थिना","पिनता","पाला","पिज्ा","पिज्ञा","खा","पित्ता","ाम","साहन","साहन"}
        has_standalone_matra = bool(re.search(r"्\s|्$|^\s*[ािीुूेैोौंःँ]", last))
        if last in _NOISE_SUFFIXES or (len(last) <= 3 and not re.fullmatch(r"[\u0900-\u097F]{4,}", last)) or has_standalone_matra:
            words = words[:-1]
    v = " ".join(words)
    return normalise_spaces(v)

def strip_embedded_label_noise(v: str) -> str:
    v = normalise_hindi_ocr_text(v or "")

    parts = re.split(r"\s+(?:नाम|पिता|पति|माता)\s*[:ः।]", v)
    parts = [clean(normalise_hindi_ocr_text(x)) for x in parts if clean(x)]
    if not parts:
        return clean(v)


    first = parts[0]
    if candidate_quality_score(first) >= 0:
        return first

    words = first.split()
    if len(words) > 2:
        trimmed = " ".join(words[:2])
        if candidate_quality_score(trimmed) >= 0:
            return trimmed
    return first

def is_low_information_name(v: str) -> bool:
    v = clean_candidate_text(normalise_hindi_ocr_text(v or ""))
    words = [w for w in v.split() if w]
    return len(words) == 1 and words[0] in _LOW_INFO_SINGLE_TOKENS

def final_clean_person(v: str, allow_low_info: bool = False) -> str:
    v = strip_embedded_label_noise(v or "")
    v = clean_candidate_text(normalise_hindi_ocr_text(v))
    if not v:
        return ""
    if candidate_quality_score(v) < 0:
        return ""
    if not allow_low_info and is_low_information_name(v):
        return ""
    return v

def validate_house_number(v: str) -> str:
    v = nd(clean(str(v or "")))
    if not v:
        return ""
    v = v.replace("ह", "अ") if re.search(r"\d", v) else v
    if re.fullmatch(r"\d{1,4}[\u0900-\u097F]?", v):
        return v
    token = v.split()[0]
    if re.fullmatch(r"\d{1,4}[\u0900-\u097F]?", token):
        return token
    return ""

def finalise_fields(fields: Dict[str, str]) -> Dict[str, str]:
    out = dict(fields)
    out["Voter_Name"] = final_clean_person(out.get("Voter_Name", ""), allow_low_info=False)
    out["Relative_Name"] = final_clean_person(out.get("Relative_Name", ""), allow_low_info=False)
    if out.get("Relative_Name") and out.get("Voter_Name") and clean(out["Relative_Name"]) == clean(out["Voter_Name"]):
        out["Relative_Name"] = ""


    if out.get("Relation_Type") not in {"पिता", "पति", "माता"}:
        out["Relation_Type"] = ""
    out["House_Number"] = validate_house_number(out.get("House_Number", ""))
    out["Age"] = norm_age(out.get("Age", ""))
    out["Gender"] = norm_gender(out.get("Gender", ""))
    return out

def extract_fields_single(
    body_items: List[Tuple[float, float, str, float]],
    box_width:  Optional[int] = None,
) -> Dict[str, str]:
    """
    Extract voter fields from one OCR source.
    """
    lines    = build_logical_lines(body_items, box_width=box_width)
    texts    = [line for _, _, line, _ in lines]
    all_text = normalise_spaces(" ".join(texts))

    voter_candidates = []
    for _, _, line, _ in lines:
        if any(rel_kw in line for rel_kw in ("पिता", "पति", "माता")):
            continue
        m = re.search(r"नाम\s*[:।]?\s*(.+)", line)
        if m:
            val = clean_candidate_text(m.group(1))
            sc = candidate_quality_score(val, explicit_label=True)
            if sc >= 0:
                voter_candidates.append((sc, val))

    if not voter_candidates:
        for i, (_, _, line, _) in enumerate(lines):
            if any(rel in line for rel in ("पिता", "पति", "माता")):
                continue
            if not re.search(r"नाम\s*[:।]", line):
                continue
            parts = []
            m = re.search(r"नाम\s*[:।]\s*(.+)", line)
            if m and clean_candidate_text(m.group(1)):
                parts.append(m.group(1))
            for _, _, nxt, _ in lines[i+1:i+4]:
                if any(kw in nxt for kw in _FIELD_WORDS):
                    break
                if has_hindi(nxt):
                    parts.append(nxt)
            val = clean_candidate_text(" ".join(parts))
            sc = candidate_quality_score(val, explicit_label=True)
            if sc >= 0:
                voter_candidates.append((sc, val))

    if not voter_candidates:
        for _, _, line, _ in lines:
            if any(kw in line for kw in _FIELD_WORDS):
                continue
            val = clean_candidate_text(line)
            sc = candidate_quality_score(val, explicit_label=False)
            if sc >= 0:
                voter_candidates.append((sc, val))

    voter_name = ""
    voter_score = -1
    if voter_candidates:
        voter_candidates.sort(key=lambda z: (z[0], len(z[1])), reverse=True)
        voter_score, voter_name = voter_candidates[0]

    rel_candidates = []
    seen_relation_type = ""
    for lbl in ("पिता", "पति", "माता"):
        if lbl in all_text and not seen_relation_type:
            seen_relation_type = lbl
        m = re.search(
            lbl + r"\s*(?:का\s*)?नाम\s*[:।]?\s*(.*?)(?=\s*(?:मकान|संख्या|आयु|लिंग)|$)",
            all_text,
        )
        if m:
            val = clean_candidate_text(m.group(1))
            sc = candidate_quality_score(val, explicit_label=True)
            if sc >= 0:
                rel_candidates.append((sc, val, lbl))

        for i, (_, _, line, _) in enumerate(lines):
            if lbl not in line:
                continue
            window = normalise_spaces(" ".join(t for _, _, t, _ in lines[i:i+5]))
            m2 = re.search(
                lbl + r".*?नाम\s*[:।]?\s*(.*?)(?=\s*(?:मकान|संख्या|आयु|लिंग)|$)",
                window,
            )
            if m2:
                val = clean_candidate_text(m2.group(1))
                sc = candidate_quality_score(val, explicit_label=True)
                if sc >= 0:
                    rel_candidates.append((sc, val, lbl))

            after = re.sub(lbl, "", line, count=1).strip()
            after = re.sub(r"^(?:का\s*)?नाम\s*[:।]?", "", after).strip()
            val = clean_candidate_text(after)
            sc = candidate_quality_score(val, explicit_label=True)
            if sc >= 0:
                rel_candidates.append((sc, val, lbl))

            for _, _, nxt, _ in lines[i+1:i+4]:
                if any(kw in nxt for kw in ("मकान", "संख्या", "आयु", "लिंग", "फोटो")):
                    break
                val = clean_candidate_text(nxt)
                sc = candidate_quality_score(val, explicit_label=False)
                if sc >= 0:
                    rel_candidates.append((sc, val, lbl))

    relative_name = ""
    relation_type = seen_relation_type
    rel_score = -1
    if rel_candidates:
        rel_candidates.sort(key=lambda z: (z[0], len(z[1])), reverse=True)
        rel_score, relative_name, relation_type = rel_candidates[0]

    if relative_name and clean(relative_name) == clean(voter_name):
        relative_name = relation_type = ""

    house = ""
    m = re.search(r"मकान\s*संख्या\s*[:।]?\s*([^\s]+)", all_text)
    if not m:
        m = re.search(r"संख्या\s*[:।]?\s*([^\s]+)", all_text)
    if m:
        house = nd(clean(m.group(1)))

    age = ""
    m = re.search(r"आयु[^\d०-९]{0,10}([\d०-९]{1,3})", all_text)
    if m:
        age = norm_age(m.group(1))

    gender = ""
    m = _RE_GENDER.search(all_text)
    if m:
        gender = norm_gender(m.group(1))
    if not gender:
        gender = norm_gender(all_text)

    return {
        "Voter_Name":    voter_name,
        "Relative_Name": relative_name,
        "Relation_Type": relation_type,
        "House_Number":  house,
        "Age":           age,
        "Gender":        gender,
        "_Voter_Score":   voter_score,
        "_Rel_Score":     rel_score,
        "_lines":         texts,
        "_all_text":      all_text,
    }

def merge_field_results(primary: Dict[str, str], secondary: Dict[str, str]) -> Dict[str, str]:
    """
    Merge body-crop and full-crop extraction.
    """

    p_name, s_name = primary.get("Voter_Name", ""), secondary.get("Voter_Name", "")
    p_score, s_score = int(primary.get("_Voter_Score", -1)), int(secondary.get("_Voter_Score", -1))
    if p_name and (p_score > s_score or (p_score == s_score and len(p_name) >= len(s_name))):
        voter = p_name
    elif s_name:
        voter = s_name
    else:
        voter = ""

    rel_primary = primary.get("Relative_Name", "")
    rel_secondary = secondary.get("Relative_Name", "")
    rp_score, rs_score = int(primary.get("_Rel_Score", -1)), int(secondary.get("_Rel_Score", -1))
    relation_type = ""
    if rel_primary and (rp_score > rs_score or (rp_score == rs_score and len(rel_primary) >= len(rel_secondary))):
        rel = rel_primary
        relation_type = primary.get("Relation_Type", "")
    elif rel_secondary:
        rel = rel_secondary
        relation_type = secondary.get("Relation_Type", "")
    else:
        rel = ""

    if rel and voter and clean(rel) == clean(voter):
        rel = relation_type = ""

    return {
        "Voter_Name":    voter,
        "Relative_Name": rel,
        "Relation_Type": relation_type,
        "House_Number":  primary.get("House_Number", "") or secondary.get("House_Number", ""),
        "Age":           primary.get("Age", "") or secondary.get("Age", ""),
        "Gender":        primary.get("Gender", "") or secondary.get("Gender", ""),
    }

def extract_fields(
    body_items: List[Tuple[float, float, str, float]],
    box_width:  Optional[int] = None,
    secondary_items: Optional[List[Tuple[float, float, str, float]]] = None,
    secondary_width: Optional[int] = None,
) -> Dict[str, str]:
    """
    Extract fields from body-crop OCR and merge with full-crop OCR fallback.
    """
    primary = extract_fields_single(body_items, box_width=box_width)
    if secondary_items is None:
        return {k: primary.get(k, "") for k in [
            "Voter_Name", "Relative_Name", "Relation_Type", "House_Number", "Age", "Gender"
        ]}
    secondary = extract_fields_single(secondary_items, box_width=secondary_width)
    return merge_field_results(primary, secondary)


def extract_pdf_metadata(pdf_path: str, dpi: int, device: str, debug: bool,
                         debug_dir: Optional[str], log: logging.Logger) -> Dict[str, str]:
    """
    Extract page-level metadata separately from voter boxes.
    """
    meta = {
        "Electoral_Year": "",
        "AC_No_Name": "",
        "Section_No_Name": "",
        "Qualifying_Date": "",
        "Publication_Date": "",
    }

    def absorb(txt: str):
        txt = normalise_spaces(nd(txt or ""))
        if not txt:
            return

        if not meta["Electoral_Year"]:
            m = re.search(r"(निर्वाचक\s+नामावली\s*[0-9]{4}|नामावली\s*[0-9]{4})", txt)
            if m:
                meta["Electoral_Year"] = clean(m.group(1))

        if not meta["AC_No_Name"]:
            m = re.search(
                r"(?:विधानसभा\s+निर्वाचन\s+क्षेत्र|विधान\s*सभा\s*निर्वाचन\s*क्षेत्र)\s*[:।-]?\s*(.+?)(?=अनुभाग|भाग\s*संख्या|मतदान|$)",
                txt
            )
            if m:
                meta["AC_No_Name"] = clean(m.group(1))

        if not meta["Section_No_Name"]:
            m = re.search(
                r"(?:अनुभाग\s+संख्या\s+(?:और|व)\s+नाम|अनुभाग\s+संख्या)\s*[:।-]?\s*(.+?)(?=मतदान|पृष्ठ|निर्वाचक|$)",
                txt
            )
            if m:
                meta["Section_No_Name"] = clean(m.group(1))

        if not meta["Qualifying_Date"]:
            m = re.search(r"(?:अर्हता|योग्यता|qualifying).*?(\d{1,2}[./-]\d{1,2}[./-]\d{4})", txt, flags=re.I)
            if m:
                meta["Qualifying_Date"] = m.group(1)

        if not meta["Publication_Date"]:
            m = re.search(r"(?:प्रकाशन|publication).*?(\d{1,2}[./-]\d{1,2}[./-]\d{4})", txt, flags=re.I)
            if m:
                meta["Publication_Date"] = m.group(1)

    try:
        doc = fitz.open(pdf_path)
        for i in range(min(3, len(doc))):
            absorb(doc[i].get_text("text"))
        doc.close()
    except Exception:
        pass

    if any(not v for v in meta.values()):
        try:
            total = get_page_count(pdf_path)
            for page_num in range(1, min(3, total) + 1):
                img = rasterize_page(pdf_path, page_num, dpi)
                if img is None:
                    continue
                W, H = img.size
                strips = [
                    ("top", img.crop((0, 0, W, int(H * 0.18)))),
                    ("bottom", img.crop((0, int(H * 0.84), W, H))),
                ]
                for name, crop in strips:
                    if debug and debug_dir:
                        d = Path(debug_dir) / Path(pdf_path).stem / "_metadata"
                        d.mkdir(parents=True, exist_ok=True)
                        crop.save(d / f"p{page_num:03d}_{name}.png")
                    items = ocr_pil(crop, device)
                    absorb(" ".join(t for _, _, t, _ in items))
                if all(meta.values()):
                    break
        except Exception as e:
            log.debug(f"  Metadata OCR failed: {e}")

    log.debug(f"  Metadata: {meta}")
    return meta


def loose_epic_from_items(items: List[Tuple[float, float, str, float]]) -> str:
    """Return the best loose EPIC-like text found in OCR items."""
    candidates = []
    for _, _, t, c in items:
        m = _EPIC_LOOSE.search(strip_token(t))
        if m:
            candidates.append((len(m.group(1)), c, m.group(1).upper()))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][2]

def recover_epic_from_crop(full_crop: Image.Image, all_items: List[Tuple[float, float, str, float]],
                           device: str) -> str:
    """
    Try to recover EPIC from a larger top-right crop.
    """
    raw = loose_epic_from_items(all_items)
    if raw and is_epic(raw):
        return raw

    try:
        w, h = full_crop.size
        crop = full_crop.crop((int(w * 0.50), 0, w, min(h, max(90, int(h * 0.30)))))
        items = ocr_pil(crop, device)
        for _, _, t, c in items:
            tok = strip_token(t)
            if is_epic(tok) and c >= EPIC_CONF_MIN:
                return tok
        raw2 = loose_epic_from_items(items)
        if raw2:
            raw = raw2
    except Exception:
        pass

    return raw or ""


_OCR_NO_ORI: Optional[PaddleOCR] = None

def get_ocr_no_orientation(device="gpu") -> PaddleOCR:
    """A second PaddleOCR instance with orientation/unwarping disabled when supported.
    """
    global _OCR_NO_ORI
    if _OCR_NO_ORI is not None:
        return _OCR_NO_ORI
    try:
        _OCR_NO_ORI = PaddleOCR(
            lang="hi", device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        try:
            _OCR_NO_ORI = PaddleOCR(lang="hi", device=device, use_angle_cls=False)
        except TypeError:
            _OCR_NO_ORI = get_ocr(device)
    except Exception:
        _OCR_NO_ORI = get_ocr(device)
    return _OCR_NO_ORI

def ocr_pil_with_engine(img: Image.Image, device="gpu", no_orientation: bool = False) -> List[Tuple[float, float, str, float]]:
    if img.width < 12 or img.height < 8:
        return []
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        img.save(tmp, format="PNG")
        engine = get_ocr_no_orientation(device) if no_orientation else get_ocr(device)
        results = list(engine.predict(tmp))
    finally:
        try: os.remove(tmp)
        except OSError: pass
    if not results:
        return []
    res    = results[0]
    texts  = res.get("rec_texts",  []) or []
    scores = res.get("rec_scores", []) or []
    polys  = res.get("rec_polys",  None)
    items  = []
    for i, text in enumerate(texts):
        text = str(text).strip()
        if not text: continue
        conf   = float(scores[i]) if i < len(scores) else 1.0
        top_y  = float(min(p[1] for p in polys[i])) if polys is not None and i < len(polys) else float(i*30)
        left_x = float(min(p[0] for p in polys[i])) if polys is not None and i < len(polys) else 0.0
        items.append((top_y, left_x, text, conf))
    items.sort(key=lambda x: (round(x[0]/20)*20, x[1]))
    return items

def preprocess_field_crop(img: Image.Image, scale: float = 3.0, threshold: bool = False) -> Image.Image:
    from PIL import ImageOps, ImageFilter
    work = img.convert("L")
    work = ImageOps.autocontrast(work)
    work = work.filter(ImageFilter.SHARPEN)
    if threshold:
        work = work.point(lambda p: 255 if p > 185 else 0)
    if scale and scale != 1:
        work = work.resize((max(1, int(work.width * scale)), max(1, int(work.height * scale))), Image.Resampling.LANCZOS)
    return work.convert("RGB")

def ocr_fixed_crop_multi(img: Image.Image, device: str) -> List[Tuple[float, float, str, float]]:
    """Run conservative multi-preprocess OCR on a small name/relative line crop."""
    out: List[Tuple[float, float, str, float]] = []
    seen = set()
    for scale in (2.0, 3.0, 4.0):
        for threshold in (False, True):
            proc = preprocess_field_crop(img, scale=scale, threshold=threshold)
            for noori in (False, True):
                try:
                    raw = ocr_pil_with_engine(proc, device=device, no_orientation=noori)
                except Exception:
                    raw = []
                for y, x, t, c in raw:
                    yy, xx = y / scale, x / scale
                    key = normalise_spaces(t)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append((yy, xx, key, c))
    out.sort(key=lambda z: (z[0], z[1], -z[3]))
    return out

def line_text(items: List[Tuple[float, float, str, float]]) -> str:
    return normalise_spaces(" ".join(t for _, _, t, _ in sorted(items, key=lambda z: (z[0], z[1]))))

def pick_name_from_items(items: List[Tuple[float, float, str, float]], require_label: bool = False) -> str:
    candidates = []
    text = line_text(items)
    for raw in [text] + [t for _, _, t, _ in items]:
        raw = normalise_hindi_ocr_text(raw)
        if not raw:
            continue
        labelled = bool(re.search(r"नाम\s*[:ः।]?", raw))
        if require_label and not labelled:
            continue
        if labelled:
            raw = re.split(r"नाम\s*[:ः।]?", raw, maxsplit=1)[-1]
        val = clean_candidate_text(raw)
        sc = candidate_quality_score(val, explicit_label=labelled)
        if sc >= 0:
            candidates.append((sc, len(val), val))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][2]

def pick_relation_from_items(items: List[Tuple[float, float, str, float]], existing_type: str = "") -> Tuple[str, str]:
    text = line_text(items)
    texts = [text] + [t for _, _, t, _ in items]
    candidates = []
    found_type = existing_type if existing_type in {"पिता", "पति", "माता"} else ""
    for raw in texts:
        raw = normalise_hindi_ocr_text(raw)
        if not raw:
            continue
        lbl = ""
        for candidate_lbl in ("पिता", "पति", "माता"):
            if candidate_lbl in raw:
                lbl = candidate_lbl
                found_type = found_type or lbl
                break


        if not lbl and not existing_type:
            continue
        work = raw
        if lbl:
            work = re.split(lbl, work, maxsplit=1)[-1]
        work = re.sub(r"^(?:\s*का\s*)?नाम\s*[:ः।]?", " ", work).strip()
        if "नाम" in work:
            work = re.split(r"नाम\s*[:ः।]?", work, maxsplit=1)[-1]
        val = clean_candidate_text(work)
        sc = candidate_quality_score(val, explicit_label=bool(lbl or "नाम" in raw))
        if sc >= 0:
            candidates.append((sc, len(val), val, lbl or existing_type))
    if not candidates:
        return "", found_type
    candidates.sort(reverse=True)
    _, _, rel, rtype = candidates[0]
    return rel, rtype or found_type

def fixed_line_fallback(body_crop: Image.Image, fields: Dict[str, str], device: str, debug_base: Optional[Path] = None) -> Dict[str, str]:
    """
    Try fixed template line crops for only missing/suspicious name-relative fields.
    """
    out = dict(fields)
    W, H = body_crop.size


    name_band = (0, int(H * 0.16), int(W * 0.80), int(H * 0.34))
    name_val_band = (70, int(H * 0.16), int(W * 0.80), int(H * 0.34))
    rel_band = (0, int(H * 0.28), int(W * 0.86), int(H * 0.49))
    rel_val_band = (105, int(H * 0.28), int(W * 0.86), int(H * 0.49))

    def crop_box(box):
        x0, y0, x1, y1 = box
        return body_crop.crop((max(0, x0), max(0, y0), min(W, x1), min(H, y1)))

    if not out.get("Voter_Name") or is_low_information_name(out.get("Voter_Name", "")):
        name_items = []
        for label, box in (("name", name_band), ("name_value", name_val_band)):
            im = crop_box(box)
            if debug_base is not None:
                try: im.save(debug_base.with_name(debug_base.stem + f"_{label}_line.png"))
                except Exception: pass
            name_items.extend(ocr_fixed_crop_multi(im, device=device))
        cand = pick_name_from_items(name_items, require_label=False)
        if cand and (not is_low_information_name(cand)):
            out["Voter_Name"] = cand


    if not out.get("Relative_Name") or not out.get("Relation_Type"):
        rel_items = []
        for label, box in (("rel", rel_band), ("rel_value", rel_val_band)):
            im = crop_box(box)
            if debug_base is not None:
                try: im.save(debug_base.with_name(debug_base.stem + f"_{label}_line.png"))
                except Exception: pass
            rel_items.extend(ocr_fixed_crop_multi(im, device=device))
        rel, rtype = pick_relation_from_items(rel_items, existing_type=out.get("Relation_Type", ""))
        if not out.get("Relation_Type") and rtype:
            out["Relation_Type"] = rtype
        if not out.get("Relative_Name") and rel and rel != out.get("Voter_Name") and not is_low_information_name(rel):
            out["Relative_Name"] = rel

    return finalise_fields(out)


def process_page(
    pdf_path: str, page_num: int,
    dpi: int, device: str,
    debug: bool, debug_dir: Optional[str],
    log: logging.Logger,
) -> List[dict]:

    img = rasterize_page(pdf_path, page_num, dpi)
    if img is None:
        log.warning(f"  Page {page_num}: rasterization failed"); return []

    W, H  = img.size
    cols  = detect_columns(img)
    log.debug(f"  Page {page_num}: columns {cols}")

    records    = []
    page_boxes = 0

    for ci, (cx0, cx1) in enumerate(cols):
        col_img = img.crop((cx0, 0, cx1, H))
        col_w   = cx1 - cx0

        box_pairs = detect_box_rows(col_img)
        if len(box_pairs) < 2:
            log.debug(f"  Page {page_num} col {ci+1}: <2 boxes"); continue
        log.debug(f"  Page {page_num} col {ci+1}: {len(box_pairs)} boxes")

        for bi, (h_top, h_bot) in enumerate(box_pairs):
            box_bot    = box_pairs[bi+1][0] if bi+1 < len(box_pairs) else int(H*PAGE_BOT_FRAC)
            box_height = box_bot - h_top
            page_boxes += 1
            lbl = f"p{page_num:03d}_c{ci+1}_b{bi+1:02d}"

            full_crop = col_img.crop((0, h_top, col_w, box_bot))
            if is_blank_box(full_crop):
                log.debug(f"  {lbl}: blank, skip"); continue

            photo_start = int(col_w * 0.777)
            body_crop   = col_img.crop((0, h_bot, photo_start, box_bot))

            if debug and debug_dir:
                d = Path(debug_dir) / f"p{page_num:03d}"
                d.mkdir(parents=True, exist_ok=True)
                full_crop.save(d / f"{lbl}_full.png")
                body_crop.save(d / f"{lbl}_body.png")

            all_items     = ocr_pil(full_crop, device)
            header_thresh = box_height * HEADER_H_FRAC
            header_items  = [(y,x,t,c) for y,x,t,c in all_items if y < header_thresh]
            full_body_items = [(y,x,t,c) for y,x,t,c in all_items if y >= header_thresh]

            body_crop_items = ocr_pil(body_crop, device)
            has_left_hindi = any(
                x < 260 and re.search(r"[\u0900-\u097F]", t)
                for _, x, t, _ in body_crop_items
            )

            serial, epic = extract_serial_and_epic(header_items, all_items)
            if not epic:
                epic = recover_epic_from_crop(full_crop, all_items, device)

            fields = extract_fields(
                body_crop_items,
                box_width=photo_start,
                secondary_items=full_body_items,
                secondary_width=col_w,
            )
            fields = finalise_fields(fields)

            zone_items: List[Tuple[float, float, str, float]] = []
            if needs_name_rel_zone_fallback(fields):
                zone_debug_path = None
                if debug and debug_dir:
                    zone_debug_path = Path(debug_dir) / f"p{page_num:03d}" / f"{lbl}_zonebase.png"
                zone_items = ocr_name_rel_zones(body_crop, device, debug_path=zone_debug_path)
                if zone_items:
                    fields = safe_apply_name_rel_fallback(
                        fields, zone_items, body_crop_items, full_body_items, photo_start, col_w
                    )


            if needs_name_rel_zone_fallback(fields):
                fixed_debug_path = None
                if debug and debug_dir:
                    fixed_debug_path = Path(debug_dir) / f"p{page_num:03d}" / f"{lbl}_fixed"
                fields = fixed_line_fallback(body_crop, fields, device, debug_base=fixed_debug_path)

            if debug and debug_dir:
                d = Path(debug_dir) / f"p{page_num:03d}"
                with open(d / f"{lbl}_ocr.txt", "w", encoding="utf-8") as f:
                    f.write(f"box_height={box_height}  hdr_thresh={header_thresh:.1f}  "
                            f"body_crop_left_hindi={has_left_hindi}  field_src=merged_body_plus_full\n")
                    f.write("--- FULL CROP ---\n")
                    for ty,tx,tt,tc in all_items:
                        zone = "HDR" if ty < header_thresh else "BOD"
                        f.write(f"{zone}  y={ty:5.0f}  x={tx:5.0f}  conf={tc:.2f}  {tt}\n")
                    f.write("--- BODY CROP ---\n")
                    for ty,tx,tt,tc in body_crop_items:
                        f.write(f"BOD  y={ty:5.0f}  x={tx:5.0f}  conf={tc:.2f}  {tt}\n")
                    if zone_items:
                        f.write("--- NAME/REL ZONE FALLBACK OCR ---\n")
                        for ty,tx,tt,tc in zone_items:
                            f.write(f"ZON  y={ty:5.0f}  x={tx:5.0f}  conf={tc:.2f}  {tt}\n")
                    f.write("--- EXTRACTED ---\n")
                    f.write(str(fields) + "\n")

            if (not epic and not serial and not fields["Voter_Name"]
                    and not fields["House_Number"] and not fields["Age"]):
                log.debug(f"  {lbl}: nothing useful, skip"); continue

            log.debug(
                f"  {lbl}: serial={serial!r} epic={epic!r} "
                f"name={fields['Voter_Name']!r} "
                f"rel={fields['Relative_Name']!r}({fields['Relation_Type']!r}) "
                f"house={fields['House_Number']!r} "
                f"age={fields['Age']!r} gender={fields['Gender']!r}"
            )

            rec = {
                "Serial_Number":     serial,
                "EPIC_Number":       epic,
                "Voter_Name":        fields["Voter_Name"],
                "Relative_Name":     fields["Relative_Name"],
                "Relation_Type":     fields["Relation_Type"],
                "House_Number":      fields["House_Number"],
                "Age":               fields["Age"],
                "Gender":            fields["Gender"],
                "_ci": ci,
                "_bi": bi,
            }
            records.append(rec)

    fill_missing_serials(records, len(cols), page_num, log)
    for r in records:
        r.pop("_ci", None); r.pop("_bi", None)

    log.info(f"  Page {page_num}: {len(records)} voters  ({page_boxes} boxes scanned)")
    return records


def process_pdf(
    pdf_path: str, output_csv: str,
    dpi=300, device="gpu", debug=False,
    debug_dir=None, logger=None,
) -> dict:
    log     = logger or logging.getLogger("electoral_roll")
    t0      = time.time()
    summary = {"pdf": pdf_path, "rows": 0, "pages": 0, "errors": 0, "time_sec": 0.0}
    log.info(f"Processing: {pdf_path}")
    pdf_dbg = None
    if debug and debug_dir:
        pdf_dbg = os.path.join(debug_dir, Path(pdf_path).stem)
        os.makedirs(pdf_dbg, exist_ok=True)
    try:
        total = get_page_count(pdf_path)
        log.info(f"  Total pages: {total}")

        all_records = []
        for page_num in range(1, total+1):
            if page_num in SKIP_PAGES:
                log.debug(f"  Skip page {page_num}"); continue
            try:
                recs = process_page(pdf_path, page_num, dpi, device, debug, pdf_dbg, log)
                all_records.extend(recs)
                summary["pages"] += 1
            except Exception as e:
                summary["errors"] += 1
                log.error(f"  Page {page_num} error: {e}")
                log.debug(traceback.format_exc())
        def _serial_key(r):
            sv = str(r.get("Serial_Number", ""))
            return int(sv) if sv.isdigit() else 10**9
        all_records.sort(key=_serial_key)

        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader(); w.writerows(all_records)
        summary["rows"]     = len(all_records)
        summary["time_sec"] = round(time.time()-t0, 2)
        log.info(f"  ✓ {len(all_records)} voters → {output_csv}  ({summary['time_sec']}s)")
    except Exception as e:
        summary["errors"]  += 1
        summary["time_sec"] = round(time.time()-t0, 2)
        log.error(f"  FAILED: {e}")
        log.debug(traceback.format_exc())
    return summary


def batch_process(input_folder, output_folder, dpi=300, device="gpu",
                  debug=False, log_file=None):
    logger = setup_logging(debug=debug, log_file=log_file)
    os.makedirs(output_folder, exist_ok=True)
    debug_dir = os.path.join(output_folder, "_debug") if debug else None
    if debug_dir: os.makedirs(debug_dir, exist_ok=True)
    pdfs = sorted(str(p) for p in Path(input_folder).rglob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDFs found: {input_folder}"); return
    logger.info(f"Found {len(pdfs)} PDF(s) | DPI={dpi} | device={device}")
    logger.info("Initialising PaddleOCR...")
    get_ocr(device)
    logger.info("PaddleOCR ready.")
    summaries = []
    for pdf_path in pdfs:
        csv_out = os.path.join(output_folder, f"{Path(pdf_path).stem}.csv")
        s = process_pdf(pdf_path, csv_out, dpi=dpi, device=device,
                        debug=debug, debug_dir=debug_dir, logger=logger)
        summaries.append(s)
    rows = sum(s["rows"] for s in summaries)
    errs = sum(s["errors"] for s in summaries)
    secs = sum(s["time_sec"] for s in summaries)
    logger.info("="*60)
    logger.info(f"DONE | PDFs:{len(summaries)} | Voters:{rows} | Errors:{errs} | Time:{secs:.1f}s")
    logger.info("="*60)
    sp = os.path.join(output_folder, "_batch_summary.csv")
    with open(sp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["pdf","rows","pages","errors","time_sec"])
        w.writeheader(); w.writerows(summaries)
    logger.info(f"Batch summary: {sp}")


def main():
    p = argparse.ArgumentParser(
        description="Electoral Roll OCR — PaddleOCR v8 merged body/full OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python electoral_roll_pipeline.py ./pdfs ./output --device gpu
  python electoral_roll_pipeline.py ./pdfs ./output --device gpu --debug
  python electoral_roll_pipeline.py ./pdfs ./output --device cpu
        """,
    )
    p.add_argument("input_folder")
    p.add_argument("output_folder")
    p.add_argument("--dpi",    type=int, default=300)
    p.add_argument("--device", default="gpu", choices=["gpu","cpu"])
    p.add_argument("--debug",  action="store_true")
    p.add_argument("--log",    default=None)
    args = p.parse_args()
    batch_process(args.input_folder, args.output_folder,
                  dpi=args.dpi, device=args.device,
                  debug=args.debug, log_file=args.log)

if __name__ == "__main__":
    main()

import io
import re
import time
import zipfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from google import genai
from google.genai import types

# =========================
# CẤU HÌNH
# =========================
st.set_page_config(
    page_title="Đình Thái - SRT Translator Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 80
MAX_RETRIES_PER_KEY = 2
RETRY_SLEEP_SECONDS = 0.35
MAX_PARALLEL_BATCHES = 4

# Ngôn ngữ đích mới
TARGET_LANGUAGE_OPTIONS = [
    "Tiếng Việt",
    "Tiếng Bồ Đào Nha",
]

LANGUAGE_LABELS = {
    "zh": "Tiếng Trung",
    "ja": "Tiếng Nhật",
    "ko": "Tiếng Hàn",
    "th": "Tiếng Thái",
    "ru": "Tiếng Nga",
    "ar": "Tiếng Ả Rập",
    "vi": "Tiếng Việt",
    "es": "Tiếng Tây Ban Nha",
    "fr": "Tiếng Pháp",
    "de": "Tiếng Đức",
    "pt": "Tiếng Bồ Đào Nha",
    "en_or_latin": "Tiếng Anh / ngôn ngữ Latin",
    "unknown": "Không xác định",
}

SOURCE_LANGUAGE_OPTIONS = [
    "Tự động",
    "Tiếng Trung",
    "Tiếng Anh",
    "Tiếng Nhật",
    "Tiếng Hàn",
    "Tiếng Thái",
    "Tiếng Pháp",
    "Tiếng Đức",
    "Tiếng Nga",
    "Tiếng Tây Ban Nha",
    "Tiếng Bồ Đào Nha",
    "Tiếng Ả Rập",
]

CUSTOM_CSS = """
<style>
.block-container {max-width: 1450px; padding-top: .8rem; padding-bottom: 1.2rem;}
html, body, [class*="css"] {font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
[data-testid="stAppViewContainer"] {background: linear-gradient(180deg, #f7f8fb 0%, #eef2f7 100%);}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
.topbar {
    display:flex; justify-content:space-between; align-items:center;
    background:rgba(255,255,255,.86); backdrop-filter: blur(12px);
    border:1px solid rgba(15,23,42,.06); box-shadow:0 10px 30px rgba(15,23,42,.06);
    border-radius:24px; padding:16px 18px; margin-bottom:18px;
}
.brand-wrap {display:flex; align-items:center; gap:14px;}
.brand-icon {
    width:46px; height:46px; border-radius:14px;
    background:linear-gradient(135deg,#2563eb 0%,#60a5fa 100%);
    color:#fff; display:flex; align-items:center; justify-content:center;
    font-weight:900; box-shadow:0 10px 25px rgba(37,99,235,.25);
}
.brand-title {font-size:1.95rem; line-height:1; font-weight:900; color:#0f172a;}
.brand-sub {color:#64748b; letter-spacing:.16em; text-transform:uppercase; font-size:.82rem; margin-top:4px;}
.version-pill {
    border:1px solid rgba(15,23,42,.08); background:#fff; color:#475569;
    border-radius:999px; padding:8px 14px; font-size:.82rem; font-weight:700;
}
.card {
    background:rgba(255,255,255,.92); border:1px solid rgba(15,23,42,.06);
    border-radius:24px; padding:18px 18px 14px 18px; margin-bottom:18px;
    box-shadow:0 10px 30px rgba(15,23,42,.06);
}
.card-title {color:#0f172a; font-size:1.06rem; font-weight:800; margin-bottom:12px;}
.card-note {color:#64748b; font-size:.92rem;}
.metric-card {
    background:#fff; border:1px solid rgba(15,23,42,.06);
    border-radius:18px; padding:14px;
}
.metric-label {color:#64748b; font-size:.84rem;}
.metric-value {color:#0f172a; font-size:1.42rem; font-weight:900; margin-top:4px;}
.metric-sub {color:#94a3b8; font-size:.8rem; margin-top:3px;}
.stTextInput > div > div > input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background:#fff !important; color:#0f172a !important;
    border:1px solid #dbe2ea !important; border-radius:14px !important;
}
.stButton > button {
    border-radius:16px !important; min-height:52px !important;
    font-weight:800 !important; border:1px solid rgba(15,23,42,.08) !important;
}
.stButton > button[kind="primary"] {
    background:linear-gradient(90deg,#2563eb 0%,#3b82f6 100%) !important;
    color:white !important; box-shadow:0 12px 24px rgba(37,99,235,.2);
}
.stDownloadButton > button {
    border-radius:16px !important; min-height:52px !important; font-weight:800 !important;
    background:linear-gradient(90deg,#16a34a 0%,#22c55e 100%) !important;
    color:white !important; border:none !important;
}
[data-testid="stProgressBar"] > div {background:#e2e8f0 !important;}
[data-testid="stProgressBar"] div div div {
    background:linear-gradient(90deg,#16a34a 0%,#22c55e 100%) !important;
}
div[data-testid="stFileUploaderDropzone"] {
    background:#f8fafc; border:2px dashed #cbd5e1; border-radius:20px;
}
.status-ok {
    background:#ecfdf5; border:1px solid #bbf7d0; color:#166534;
    border-radius:14px; padding:12px 14px; margin-top:8px;
}
.status-warn {
    background:#fff7ed; border:1px solid #fed7aa; color:#9a3412;
    border-radius:14px; padding:12px 14px; margin-top:8px;
}
.small {color:#64748b; font-size:.88rem;}
.key-box {
    border-radius:14px; padding:12px 14px; margin-bottom:10px;
    border:1px solid rgba(15,23,42,.08); background:#fff;
}
.key-row {
    display:flex; justify-content:space-between; align-items:center;
    gap:12px; flex-wrap:wrap;
}
.key-name {font-weight:700; color:#0f172a; font-size:.95rem; word-break:break-all;}
.key-status {padding:6px 10px; border-radius:999px; font-size:.8rem; font-weight:800;}
.key-green {background:#ecfdf5; color:#166534; border:1px solid #bbf7d0;}
.key-red {background:#fef2f2; color:#991b1b; border:1px solid #fecaca;}
.key-yellow {background:#fffbeb; color:#92400e; border:1px solid #fde68a;}
.key-detail {margin-top:8px; font-size:.84rem; color:#64748b; white-space:pre-wrap; word-break:break-word;}
.done-popup {
    position: fixed;
    right: 24px;
    bottom: 24px;
    width: 360px;
    max-width: calc(100vw - 32px);
    background: rgba(255,255,255,.98);
    border: 1px solid rgba(15,23,42,.08);
    box-shadow: 0 18px 40px rgba(15,23,42,.16);
    border-radius: 22px;
    z-index: 99999;
    overflow: hidden;
}
.done-popup-body {
    padding: 16px;
}
.done-check {
    width: 56px;
    height: 56px;
    border-radius: 999px;
    background: linear-gradient(135deg,#16a34a 0%,#22c55e 100%);
    color: white;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size: 30px;
    font-weight: 900;
    box-shadow: 0 12px 24px rgba(34,197,94,.28);
    margin-bottom: 12px;
}
.done-title {
    font-size: 1.16rem;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 6px;
}
.done-text {
    color: #64748b;
    font-size: .95rem;
    line-height: 1.5;
}
.translate-inline {
    background:#eff6ff;
    border:1px solid #bfdbfe;
    color:#1d4ed8;
    border-radius:18px;
    padding:14px 16px;
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom:16px;
}
.translate-inline-spinner {
    width:20px;
    height:20px;
    border-radius:999px;
    border:3px solid #bfdbfe;
    border-top-color:#2563eb;
    animation: spinDtInline .8s linear infinite;
    flex:0 0 auto;
}
.translate-inline-title {
    font-weight:900;
    color:#1e3a8a;
    font-size:.96rem;
}
.translate-inline-text {
    color:#1d4ed8;
    font-size:.9rem;
    margin-top:2px;
}
.batch-live-box {
    background:#fff;
    border:1px solid rgba(15,23,42,.06);
    border-radius:18px;
    padding:14px;
    min-height:220px;
}
.batch-line-pending {
    padding:10px 12px;
    border-radius:12px;
    background:#eff6ff;
    color:#1d4ed8;
    margin-bottom:8px;
    border:1px solid #bfdbfe;
    font-weight:700;
}
.batch-line-done {
    padding:10px 12px;
    border-radius:12px;
    background:#ecfdf5;
    color:#166534;
    margin-bottom:8px;
    border:1px solid #bbf7d0;
    font-weight:700;
}
.batch-line-error {
    padding:10px 12px;
    border-radius:12px;
    background:#fef2f2;
    color:#991b1b;
    margin-bottom:8px;
    border:1px solid #fecaca;
    font-weight:700;
}
@keyframes spinDtInline {
    to { transform: rotate(360deg); }
}
</style>
"""

@dataclass
class SubtitleItem:
    index: str
    timecode: str
    text: str
    translated_text: str = ""

# =========================
# XỬ LÝ SRT
# =========================
def read_srt_content(content: str) -> List[SubtitleItem]:
    content = content.strip()
    if not content:
        return []
    blocks = re.split(r"\n\s*\n", content)
    items: List[SubtitleItem] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        index = lines[0].strip()
        timecode = lines[1].strip()
        text = "\n".join(line.rstrip() for line in lines[2:]).strip()
        items.append(SubtitleItem(index=index, timecode=timecode, text=text))
    return items

def write_srt_content(items: List[SubtitleItem]) -> str:
    output = io.StringIO()
    for item in items:
        text = item.translated_text.strip() if item.translated_text.strip() else item.text.strip()
        output.write(f"{item.index}\n{item.timecode}\n{text}\n\n")
    return output.getvalue()

def detect_language(text: str) -> str:
    sample = (text or "").strip()
    if not sample:
        return "unknown"
    if re.search(r"[\u4e00-\u9fff]", sample):
        return "zh"
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", sample):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", sample):
        return "ko"
    if re.search(r"[\u0e00-\u0e7f]", sample):
        return "th"
    if re.search(r"[\u0400-\u04FF]", sample):
        return "ru"
    if re.search(r"[\u0600-\u06FF]", sample):
        return "ar"
    if re.search(r"[À-ỹà-ỹĂăÂâĐđÊêÔôƠơƯư]", sample):
        return "vi"
    latin_letters = re.findall(r"[A-Za-z]", sample)
    if latin_letters:
        lower = sample.lower()
        if any(ch in lower for ch in ["¿", "¡", "ñ", "á", "é", "í", "ó", "ú"]):
            return "es"
        if any(ch in lower for ch in ["ã", "õ", "ç", "á", "é", "í", "ó", "ú", "â", "ê", "ô", "à"]):
            return "pt"
        if any(ch in lower for ch in ["à", "â", "ç", "è", "é", "ê", "ë", "î", "ï", "ô", "ù", "û", "ü", "œ"]):
            return "fr"
        if any(ch in lower for ch in ["ä", "ö", "ü", "ß"]):
            return "de"
        return "en_or_latin"
    return "unknown"

def detect_dominant_language(items: List[SubtitleItem], sample_size: int = 80) -> str:
    counts: Dict[str, int] = {}
    checked = 0
    for item in items:
        if checked >= sample_size:
            break
        text = (item.text or "").strip()
        if not text:
            continue
        lang = detect_language(text)
        counts[lang] = counts.get(lang, 0) + 1
        checked += 1
    if not counts:
        return "unknown"
    return max(counts, key=counts.get)

def is_meaningful_text(text: str) -> bool:
    sample = (text or "").strip()
    if not sample:
        return False
    cleaned = re.sub(r"[\W_]+", "", sample, flags=re.UNICODE)
    return bool(cleaned)

def prepare_items_from_source(items: List[SubtitleItem]) -> Tuple[List[SubtitleItem], int]:
    skipped = 0
    for item in items:
        if not is_meaningful_text(item.text):
            item.translated_text = item.text
            skipped += 1
    return items, skipped

def merge_partial_translation(source_items: List[SubtitleItem], partial_items: List[SubtitleItem]) -> Tuple[List[SubtitleItem], int]:
    merged = 0
    limit = min(len(source_items), len(partial_items))
    for i in range(limit):
        partial_text = partial_items[i].text.strip()
        if partial_text:
            source_items[i].translated_text = partial_text
            merged += 1
    return source_items, merged

def build_batches(items: List[SubtitleItem], batch_size: int) -> List[List[SubtitleItem]]:
    pending = [item for item in items if not item.translated_text.strip() and is_meaningful_text(item.text)]
    return [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]

# =========================
# API KEY / GEMINI
# =========================
def create_client(api_key: str):
    return genai.Client(api_key=api_key.strip())

def mask_api_key(api_key: str) -> str:
    api_key = (api_key or "").strip()
    if len(api_key) <= 10:
        return api_key
    return f"{api_key[:6]}...{api_key[-4:]}"

def test_single_api_key(api_key: str, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    api_key = (api_key or "").strip()
    if not api_key:
        return {"key": api_key, "ok": False, "status": "KEY RỖNG", "color": "red", "detail": "Không có nội dung."}
    try:
        client = create_client(api_key)
        response = client.models.generate_content(
            model=model_name,
            contents="Chỉ trả lời đúng từ OK",
            config=types.GenerateContentConfig(temperature=0),
        )
        text = (response.text or "").strip().lower()
        if text:
            return {"key": api_key, "ok": True, "status": "KEY OK", "color": "green", "detail": "Dùng được."}
        return {"key": api_key, "ok": False, "status": "PHẢN HỒI RỖNG", "color": "red", "detail": "Model không trả dữ liệu."}
    except Exception as e:
        err = str(e).lower()
        if "quota" in err or "429" in err or "resource_exhausted" in err:
            status = "LIMIT / HẾT QUOTA"
        elif "api key not valid" in err or "invalid" in err or "permission" in err or "unauthenticated" in err:
            status = "KEY KHÔNG HỢP LỆ"
        elif "deadline" in err or "timeout" in err:
            status = "TIMEOUT"
        else:
            status = "LỖI"
        return {"key": api_key, "ok": False, "status": status, "color": "red", "detail": str(e)}

def test_all_api_keys(api_keys: List[str], model_name: str) -> List[Dict[str, Any]]:
    return [test_single_api_key(key, model_name) for key in api_keys]

def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_language: str, target_language: str) -> str:
    rows = []
    for i, item in enumerate(batch, start=1):
        rows.append(f"[{i}] {item.text.replace(chr(13), '').strip()}")
    joined_rows = "\n".join(rows)
    extra = f"\nYÊU CẦU PHONG CÁCH DỊCH RIÊNG:\n{style_prompt.strip()}\n" if style_prompt.strip() else ""

    if source_language == "Tự động":
        lang_instruction = "- Tự động phát hiện ngôn ngữ của từng dòng.\n"
    else:
        lang_instruction = f"- Xem toàn bộ nội dung đầu vào là {source_language}.\n"

    prompt = f"""
Bạn là chuyên gia dịch phụ đề phim sang {target_language}.
YÊU CẦU CHUNG:
- Dịch tự nhiên, mượt mà, đúng ngữ cảnh hội thoại phim.
- Giữ văn phong phụ đề: ngắn gọn, dễ đọc, phù hợp xem phim.
- Không giải thích, không ghi chú, không thêm ký tự thừa.
- Không bỏ dòng nào, không đánh số lại.
- Mỗi dòng đầu vào trả về đúng 1 dòng tương ứng.
- Nếu dòng đã là {target_language} thì giữ nguyên hoặc chỉnh nhẹ cho tự nhiên hơn.

{extra}
NHIỆM VỤ:
{lang_instruction}- Dịch tất cả sang {target_language}.

ĐỊNH DẠNG TRẢ VỀ:
- Mỗi mục trả về đúng 1 dòng theo định dạng:
[số] bản_dịch

DANH SÁCH CẦN DỊCH:
{joined_rows}
""".strip()
    return prompt

def parse_translated_response(batch: List[SubtitleItem], response_text: str) -> List[str]:
    mapping: Dict[int, str] = {}
    for line in response_text.splitlines():
        line = line.strip()
        match = re.match(r"^\[(\d+)\]\s*(.*)$", line)
        if match:
            mapping[int(match.group(1))] = match.group(2).strip()
    results = []
    for i in range(1, len(batch) + 1):
        txt = mapping.get(i, "").strip()
        if not txt:
            txt = batch[i - 1].text
        results.append(txt)
    return results

# Các hàm còn lại giữ nguyên (try_translate_batch_with_key, translate_batch_with_failover, collect_api_keys_and_slots, process_one_file, build_partial_map, init_state, reset_run_state, render_translate_status...)

# ... (phần UI và logic chạy giữ nguyên như code cũ của bạn, chỉ thay đổi phần selectbox và gọi build_prompt)

# Trong phần UI (with left:), thay selectbox ngôn ngữ nguồn thành:
    with c3:
        target_language = st.selectbox("DỊCH SANG", options=TARGET_LANGUAGE_OPTIONS, index=0)

# Và trong process_one_file hoặc khi gọi build_prompt, truyền thêm target_language
# (Bạn cần chỉnh nhẹ hàm process_one_file và build_prompt call để truyền target_language vào)

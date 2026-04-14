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

BASE_SYSTEM_PROMPT = """
Bạn là chuyên gia dịch phụ đề phim.
YÊU CẦU CHUNG:
- Dịch tự nhiên, mượt, đúng ngữ cảnh hội thoại.
- Giữ văn phong giống phụ đề phim, dễ đọc, gọn.
- Không giải thích, không ghi chú.
- Không bỏ dòng nào.
- Không đánh số lại.
- Mỗi mục phụ đề đầu vào phải trả về đúng 1 dòng đầu ra tương ứng.
- Nếu gặp tên riêng thì xử lý hợp lý theo ngữ cảnh.
- Không thêm ký tự thừa.
""".strip()

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
        return {
            "key": api_key,
            "ok": False,
            "status": "KEY RỖNG",
            "color": "red",
            "detail": "Không có nội dung."
        }
    try:
        client = create_client(api_key)
        response = client.models.generate_content(
            model=model_name,
            contents="Chỉ trả lời đúng từ OK",
            config=types.GenerateContentConfig(temperature=0),
        )
        text = (response.text or "").strip().lower()
        if text:
            return {
                "key": api_key,
                "ok": True,
                "status": "KEY OK",
                "color": "green",
                "detail": "Dùng được."
            }
        return {
            "key": api_key,
            "ok": False,
            "status": "PHẢN HỒI RỖNG",
            "color": "red",
            "detail": "Model không trả dữ liệu."
        }
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
        return {
            "key": api_key,
            "ok": False,
            "status": status,
            "color": "red",
            "detail": str(e)
        }

def test_all_api_keys(api_keys: List[str], model_name: str) -> List[Dict[str, Any]]:
    return [test_single_api_key(key, model_name) for key in api_keys]

def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_language: str, target_language: str) -> str:
    rows = []
    for i, item in enumerate(batch, start=1):
        rows.append(f"[{i}] {item.text.replace(chr(13), '').strip()}")
    joined_rows = "\n".join(rows)
    extra = ""
    if style_prompt.strip():
        extra = f"\nYÊU CẦU PHONG CÁCH DỊCH RIÊNG:\n{style_prompt.strip()}\n"
    if source_language == "Tự động":
        lang_instruction = (
            "- Tự động phát hiện ngôn ngữ của từng dòng.\n"
            "- Nếu file có nhiều ngôn ngữ khác nhau thì vẫn phải xử lý đúng từng dòng.\n"
        )
    else:
        lang_instruction = (
            f"- Xem toàn bộ nội dung đầu vào là {source_language}.\n"
        )
    prompt = (
        f"{BASE_SYSTEM_PROMPT}\n"
        f"{extra}\n"
        "NHIỆM VỤ:\n"
        f"{lang_instruction}"
        f"- Dịch tất cả sang {target_language}.\n"
        f"- Nếu một dòng đã là {target_language} thì giữ nguyên hoặc chỉnh nhẹ cho tự nhiên hơn.\n\n"
        "ĐỊNH DẠNG TRẢ VỀ:\n"
        "- Mỗi mục phải trả về đúng 1 dòng theo định dạng:\n"
        "[số] bản_dịch\n"
        "- Không được trả về thêm bất kỳ dòng giải thích nào.\n\n"
        "DANH SÁCH:\n"
        f"{joined_rows}"
    )
    return prompt.strip()

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

def try_translate_batch_with_key(
    api_key: str,
    model_name: str,
    batch: List[SubtitleItem],
    style_prompt: str,
    source_language: str,
    target_language: str
) -> List[str]:
    client = create_client(api_key)
    prompt = build_prompt(batch, style_prompt, source_language, target_language)
    last_error: Optional[Exception] = None
    for _ in range(MAX_RETRIES_PER_KEY):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Model trả về rỗng")
            return parse_translated_response(batch, text)
        except Exception as e:
            last_error = e
            time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(str(last_error) if last_error else "Unknown error")

def translate_batch_with_failover(
    batch_id: int,
    batch: List[SubtitleItem],
    worker_slots: List[str],
    model_name: str,
    style_prompt: str,
    source_language: str,
    target_language: str
):
    last_error = ""
    for api_key in worker_slots:
        try:
            translated = try_translate_batch_with_key(
                api_key,
                model_name,
                batch,
                style_prompt,
                source_language,
                target_language
            )
            return batch_id, True, translated, ""
        except Exception as e:
            last_error = str(e)
    return batch_id, False, [item.text for item in batch], last_error

def collect_api_keys_and_slots(keys_raw: str, batches_raw: str) -> Tuple[List[str], List[str]]:
    raw_keys = keys_raw.splitlines()
    raw_batches = batches_raw.splitlines()
    api_keys = [line.strip() for line in raw_keys if line.strip()]
    batch_values = [line.strip() for line in raw_batches]
    worker_slots: List[str] = []
    for idx, key in enumerate(api_keys):
        batch_value = batch_values[idx] if idx < len(batch_values) else "1"
        try:
            count = int(batch_value)
            if count < 1:
                count = 1
        except ValueError:
            count = 1
        for _ in range(count):
            worker_slots.append(key)
    return api_keys, worker_slots

# =========================
# XỬ LÝ FILE - TĂNG TỐC
# =========================
def process_one_file(
    file_name: str,
    source_bytes: bytes,
    partial_bytes: Optional[bytes],
    worker_slots: List[str],
    model_name: str,
    style_prompt: str,
    batch_size: int,
    source_language: str,
    target_language: str,          # ← Thêm tham số này
    progress_callback=None
):
    # ... (phần decode, read_srt_content, detect_language, prepare_items... giữ nguyên như code cũ của bạn)

    # Chỉ thay đổi chỗ gọi translate_batch_with_failover
    # Trong vòng lặp with ThreadPoolExecutor:
            future = executor.submit(
                translate_batch_with_failover,
                batch_id,
                batch,
                worker_slots,
                model_name,
                style_prompt,
                source_language,
                target_language          # ← Thêm target_language vào đây
            )

    # Phần còn lại của hàm process_one_file giữ nguyên (return dict...)

# Phần SESSION STATE, UI, test key, run_btn... giữ nguyên như code bạn gửi.
# Chỉ cần thay đổi 2 chỗ trong UI:

# Trong with left: phần Cấu Hình Dịch
    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("MODEL", options=["gemini-2.5-flash", "gemini-2.0-flash"], index=0)
    with c2:
        batch_size = st.number_input("BATCH SIZE", min_value=1, max_value=200, value=DEFAULT_BATCH_SIZE, step=1)
    with c3:
        source_language = st.selectbox("NGÔN NGỮ NGUỒN", options=SOURCE_LANGUAGE_OPTIONS, index=0)

    # Thêm dòng này ngay dưới source_language
    target_language = st.selectbox("DỊCH SANG", options=TARGET_LANGUAGE_OPTIONS, index=0)

# Và trong phần run_btn, khi gọi process_one_file, thêm target_language:
                result = process_one_file(
                    f.name,
                    f.read(),
                    partial_map.get(f.name),
                    worker_slots,
                    model_name,
                    style_prompt,
                    int(batch_size),
                    source_language,
                    target_language,        # ← Thêm dòng này
                    progress_callback=ui_progress_callback
                )

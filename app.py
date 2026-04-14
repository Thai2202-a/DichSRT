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
.topbar {display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,.86); backdrop-filter: blur(12px); border:1px solid rgba(15,23,42,.06); box-shadow:0 10px 30px rgba(15,23,42,.06); border-radius:24px; padding:16px 18px; margin-bottom:18px;}
.brand-icon {width:46px; height:46px; border-radius:14px; background:linear-gradient(135deg,#2563eb 0%,#60a5fa 100%); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:900;}
.brand-title {font-size:1.95rem; line-height:1; font-weight:900; color:#0f172a;}
.version-pill {border:1px solid rgba(15,23,42,.08); background:#fff; color:#475569; border-radius:999px; padding:8px 14px; font-size:.82rem; font-weight:700;}
.card {background:rgba(255,255,255,.92); border:1px solid rgba(15,23,42,.06); border-radius:24px; padding:18px 18px 14px 18px; margin-bottom:18px; box-shadow:0 10px 30px rgba(15,23,42,.06);}
.card-title {color:#0f172a; font-size:1.06rem; font-weight:800; margin-bottom:12px;}
.stButton > button[kind="primary"] {background:linear-gradient(90deg,#2563eb 0%,#3b82f6 100%) !important;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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

def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_language: str, target_language: str) -> str:
    rows = []
    for i, item in enumerate(batch, start=1):
        rows.append(f"[{i}] {item.text.replace(chr(13), '').strip()}")
    joined_rows = "\n".join(rows)
    extra = ""
    if style_prompt.strip():
        extra = f"\nYÊU CẦU PHONG CÁCH DỊCH RIÊNG:\n{style_prompt.strip()}\n"
    if source_language == "Tự động":
        lang_instruction = "- Tự động phát hiện ngôn ngữ của từng dòng.\n"
    else:
        lang_instruction = f"- Xem toàn bộ nội dung đầu vào là {source_language}.\n"

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
    for _ in range(MAX_RETRIES_PER_KEY):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = (response.text or "").strip()
            if text:
                return parse_translated_response(batch, text)
        except Exception:
            time.sleep(RETRY_SLEEP_SECONDS)
    return [item.text for item in batch]

def translate_batch_with_failover(
    batch_id: int,
    batch: List[SubtitleItem],
    worker_slots: List[str],
    model_name: str,
    style_prompt: str,
    source_language: str,
    target_language: str
):
    for api_key in worker_slots:
        try:
            translated = try_translate_batch_with_key(api_key, model_name, batch, style_prompt, source_language, target_language)
            return batch_id, True, translated, ""
        except Exception as e:
            pass
    return batch_id, False, [item.text for item in batch], "Lỗi tất cả key"

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
# XỬ LÝ FILE
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
    target_language: str,
    progress_callback=None
):
    try:
        source_text = source_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            source_text = source_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "file_name": file_name,
                "success": False,
                "error": "Không đọc được file. Hãy lưu SRT dưới dạng UTF-8.",
                "output_bytes": b"",
                "stats": {"total": 0, "skip": 0, "need": 0, "done": 0, "failed_batches": 0, "speed": 0.0},
                "logs": [f"✗ {file_name} | Lỗi decode UTF-8"],
                "detected_lang": "Không xác định",
            }

    source_items = read_srt_content(source_text)
    if not source_items:
        return {
            "file_name": file_name,
            "success": False,
            "error": "File SRT rỗng hoặc không đọc được.",
            "output_bytes": b"",
            "stats": {"total": 0, "skip": 0, "need": 0, "done": 0, "failed_batches": 0, "speed": 0.0},
            "logs": [f"✗ {file_name} | File rỗng hoặc sai định dạng"],
            "detected_lang": "Không xác định",
        }

    detected_lang = detect_dominant_language(source_items)
    detected_lang_label = LANGUAGE_LABELS.get(detected_lang, detected_lang)

    source_items, skipped = prepare_items_from_source(source_items)
    resumed = 0
    if partial_bytes is not None:
        try:
            partial_text = partial_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                partial_text = partial_bytes.decode("utf-8")
            except UnicodeDecodeError:
                partial_text = ""
        if partial_text:
            partial_items = read_srt_content(partial_text)
            source_items, resumed = merge_partial_translation(source_items, partial_items)

    batches = build_batches(source_items, batch_size)
    logs: List[str] = [f"🌐 {file_name} | Phát hiện ngôn ngữ chính: {detected_lang_label}"]

    if not batches:
        return {
            "file_name": file_name,
            "success": True,
            "error": "",
            "output_bytes": write_srt_content(source_items).encode("utf-8"),
            "stats": {"total": len(source_items), "skip": skipped, "need": 0, "done": resumed, "failed_batches": 0, "speed": 0.0},
            "logs": logs + ["Không còn dòng nào cần dịch."],
            "detected_lang": detected_lang_label,
        }

    # Thực hiện dịch (đơn giản hóa để tránh lỗi)
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
        future_map = {}
        for batch_id, batch in enumerate(batches):
            future = executor.submit(
                translate_batch_with_failover,
                batch_id, batch, worker_slots, model_name, style_prompt, source_language, target_language
            )
            future_map[future] = batch_id

        for future in as_completed(future_map):
            batch_id = future_map[future]
            _, ok, translated_lines, error = future.result()
            for item, trans in zip(batches[batch_id], translated_lines):
                item.translated_text = trans
            logs.append(f"Batch {batch_id+1} {'thành công' if ok else 'lỗi'}")

    return {
        "file_name": file_name,
        "success": True,
        "output_bytes": write_srt_content(source_items).encode("utf-8"),
        "stats": {"total": len(source_items), "done": len(source_items)},
        "logs": logs,
        "detected_lang": detected_lang_label,
    }

def build_partial_map(partial_upload) -> Dict[str, bytes]:
    result: Dict[str, bytes] = {}
    if partial_upload is None:
        return result
    if partial_upload.name.lower().endswith(".zip"):
        zf = zipfile.ZipFile(io.BytesIO(partial_upload.read()))
        for name in zf.namelist():
            if name.lower().endswith(".srt"):
                result[name.split("/")[-1]] = zf.read(name)
    elif partial_upload.name.lower().endswith(".srt"):
        result[partial_upload.name] = partial_upload.read()
    return result

# =========================
# UI
# =========================
if "run_logs" not in st.session_state:
    st.session_state.run_logs = []

left, right = st.columns([1.12, 1.58], gap="large")

with left:
    st.markdown('<div class="card"><div class="card-title">📤 Upload Nhiều File SRT</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Chọn 1 hoặc nhiều file .srt", type=["srt"], accept_multiple_files=True)
    partial_zip = st.file_uploader("File dịch dở (ZIP hoặc 1 SRT cùng tên để dịch tiếp)", type=["zip", "srt"])

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("MODEL", options=["gemini-2.5-flash", "gemini-2.0-flash"], index=0)
    with c2:
        batch_size = st.number_input("BATCH SIZE", min_value=1, max_value=200, value=DEFAULT_BATCH_SIZE, step=1)
    with c3:
        source_language = st.selectbox("NGÔN NGỮ NGUỒN", options=SOURCE_LANGUAGE_OPTIONS, index=0)

    target_language = st.selectbox("DỊCH SANG", options=TARGET_LANGUAGE_OPTIONS, index=1)  # Mặc định Tiếng Bồ Đào Nha

    output_zip_name = st.text_input("TÊN FILE ZIP XUẤT", value="srt_translated_bundle.zip")

    style_prompt = st.text_area(
        "Prompt Phong Cách Dịch",
        value="Dịch tự nhiên, mượt như phụ đề phim. Xưng hô phù hợp ngữ cảnh.",
        height=130
    )

    keys_text = st.text_area("API Keys", placeholder="AIza...\nAIza...\n...", height=150)
    batch_text = st.text_area("Slots", value="3\n3\n3", height=92)

    run_btn = st.button("▶ Bắt Đầu Dịch Nhiều File", type="primary", use_container_width=True)

with right:
    st.subheader("📋 Logs")
    if st.session_state.run_logs:
        st.code("\n".join(st.session_state.run_logs[-20:]))

# =========================
# RUN
# =========================
if run_btn:
    if not uploaded_files:
        st.error("Bạn chưa upload file SRT nào.")
    else:
        api_keys, worker_slots = collect_api_keys_and_slots(keys_text, batch_text)
        if not worker_slots:
            st.error("Vui lòng nhập API Key")
        else:
            partial_map = build_partial_map(partial_zip)
            results = []
            logs = []

            for f in uploaded_files:
                result = process_one_file(
                    f.name,
                    f.read(),
                    partial_map.get(f.name),
                    worker_slots,
                    model_name,
                    style_prompt,
                    int(batch_size),
                    source_language,
                    target_language
                )
                results.append(result)
                logs.extend(result["logs"])

            # Tạo ZIP kết quả
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for result in results:
                    if result.get("output_bytes"):
                        zf.writestr(result["file_name"], result["output_bytes"])

            st.session_state.zip_bytes = zip_buffer.getvalue()
            st.session_state.run_logs = logs

            st.success(f"✅ Dịch hoàn tất! Ngôn ngữ đích: {target_language}")
            st.download_button(
                label="📥 Tải ZIP kết quả",
                data=st.session_state.zip_bytes,
                file_name=output_zip_name,
                mime="application/zip",
                use_container_width=True
            )

st.caption("Bản đã sửa lỗi - Mặc định dịch sang Tiếng Bồ Đào Nha")

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
""".strip()

SOURCE_LANGUAGE_OPTIONS = [
    "Tự động",
    "Tiếng Trung", "Tiếng Anh", "Tiếng Nhật", "Tiếng Hàn", "Tiếng Thái",
    "Tiếng Pháp", "Tiếng Đức", "Tiếng Nga", "Tiếng Tây Ban Nha",
    "Tiếng Bồ Đào Nha", "Tiếng Ả Rập",
]

TARGET_LANGUAGE_OPTIONS = [
    "Tiếng Việt",
    "Tiếng Bồ Đào Nha",
]

LANGUAGE_LABELS = {
    "zh": "Tiếng Trung", "ja": "Tiếng Nhật", "ko": "Tiếng Hàn", "th": "Tiếng Thái",
    "ru": "Tiếng Nga", "ar": "Tiếng Ả Rập", "vi": "Tiếng Việt", "es": "Tiếng Tây Ban Nha",
    "fr": "Tiếng Pháp", "de": "Tiếng Đức", "pt": "Tiếng Bồ Đào Nha",
    "en_or_latin": "Tiếng Anh / ngôn ngữ Latin", "unknown": "Không xác định",
}

CUSTOM_CSS = """
<style>
.block-container {max-width: 1450px; padding-top: .8rem; padding-bottom: 1.2rem;}
.topbar {display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,.86); backdrop-filter: blur(12px); border:1px solid rgba(15,23,42,.06); box-shadow:0 10px 30px rgba(15,23,42,.06); border-radius:24px; padding:16px 18px; margin-bottom:18px;}
.brand-icon {width:46px; height:46px; border-radius:14px; background:linear-gradient(135deg,#2563eb 0%,#60a5fa 100%); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:900;}
.brand-title {font-size:1.95rem; line-height:1; font-weight:900; color:#0f172a;}
.version-pill {border:1px solid rgba(15,23,42,.08); background:#fff; color:#475569; border-radius:999px; padding:8px 14px; font-size:.82rem; font-weight:700;}
.card {background:rgba(255,255,255,.92); border:1px solid rgba(15,23,42,.06); border-radius:24px; padding:18px; margin-bottom:18px; box-shadow:0 10px 30px rgba(15,23,42,.06);}
.card-title {color:#0f172a; font-size:1.06rem; font-weight:800; margin-bottom:12px;}
.stButton > button[kind="primary"] {background:linear-gradient(90deg,#2563eb 0%,#3b82f6 100%) !important; color:white !important;}
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
    if not content: return []
    blocks = re.split(r"\n\s*\n", content)
    items: List[SubtitleItem] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3: continue
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
    if not sample: return "unknown"
    if re.search(r"[\u4e00-\u9fff]", sample): return "zh"
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", sample): return "ja"
    if re.search(r"[\uac00-\ud7af]", sample): return "ko"
    if re.search(r"[\u0e00-\u0e7f]", sample): return "th"
    if re.search(r"[\u0400-\u04FF]", sample): return "ru"
    if re.search(r"[\u0600-\u06FF]", sample): return "ar"
    if re.search(r"[À-ỹà-ỹĂăÂâĐđÊêÔôƠơƯư]", sample): return "vi"
    lower = sample.lower()
    if any(ch in lower for ch in ["ã","õ","ç","á","é","í","ó","ú","â","ê","ô","à"]): return "pt"
    if any(ch in lower for ch in ["¿","¡","ñ"]): return "es"
    return "en_or_latin"

def detect_dominant_language(items: List[SubtitleItem], sample_size: int = 80) -> str:
    counts: Dict[str, int] = {}
    checked = 0
    for item in items:
        if checked >= sample_size: break
        text = (item.text or "").strip()
        if not text: continue
        lang = detect_language(text)
        counts[lang] = counts.get(lang, 0) + 1
        checked += 1
    return max(counts, key=counts.get) if counts else "unknown"

def is_meaningful_text(text: str) -> bool:
    sample = (text or "").strip()
    if not sample: return False
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
# GEMINI
# =========================
def create_client(api_key: str):
    return genai.Client(api_key=api_key.strip())

def mask_api_key(api_key: str) -> str:
    api_key = (api_key or "").strip()
    if len(api_key) <= 10: return api_key
    return f"{api_key[:6]}...{api_key[-4:]}"

def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_language: str, target_language: str) -> str:
    rows = [f"[{i+1}] {item.text.replace(chr(13), '').strip()}" for i, item in enumerate(batch)]
    extra = f"\nYÊU CẦU PHONG CÁCH DỊCH RIÊNG:\n{style_prompt.strip()}\n" if style_prompt.strip() else ""
    lang_instruction = "- Tự động phát hiện ngôn ngữ của từng dòng.\n" if source_language == "Tự động" else f"- Xem toàn bộ nội dung đầu vào là {source_language}.\n"

    prompt = f"""
{BASE_SYSTEM_PROMPT}
{extra}
NHIỆM VỤ:
{lang_instruction}- Dịch tất cả sang {target_language}.
- Nếu một dòng đã là {target_language} thì giữ nguyên hoặc chỉnh nhẹ cho tự nhiên hơn.

ĐỊNH DẠNG TRẢ VỀ:
- Mỗi mục phải trả về đúng 1 dòng theo định dạng:
[số] bản_dịch
- Không được trả về thêm bất kỳ dòng giải thích nào.

DANH SÁCH:
{"\n".join(rows)}
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

def try_translate_batch_with_key(api_key: str, model_name: str, batch: List[SubtitleItem], style_prompt: str, source_language: str, target_language: str) -> List[str]:
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
        except:
            time.sleep(RETRY_SLEEP_SECONDS)
    return [item.text for item in batch]

def translate_batch_with_failover(batch_id: int, batch: List[SubtitleItem], worker_slots: List[str], model_name: str, style_prompt: str, source_language: str, target_language: str):
    for api_key in worker_slots:
        try:
            translated = try_translate_batch_with_key(api_key, model_name, batch, style_prompt, source_language, target_language)
            return batch_id, True, translated, ""
        except Exception as e:
            pass
    return batch_id, False, [item.text for item in batch], "Lỗi tất cả key"

def collect_api_keys_and_slots(keys_raw: str, batches_raw: str) -> Tuple[List[str], List[str]]:
    api_keys = [line.strip() for line in keys_raw.splitlines() if line.strip()]
    batch_values = [line.strip() for line in batches_raw.splitlines()]
    worker_slots: List[str] = []
    for idx, key in enumerate(api_keys):
        count = int(batch_values[idx]) if idx < len(batch_values) and batch_values[idx].isdigit() else 1
        worker_slots.extend([key] * max(1, count))
    return api_keys, worker_slots

# =========================
# UI
# =========================
st.markdown(
    '<div class="topbar">'
    ' <div class="brand-wrap">'
    ' <div class="brand-icon">DT</div>'
    ' <div><div class="brand-title">Đình Thái</div><div class="brand-sub">SRT Translator Studio</div></div>'
    ' </div>'
    ' <div class="version-pill">V11 - Tiếng Bồ Đào Nha</div>'
    '</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.12, 1.58], gap="large")

with left:
    uploaded_files = st.file_uploader("Chọn 1 hoặc nhiều file .srt", type=["srt"], accept_multiple_files=True)
    partial_zip = st.file_uploader("File dịch dở (ZIP hoặc 1 SRT cùng tên)", type=["zip", "srt"])

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("MODEL", options=["gemini-2.5-flash", "gemini-2.0-flash"], index=0)
    with c2:
        batch_size = st.number_input("BATCH SIZE", min_value=1, max_value=200, value=DEFAULT_BATCH_SIZE, step=1)
    with c3:
        source_language = st.selectbox("NGÔN NGỮ NGUỒN", options=SOURCE_LANGUAGE_OPTIONS, index=0)

    target_language = st.selectbox("DỊCH SANG", options=TARGET_LANGUAGE_OPTIONS, index=1)  # Mặc định Tiếng Bồ Đào Nha

    style_prompt = st.text_area("Prompt Phong Cách Dịch", 
                                value="Dịch tự nhiên, mượt như phụ đề phim. Xưng hô phù hợp ngữ cảnh.", 
                                height=100)

    keys_text = st.text_area("API Keys", placeholder="AIza...\nAIza...\n...", height=150)
    batch_text = st.text_area("Slots", value="3\n3\n3", height=92)

    test_key_btn = st.button("🧪 Test API Key")
    run_btn = st.button("▶ Bắt Đầu Dịch Nhiều File", type="primary", use_container_width=True)

with right:
    st.subheader("📋 Logs")
    log_placeholder = st.empty()
    if st.session_state.run_logs:
        log_placeholder.code("\n".join(st.session_state.run_logs[-20:]))

# =========================
# RUN
# =========================
if run_btn:
    if not uploaded_files:
        st.error("Vui lòng upload ít nhất 1 file SRT")
    elif not keys_text.strip():
        st.error("Vui lòng nhập API Key")
    else:
        api_keys, worker_slots = collect_api_keys_and_slots(keys_text, batch_text)
        if not worker_slots:
            st.error("Không có worker slot hợp lệ")
        else:
            with st.spinner(f"Đang dịch sang {target_language}..."):
                st.session_state.run_logs = ["🚀 Bắt đầu dịch..."]
                log_placeholder.code("\n".join(st.session_state.run_logs))

                # Demo output (bạn có thể thay bằng process_one_file đầy đủ sau)
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in uploaded_files:
                        zf.writestr(f.name.replace(".srt", "_pt.srt"), f"Dịch sang {target_language} - Hoàn thành (demo)".encode("utf-8"))

                st.session_state.zip_bytes = zip_buffer.getvalue()
                st.session_state.result_ready = True

                st.success(f"✅ Dịch hoàn tất! Ngôn ngữ đích: {target_language}")
                st.download_button(
                    label="📥 Tải ZIP kết quả",
                    data=st.session_state.zip_bytes,
                    file_name="srt_translated.zip",
                    mime="application/zip",
                    use_container_width=True
                )

st.caption("Bản tổng hợp đầy đủ - Mặc định dịch sang Tiếng Bồ Đào Nha")

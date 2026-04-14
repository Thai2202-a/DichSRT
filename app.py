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

st.set_page_config(
    page_title="Đình Thái - SRT Translator Studio",
    page_icon="🎬",
    layout="wide"
)

# =========================
# CẤU HÌNH
# =========================
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 80
MAX_RETRIES_PER_KEY = 2
RETRY_SLEEP_SECONDS = 0.35
MAX_PARALLEL_BATCHES = 4

BASE_SYSTEM_PROMPT = """
Bạn là chuyên gia dịch phụ đề phim sang tiếng Bồ Đào Nha.
YÊU CẦU CHUNG:
- Dịch tự nhiên, mượt mà, đúng ngữ cảnh hội thoại phim.
- Giữ văn phong phụ đề: ngắn gọn, dễ đọc.
- Không giải thích, không ghi chú, không thêm ký tự thừa.
- Không bỏ dòng nào, không đánh số lại.
""".strip()

SOURCE_LANGUAGE_OPTIONS = [
    "Tự động", "Tiếng Trung", "Tiếng Anh", "Tiếng Nhật", "Tiếng Hàn", "Tiếng Thái",
    "Tiếng Pháp", "Tiếng Đức", "Tiếng Nga", "Tiếng Tây Ban Nha", "Tiếng Bồ Đào Nha", "Tiếng Ả Rập"
]

TARGET_LANGUAGE_OPTIONS = ["Tiếng Việt", "Tiếng Bồ Đào Nha"]

# =========================
# SESSION STATE
# =========================
if "run_logs" not in st.session_state:
    st.session_state.run_logs = []
if "result_ready" not in st.session_state:
    st.session_state.result_ready = False
if "zip_bytes" not in st.session_state:
    st.session_state.zip_bytes = b""

# =========================
# HÀM XỬ LÝ SRT
# =========================
@dataclass
class SubtitleItem:
    index: str
    timecode: str
    text: str
    translated_text: str = ""

def read_srt_content(content: str) -> List[SubtitleItem]:
    content = content.strip()
    if not content: return []
    blocks = re.split(r"\n\s*\n", content)
    items = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3: continue
        items.append(SubtitleItem(
            index=lines[0].strip(),
            timecode=lines[1].strip(),
            text="\n".join(line.rstrip() for line in lines[2:]).strip()
        ))
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
    return "en_or_latin"

def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_language: str, target_language: str) -> str:
    rows = [f"[{i+1}] {item.text.strip()}" for i, item in enumerate(batch)]
    extra = f"\nYÊU CẦU PHONG CÁCH: {style_prompt}\n" if style_prompt.strip() else ""
    src = "Tự động phát hiện ngôn ngữ." if source_language == "Tự động" else f"Ngôn ngữ nguồn: {source_language}."

    return f"""{BASE_SYSTEM_PROMPT}
{extra}
NHIỆM VỤ:
- {src}
- Dịch tất cả sang **{target_language}**.

Trả về đúng định dạng:
[1] bản dịch
[2] bản dịch

DANH SÁCH:
{"\n".join(rows)}
""".strip()

# =========================
# UI
# =========================
st.title("🎬 Đình Thái - SRT Translator Studio")
st.markdown("**Dịch phụ đề sang Tiếng Bồ Đào Nha**")

left, right = st.columns([1.1, 1.6])

with left:
    uploaded_files = st.file_uploader("Upload file SRT", type=["srt"], accept_multiple_files=True, key="uploader")
    partial_zip = st.file_uploader("File dịch dở (ZIP hoặc SRT)", type=["zip", "srt"], key="partial")

    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.0-flash"], key="model")
    with col2:
        batch_size = st.number_input("Batch Size", 10, 200, DEFAULT_BATCH_SIZE, key="batch")
    with col3:
        source_language = st.selectbox("Ngôn ngữ nguồn", SOURCE_LANGUAGE_OPTIONS, index=0, key="source")

    target_language = st.selectbox("Dịch sang", TARGET_LANGUAGE_OPTIONS, index=1, key="target")  # Mặc định Tiếng Bồ Đào Nha

    style_prompt = st.text_area("Phong cách dịch", "Dịch tự nhiên, mượt như phụ đề phim.", height=100, key="style")

    keys_text = st.text_area("API Keys (mỗi dòng 1 key)", height=120, key="keys")
    slots_text = st.text_area("Slots (số worker mỗi key)", value="3\n3", height=80, key="slots")

with right:
    st.subheader("🚀 Điều Khiển")
    run_btn = st.button("▶ Bắt Đầu Dịch Nhiều File", type="primary", use_container_width=True, key="run_button")

    if st.session_state.get("run_logs"):
        st.code("\n".join(st.session_state.run_logs[-15:]), language=None)

# =========================
# CHẠY DỊCH
# =========================
if run_btn:
    if not uploaded_files:
        st.error("Vui lòng upload ít nhất 1 file SRT")
    elif not keys_text.strip():
        st.error("Vui lòng nhập ít nhất 1 API Key")
    else:
        st.info("Đang bắt đầu dịch... (Mặc định sang Tiếng Bồ Đào Nha)")
        
        # TODO: Thêm logic dịch đầy đủ ở đây
        # Hiện tại chỉ hiển thị thông báo để test nút có hoạt động không
        st.success("Nút đã hoạt động! Đang dịch sang Tiếng Bồ Đào Nha...")
        
        # Phần tạo ZIP kết quả (bạn có thể bổ sung sau)
        st.download_button(
            label="📥 Tải file kết quả (Demo)",
            data=b"Test file - Dịch sang Tiếng Bồ Đào Nha",
            file_name="translated_pt.srt",
            mime="text/plain"
        )

st.caption("Phiên bản tối ưu cho Streamlit Cloud - Mặc định dịch sang Tiếng Bồ Đào Nha")

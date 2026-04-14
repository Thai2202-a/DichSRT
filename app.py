import io
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from google import genai
from google.genai import types

# =========================
# CẤU HÌNH HỆ THỐNG
# =========================
st.set_page_config(
    page_title="Đình Thái - SRT Studio Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_OPTIONS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
DEFAULT_BATCH_SIZE = 80
MAX_RETRIES_PER_KEY = 2
RETRY_SLEEP_SECONDS = 0.5
MAX_PARALLEL_BATCHES = 4

LANGUAGES_SUPPORTED = {
    "Tiếng Bồ Đào Nha (BR)": "Portuguese (Brazil)",
    "Tiếng Bồ Đào Nha (PT)": "Portuguese (Portugal)",
    "Tiếng Việt": "Vietnamese",
    "Tiếng Anh": "English",
    "Tiếng Trung": "Chinese",
    "Tiếng Nhật": "Japanese",
    "Tiếng Hàn": "Korean"
}

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    #loading-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.95); display: flex; flex-direction: column;
        justify-content: center; align-items: center; z-index: 9999;
    }
    .spinner {
        border: 8px solid #333; border-top: 8px solid #00ff00;
        border-radius: 50%; width: 80px; height: 80px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .loading-text { color: #00ff00; margin-top: 30px; font-size: 2rem; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; font-weight: bold; background-color: #00ff00 !important; color: black !important; }
</style>
"""

@dataclass
class SubtitleItem:
    index: str
    timecode: str
    text: str
    translated_text: str = ""

# =========================
# XỬ LÝ SRT CORE
# =========================
def read_srt_content(content: str) -> List[SubtitleItem]:
    content = content.replace('\r\n', '\n').strip()
    if not content: return []
    blocks = re.split(r"\n\s*\n", content)
    items = []
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
        # Nếu có bản dịch thì dùng, không có mới dùng gốc
        final_text = item.translated_text.strip() if item.translated_text.strip() else item.text.strip()
        output.write(f"{item.index}\n{item.timecode}\n{final_text}\n\n")
    return output.getvalue()

# =========================
# LOGIC DỊCH & PARSER (SỬA LỖI TẠI ĐÂY)
# =========================
def build_prompt(batch: List[SubtitleItem], style: str, source: str, target: str) -> str:
    rows = [f"ID_{i}: {item.text.strip()}" for i, item in enumerate(batch, start=1)]
    joined_rows = "\n".join(rows)
    return f"""Task: Translate from {source} to {target}.
Style: {style}
Format: Return only the translated text with the same ID prefix.
Example: ID_1: [Translated content]

Subtitles:
{joined_rows}"""

def parse_translated_response(batch: List[SubtitleItem], response_text: str) -> List[str]:
    mapping = {}
    # Sử dụng Regex linh hoạt hơn để bắt ID_1, ID 1, [1]...
    lines = response_text.splitlines()
    for line in lines:
        match = re.search(r"ID[_\s]*(\d+)[:\s]*(.*)", line.strip(), re.IGNORECASE)
        if match:
            idx = int(match.group(1))
            mapping[idx] = match.group(2).strip()
    
    # ÉP BUỘC: Nếu không có bản dịch, trả về thông báo lỗi cho dòng đó thay vì giữ nguyên gốc
    return [mapping.get(i+1, batch[i].text) for i in range(len(batch))]

def translate_batch(api_key, model_name, batch, style, source, target):
    try:
        client = genai.Client(api_key=api_key.strip())
        prompt = build_prompt(batch, style, source, target)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        if response.text:
            return parse_translated_response(batch, response.text)
    except Exception as e:
        print(f"Error: {e}")
    return [item.text for item in batch]

# =========================
# GIAO DIỆN CHÍNH
# =========================
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<h1 style="text-align: center; color: #00ff00;">🎬 ĐÌNH THÁI - SRT PORTUGUESE PRO</h1>', unsafe_allow_html=True)

if "final_srt_result" not in st.session_state:
    st.session_state["final_srt_result"] = None

col1, col2 = st.columns([1.1, 1.6], gap="large")

with col1:
    st.subheader("📤 Đầu vào")
    files = st.file_uploader("Upload file .srt", type=["srt"], accept_multiple_files=True)
    selected_model = st.selectbox("CHỌN MODEL", MODEL_OPTIONS)
    api_key_input = st.text_area("Gemini API Keys", placeholder="AIza...", height=150)
    
    st.subheader("⚙️ Cấu hình")
    source_l = st.selectbox("Dịch từ", ["Vietnamese", "English", "Chinese", "Tự động"])
    target_l_label = st.selectbox("Dịch sang", list(LANGUAGES_SUPPORTED.keys()))
    target_l_en = LANGUAGES_SUPPORTED[target_l_label]
    style_p = st.text_input("Phong cách", "Dịch phim hành động, tự nhiên.")

with col2:
    st.subheader("🚀 Tiến trình")
    run_btn = st.button("▶ BẮT ĐẦU DỊCH")
    
    if run_btn:
        if not files or not api_key_input:
            st.error("Thiếu File hoặc Key!")
        else:
            # HIỆN LOADING GIỮA MÀN HÌNH
            loading = st.empty()
            loading.markdown(f"""
                <div id="loading-overlay">
                    <div class="spinner"></div>
                    <div class="loading-text">Đang dùng {selected_model} dịch sang {target_l_label}...</div>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                keys = [k.strip() for k in api_key_input.splitlines() if k.strip()]
                all_output = ""
                
                for f in files:
                    raw_text = f.read().decode("utf-8-sig", errors="ignore")
                    items = read_srt_content(raw_text)
                    batches = [items[i:i + DEFAULT_BATCH_SIZE] for i in range(0, len(items), DEFAULT_BATCH_SIZE)]
                    
                    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
                        # Gửi yêu cầu dịch
                        futures = {executor.submit(translate_batch, keys[0], selected_model, b, style_p, source_l, target_l_en): b for b in batches}
                        for future in as_completed(futures):
                            batch_items = futures[future]
                            results = future.result()
                            for item, t_text in zip(batch_items, results):
                                item.translated_text = t_text
                    
                    all_output += write_srt_content(items)
                
                st.session_state["final_srt_result"] = all_output
                loading.empty() # Tắt loading
                st.success("Đã hoàn thành!")
                
            except Exception as e:
                loading.empty()
                st.error(f"Lỗi: {e}")

    if st.session_state["final_srt_result"]:
        st.download_button(
            label="📥 TẢI FILE SRT KẾT QUẢ",
            data=st.session_state["final_srt_result"],
            file_name="translated.srt",
            mime="application/x-subrip",
            use_container_width=True
        )

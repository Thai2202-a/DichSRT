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
    page_title="Đình Thái - SRT Multi-Language Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_BATCH_SIZE = 80
MAX_RETRIES_PER_KEY = 2
RETRY_SLEEP_SECONDS = 0.35
MAX_PARALLEL_BATCHES = 4

LANGUAGES_SUPPORTED = {
    "Tiếng Bồ Đào Nha (BR)": "Portuguese (Brazil)",
    "Tiếng Bồ Đào Nha (PT)": "Portuguese (Portugal)",
    "Tiếng Việt": "Vietnamese",
    "Tiếng Anh": "English",
    "Tiếng Trung (Giản)": "Chinese (Simplified)",
    "Tiếng Nhật": "Japanese",
    "Tiếng Hàn": "Korean",
    "Tiếng Tây Ban Nha": "Spanish",
    "Tiếng Pháp": "French",
    "Tiếng Đức": "German"
}

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    #loading-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.85); display: flex; flex-direction: column;
        justify-content: center; align-items: center; z-index: 9999;
    }
    .spinner {
        border: 6px solid #f3f3f3; border-top: 6px solid #00ff00;
        border-radius: 50%; width: 60px; height: 60px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .loading-text { color: #00ff00; margin-top: 20px; font-size: 1.5rem; font-weight: bold; }
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
        text = item.translated_text.strip() if item.translated_text.strip() else item.text.strip()
        output.write(f"{item.index}\n{item.timecode}\n{text}\n\n")
    return output.getvalue()

def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_lang: str, target_lang: str) -> str:
    rows = [f"[{i}] {item.text.strip()}" for i, item in enumerate(batch, start=1)]
    joined_rows = "\n".join(rows)
    
    prompt = (
        f"You are a professional subtitle translator. Your task is to translate from {source_lang} to {target_lang}.\n"
        f"Style: {style_prompt}\n"
        f"Requirements:\n"
        f"1. Keep the format exactly as: [number] translation\n"
        f"2. No explanations or extra notes.\n"
        f"3. Translate every single line.\n"
        f"Input List:\n{joined_rows}"
    )
    return prompt

def parse_translated_response(batch: List[SubtitleItem], response_text: str) -> List[str]:
    mapping = {}
    for line in response_text.splitlines():
        match = re.search(r"^\[(\d+)\]\s*(.*)$", line.strip())
        if match:
            idx = int(match.group(1))
            mapping[idx] = match.group(2).strip()
    
    return [mapping.get(i+1, batch[i].text) for i in range(len(batch))]

def translate_batch(api_key, model_name, batch, style_prompt, source_lang, target_lang):
    client = genai.Client(api_key=api_key.strip())
    prompt = build_prompt(batch, style_prompt, source_lang, target_lang)
    for _ in range(MAX_RETRIES_PER_KEY):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            if response.text:
                return parse_translated_response(batch, response.text)
        except:
            time.sleep(RETRY_SLEEP_SECONDS)
    return [item.text for item in batch]

# =========================
# GIAO DIỆN STREAMLIT
# =========================
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('## 🎬 Đình Thái - SRT Translation Studio')

if "final_srt_data" not in st.session_state:
    st.session_state["final_srt_data"] = None

col1, col2 = st.columns([1.2, 1.5], gap="large")

with col1:
    st.subheader("📤 Đầu vào")
    files = st.file_uploader("Chọn file .srt", type=["srt"], accept_multiple_files=True)
    api_key_input = st.text_area("Gemini API Keys (Mỗi dòng 1 key)", height=100)
    
    st.subheader("⚙️ Cấu hình")
    src_lang = st.selectbox("Ngôn ngữ nguồn", ["Tự động", "Vietnamese", "English", "Chinese", "Japanese"])
    
    # MỤC CHỌN NGÔN NGỮ ĐÍCH THEO YÊU CẦU CỦA BẠN
    target_lang_label = st.selectbox("Dịch sang ngôn ngữ", list(LANGUAGES_SUPPORTED.keys()))
    target_lang_en = LANGUAGES_SUPPORTED[target_lang_label]
    
    style = st.text_input("Phong cách", "Dịch tự nhiên, ngắn gọn, phù hợp ngữ cảnh phim.")

with col2:
    st.subheader("🚀 Trạng thái")
    run_btn = st.button("▶ BẮT ĐẦU DỊCH", type="primary", use_container_width=True)
    
    if run_btn:
        if not files or not api_key_input:
            st.error("Vui lòng cung cấp đủ File và API Key!")
        else:
            # HIỆN LOADING GIỮA MÀN HÌNH
            loading = st.empty()
            loading.markdown(f"""
                <div id="loading-overlay">
                    <div class="spinner"></div>
                    <div class="loading-text">Đang dịch sang {target_lang_label}...</div>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                keys = [k.strip() for k in api_key_input.splitlines() if k.strip()]
                combined_content = ""
                
                for f in files:
                    raw_content = f.read().decode("utf-8-sig", errors="ignore")
                    items = read_srt_content(raw_content)
                    batches = [items[i:i + DEFAULT_BATCH_SIZE] for i in range(0, len(items), DEFAULT_BATCH_SIZE)]
                    
                    # Dùng ThreadPoolExecutor để tăng tốc độ dịch song song
                    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
                        futures = {executor.submit(translate_batch, keys[0], DEFAULT_MODEL, b, style, src_lang, target_lang_en): b for b in batches}
                        for future in as_completed(futures):
                            batch_items = futures[future]
                            translations = future.result()
                            for item, t_text in zip(batch_items, translations):
                                item.translated_text = t_text
                    
                    combined_content += write_srt_content(items)
                
                st.session_state["final_srt_data"] = combined_content
                loading.empty() # Tắt loading
                st.success(f"Đã hoàn thành dịch sang {target_lang_label}!")
                
            except Exception as e:
                loading.empty()
                st.error(f"Lỗi: {e}")

    # NÚT TẢI FILE SRT DUY NHẤT
    if st.session_state["final_srt_data"]:
        st.download_button(
            label="📥 TẢI FILE .SRT KẾT QUẢ",
            data=st.session_state["final_srt_data"],
            file_name="translated_subtitles.srt",
            mime="application/x-subrip",
            use_container_width=True
        )

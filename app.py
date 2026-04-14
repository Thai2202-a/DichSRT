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
# CẤU HÌNH HỆ THỐNG
# =========================
st.set_page_config(
    page_title="Đình Thái - SRT Multi-Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Các model hỗ trợ
MODEL_OPTIONS = [
    "gemini-2.5-flash", 
    "gemini-2.0-flash", 
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

DEFAULT_BATCH_SIZE = 80
MAX_RETRIES_PER_KEY = 2
RETRY_SLEEP_SECONDS = 0.35
MAX_PARALLEL_BATCHES = 4

LANGUAGES_SUPPORTED = {
    "Tiếng Bồ Đào Nha (BR)": "Portuguese (Brazil)",
    "Tiếng Bồ Đào Nha (PT)": "Portuguese (Portugal)",
    "Tiếng Việt": "Vietnamese",
    "Tiếng Anh": "English",
    "Tiếng Trung": "Chinese",
    "Tiếng Nhật": "Japanese",
    "Tiếng Hàn": "Korean",
    "Tiếng Tây Ban Nha": "Spanish",
    "Tiếng Pháp": "French"
}

# CSS Đầy đủ + Hiệu ứng Loading
CUSTOM_CSS = """
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    #loading-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.92); display: flex; flex-direction: column;
        justify-content: center; align-items: center; z-index: 9999;
    }
    .spinner {
        border: 8px solid #333; border-top: 8px solid #00ff00;
        border-radius: 50%; width: 70px; height: 70px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .loading-text { color: #00ff00; margin-top: 25px; font-size: 1.8rem; font-weight: bold; font-family: sans-serif; }
    .log-box { padding: 10px; border-radius: 5px; background: #1e2130; border-left: 5px solid #00ff00; margin-bottom: 5px; font-size: 0.9rem; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; font-weight: bold; background-color: #00ff00 !important; color: black !important; border: none; }
    .stButton>button:hover { background-color: #00cc00 !important; }
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
        text = item.translated_text.strip() if item.translated_text.strip() else item.text.strip()
        output.write(f"{item.index}\n{item.timecode}\n{text}\n\n")
    return output.getvalue()

# =========================
# LOGIC DỊCH THUẬT
# =========================
def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_lang: str, target_lang: str) -> str:
    rows = [f"[{i}] {item.text.strip()}" for i, item in enumerate(batch, start=1)]
    joined_rows = "\n".join(rows)
    return f"""Role: Expert Subtitle Translator.
Task: Translate from {source_lang} to {target_lang}.
Style: {style_prompt}
Strict Rules:
- Format: [number] translated_text
- No notes, No original text in output.
- Every ID must have a translation.
Subtitles:
{joined_rows}"""

def parse_translated_response(batch: List[SubtitleItem], response_text: str) -> List[str]:
    mapping = {}
    lines = response_text.splitlines()
    for line in lines:
        match = re.search(r"\[\s*(\d+)\s*\]\s*(.*)", line.strip())
        if match:
            idx = int(match.group(1))
            mapping[idx] = match.group(2).strip()
    return [mapping.get(i+1, batch[i].text).strip() for i in range(len(batch))]

def translate_batch(api_key, model_name, batch, style_prompt, source_lang, target_lang):
    try:
        client = genai.Client(api_key=api_key.strip())
        prompt = build_prompt(batch, style_prompt, source_lang, target_lang)
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
# UI STREAMLIT
# =========================
init_defaults = {"run_logs": [], "final_srt": None, "stats": {"files": 0, "total": 0, "done": 0, "speed": "0 d/s"}}
for k, v in init_defaults.items():
    if k not in st.session_state: st.session_state[k] = v

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<h1 style="text-align: center; color: #00ff00;">🎬 Đình Thái - SRT Portuguese Studio (V2.5)</h1>', unsafe_allow_html=True)

left, right = st.columns([1.1, 1.6], gap="large")

with left:
    st.markdown("### 📤 Upload Files")
    uploaded_files = st.file_uploader("Chọn 1 hoặc nhiều file .srt", type=["srt"], accept_multiple_files=True)
    
    st.markdown("### ⚙️ Cấu hình Model & Ngôn ngữ")
    # ĐÃ THÊM PHẦN CHỌN MODEL Ở ĐÂY
    selected_model = st.selectbox("CHỌN MODEL GEMINI", MODEL_OPTIONS, index=0)
    
    c1, c2 = st.columns(2)
    source_lang = c1.selectbox("Nguồn", ["Tự động", "Vietnamese", "English", "Chinese"])
    target_lang_label = c2.selectbox("Dịch sang", list(LANGUAGES_SUPPORTED.keys()))
    target_lang_en = LANGUAGES_SUPPORTED[target_lang_label]
    
    style_prompt = st.text_area("Style Prompt", value="Dịch tự nhiên, văn phong phim ảnh, ngắn gọn.", height=80)
    
    st.markdown("### 🔑 API Keys")
    keys_input = st.text_area("Dán danh sách Key (mỗi dòng 1 key)", placeholder="AIza...", height=120)

with right:
    st.markdown("### 🚀 Điều khiển & Thống kê")
    run_btn = st.button("▶ BẮT ĐẦU DỊCH NGAY", type="primary")
    
    st_c1, st_c2, st_c3 = st.columns(3)
    st_c1.metric("Số file", st.session_state["stats"]["files"])
    st_c2.metric("Hoàn thành", f"{st.session_state['stats']['done']} dòng")
    st_c3.metric("Tốc độ", st.session_state["stats"]["speed"])
    
    st.markdown("---")
    log_placeholder = st.empty()

    if run_btn:
        if not uploaded_files or not keys_input:
            st.error("Bạn chưa upload file hoặc nhập API Key!")
        else:
            loading = st.empty()
            loading.markdown(f"""
                <div id="loading-overlay">
                    <div class="spinner"></div>
                    <div class="loading-text">Đang sử dụng {selected_model} để dịch sang {target_lang_label}...</div>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                api_keys = [k.strip() for k in keys_input.splitlines() if k.strip()]
                all_srt_output = ""
                total_lines_done = 0
                start_time = time.time()
                
                for f in uploaded_files:
                    content = f.read().decode("utf-8-sig", errors="ignore")
                    items = read_srt_content(content)
                    batches = [items[i:i + DEFAULT_BATCH_SIZE] for i in range(0, len(items), DEFAULT_BATCH_SIZE)]
                    
                    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
                        # Dùng key đầu tiên (Thái có thể sửa logic loop key ở đây)
                        futures = {executor.submit(translate_batch, api_keys[0], selected_model, b, style_prompt, source_lang, target_lang_en): b for b in batches}
                        for future in as_completed(futures):
                            batch_items = futures[future]
                            res = future.result()
                            for item, t_text in zip(batch_items, res): item.translated_text = t_text
                            
                            total_lines_done += len(batch_items)
                            elapsed = time.time() - start_time
                            st.session_state["stats"]["done"] = total_lines_done
                            st.session_state["stats"]["speed"] = f"{total_lines_done/elapsed:.1f} d/s" if elapsed > 0 else "0 d/s"
                    
                    all_srt_output += write_srt_content(items)
                    st.session_state["run_logs"].append(f"✅ Xong: {f.name}")
                    
                st.session_state["final_srt"] = all_srt_output
                st.session_state["stats"]["files"] = len(uploaded_files)
                loading.empty()
                st.success("Tất cả file đã được dịch thành công!")
                
            except Exception as e:
                loading.empty()
                st.error(f"Lỗi: {e}")

    if st.session_state["final_srt"]:
        st.download_button(
            label="📥 TẢI FILE .SRT KẾT QUẢ",
            data=st.session_state["final_srt"],
            file_name="translated_studio_result.srt",
            mime="application/x-subrip",
            use_container_width=True
        )

    if st.session_state["run_logs"]:
        with st.expander("Nhật ký xử lý", expanded=True):
            for log in st.session_state["run_logs"][-10:]:
                st.markdown(f'<div class="log-box">{log}</div>', unsafe_allow_html=True)

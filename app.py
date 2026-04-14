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

DEFAULT_MODEL = "gemini-1.5-flash"
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

# CSS đầy đủ bao gồm hiệu ứng Loading Overlay
CUSTOM_CSS = """
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    #loading-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.9); display: flex; flex-direction: column;
        justify-content: center; align-items: center; z-index: 9999;
    }
    .spinner {
        border: 8px solid #f3f3f3; border-top: 8px solid #00ff00;
        border-radius: 50%; width: 70px; height: 70px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .loading-text { color: #00ff00; margin-top: 25px; font-size: 1.8rem; font-weight: bold; }
    
    /* Style cho các hộp log live */
    .log-box { padding: 10px; border-radius: 5px; background: #1e2130; border-left: 5px solid #00ff00; margin-bottom: 5px; font-size: 0.9rem; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
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
# LOGIC DỊCH THUẬT ÉP BUỘC (FORCE TRANSLATION)
# =========================
def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_lang: str, target_lang: str) -> str:
    rows = [f"[{i}] {item.text.strip()}" for i, item in enumerate(batch, start=1)]
    joined_rows = "\n".join(rows)
    return f"""Role: Professional Subtitle Translator.
Task: Translate subtitles from {source_lang} to {target_lang}.
Style: {style_prompt}
Strict Rules:
1. Output format: [number] translated_text
2. Do not include any notes, explanations or original text.
3. Every input ID must have a translated output line.
4. Translate everything, even if it is short.

Subtitles:
{joined_rows}"""

def parse_translated_response(batch: List[SubtitleItem], response_text: str) -> List[str]:
    mapping = {}
    lines = response_text.splitlines()
    for line in lines:
        # Regex bắt được cả [1], [ 1], [1 ]
        match = re.search(r"\[\s*(\d+)\s*\]\s*(.*)", line.strip())
        if match:
            idx = int(match.group(1))
            mapping[idx] = match.group(2).strip()
    
    results = []
    for i in range(1, len(batch) + 1):
        res = mapping.get(i, "").strip()
        results.append(res if res else batch[i-1].text)
    return results

def try_translate_batch_with_key(api_key, model_name, batch, style_prompt, source_lang, target_lang):
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
# SESSION STATE
# =========================
def init_state():
    if "run_logs" not in st.session_state: st.session_state["run_logs"] = []
    if "final_srt" not in st.session_state: st.session_state["final_srt"] = None
    if "stats" not in st.session_state: 
        st.session_state["stats"] = {"files": 0, "total": 0, "done": 0, "speed": "0 d/s"}

init_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<h1 style="text-align: center;">🎬 Đình Thái - SRT Multi-Studio</h1>', unsafe_allow_html=True)

# =========================
# GIAO DIỆN 2 CỘT FULL NHƯ BẢN GỐC
# =========================
left, right = st.columns([1.1, 1.6], gap="large")

with left:
    st.markdown("### 📤 Upload & Cài đặt")
    uploaded_files = st.file_uploader("Chọn file .srt (có thể chọn nhiều)", type=["srt"], accept_multiple_files=True)
    
    st.markdown("### ⚙️ Cấu hình dịch")
    c1, c2 = st.columns(2)
    source_lang = c1.selectbox("Ngôn ngữ gốc", ["Tự động", "Vietnamese", "English", "Chinese", "Japanese"])
    
    # Mục chọn ngôn ngữ muốn dịch sang
    target_lang_label = c2.selectbox("Dịch sang", list(LANGUAGES_SUPPORTED.keys()))
    target_lang_en = LANGUAGES_SUPPORTED[target_lang_label]
    
    style_prompt = st.text_area("Style Prompt", value="Dịch tự nhiên, mượt, xưng hô phù hợp phim.", height=80)
    
    st.markdown("### 🔑 Tài khoản API")
    keys_input = st.text_area("Gemini API Keys (Mỗi dòng 1 key)", placeholder="AIza...", height=120)
    slots = st.text_area("Số luồng (Slots)", value="3", height=50)
    
    if st.button("♻️ Làm mới hệ thống"):
        st.session_state["final_srt"] = None
        st.session_state["run_logs"] = []
        st.rerun()

with right:
    st.markdown("### 🚀 Trạng thái & Điều khiển")
    run_btn = st.button("▶ BẮT ĐẦU DỊCH NGAY", type="primary")
    
    # Khu vực thống kê nhanh
    st_c1, st_c2, st_c3 = st.columns(3)
    st_c1.metric("Số file", st.session_state["stats"]["files"])
    st_c2.metric("Đã xong", f"{st.session_state['stats']['done']} dòng")
    st_c3.metric("Tốc độ", st.session_state["stats"]["speed"])
    
    st.markdown("---")
    progress_placeholder = st.empty()
    log_placeholder = st.empty()

    if run_btn:
        if not uploaded_files or not keys_input:
            st.error("Bạn chưa upload file hoặc nhập API Key!")
        else:
            # HIỆN LOADING OVERLAY
            loading = st.empty()
            loading.markdown(f"""
                <div id="loading-overlay">
                    <div class="spinner"></div>
                    <div class="loading-text">Đang dịch sang {target_lang_label}... Vui lòng không đóng tab</div>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                api_keys = [k.strip() for k in keys_input.splitlines() if k.strip()]
                all_output_srt = ""
                total_lines_done = 0
                start_time = time.time()
                
                for f_idx, f in enumerate(uploaded_files):
                    content = f.read().decode("utf-8-sig", errors="ignore")
                    items = read_srt_content(content)
                    
                    batches = [items[i:i + DEFAULT_BATCH_SIZE] for i in range(0, len(items), DEFAULT_BATCH_SIZE)]
                    
                    # Dùng ThreadPool để dịch song song như bản gốc
                    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
                        future_to_batch = {
                            executor.submit(try_translate_batch_with_key, api_keys[0], DEFAULT_MODEL, b, style_prompt, source_lang, target_lang_en): b 
                            for b in batches
                        }
                        
                        for future in as_completed(future_to_batch):
                            batch_items = future_to_batch[future]
                            translations = future.result()
                            for item, t_text in zip(batch_items, translations):
                                item.translated_text = t_text
                            
                            total_lines_done += len(batch_items)
                            elapsed = time.time() - start_time
                            speed = total_lines_done / elapsed if elapsed > 0 else 0
                            
                            st.session_state["stats"]["done"] = total_lines_done
                            st.session_state["stats"]["speed"] = f"{speed:.1f} d/s"
                    
                    all_output_srt += write_srt_content(items)
                    st.session_state["run_logs"].append(f"✅ Đã xong file: {f.name}")
                    
                st.session_state["final_srt"] = all_output_srt
                st.session_state["stats"]["files"] = len(uploaded_files)
                loading.empty() # Tắt loading
                st.success(f"Đã dịch xong {len(uploaded_files)} file!")
                
            except Exception as e:
                loading.empty()
                st.error(f"Lỗi nghiêm trọng: {e}")

    # NÚT TẢI FILE SRT DUY NHẤT
    if st.session_state["final_srt"]:
        st.download_button(
            label="📥 TẢI FILE .SRT KẾT QUẢ",
            data=st.session_state["final_srt"],
            file_name="translated_studio_result.srt",
            mime="application/x-subrip",
            use_container_width=True
        )

    # Hiển thị logs live
    if st.session_state["run_logs"]:
        with st.expander("Xem nhật ký chi tiết", expanded=True):
            for log in st.session_state["run_logs"][-10:]:
                st.markdown(f'<div class="log-box">{log}</div>', unsafe_allow_html=True)

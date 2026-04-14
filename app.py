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
    page_title="Đình Thái - SRT Portuguese Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_BATCH_SIZE = 80
MAX_RETRIES_PER_KEY = 2
RETRY_SLEEP_SECONDS = 0.35
MAX_PARALLEL_BATCHES = 4

BASE_SYSTEM_PROMPT = """
Você é um especialista em tradução de legendas de filmes para o idioma Português.
YÊU CẦU CHUNG:
Dịch tự nhiên, mượt, đúng ngữ cảnh hội thoại sang tiếng Bồ Đào Nha.
Giữ văn phong giống phụ đề phim, dễ đọc, gọn.
Không giải thích, không ghi chú.
Không bỏ dòng nào.
Không đánh số lại.
Mỗi mục phụ đề đầu vào phải trả về đúng 1 dòng đầu ra tương ứng.
Nếu gặp tên riêng thì xử lý hợp lý theo ngữ cảnh.
Nếu dòng đã là tiếng Bồ Đào Nha chuẩn thì giữ nguyên hoặc chỉnh nhẹ cho tự nhiên hơn.
Không thêm ký tự thừa. """.strip()

SOURCE_LANGUAGE_OPTIONS = [
    "Tự động",
    "Tiếng Việt",
    "Tiếng Trung",
    "Tiếng Anh",
    "Tiếng Nhật",
    "Tiếng Hàn",
    "Tiếng Thái",
    "Tiếng Pháp",
    "Tiếng Đức",
    "Tiếng Nga",
    "Tiếng Tây Ban Nha",
    "Tiếng Ả Rập",
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
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .status-box { padding: 20px; border-radius: 10px; background: #1e2130; border: 1px solid #3e445e; margin-bottom: 20px; }
    .key-green { color: #00ff00; }
    .key-red { color: #ff4b4b; }
    .key-yellow { color: #ffff00; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    .title-text { font-size: 2.2rem; font-weight: 800; color: #ffffff; margin-bottom: 0.5rem; }
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
    if not sample: return "unknown"
    if re.search(r"[À-ỹà-ỹĂăÂâĐđÊêÔôƠơƯư]", sample): return "vi"
    if re.search(r"[\u4e00-\u9fff]", sample): return "zh"
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", sample): return "ja"
    latin_letters = re.findall(r"[A-Za-z]", sample)
    if latin_letters:
        lower = sample.lower()
        if any(ch in lower for ch in ["ã", "õ", "ç", "á", "é", "í", "ó", "ú"]): return "pt"
        return "en_or_latin"
    return "unknown"

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
    if not counts: return "unknown"
    return max(counts, key=counts.get)

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
        p_text = partial_items[i].text.strip()
        if p_text:
            source_items[i].translated_text = p_text
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
    if len(api_key) <= 10: return api_key
    return f"{api_key[:6]}...{api_key[-4:]}"

def test_single_api_key(api_key: str, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    try:
        client = create_client(api_key)
        response = client.models.generate_content(
            model=model_name, contents="Respond with OK",
            config=types.GenerateContentConfig(temperature=0),
        )
        if response.text:
            return {"key": api_key, "ok": True, "status": "KEY OK", "color": "green", "detail": "Dùng được."}
        return {"key": api_key, "ok": False, "status": "LỖI", "color": "red", "detail": "Rỗng."}
    except Exception as e:
        return {"key": api_key, "ok": False, "status": "LỖI", "color": "red", "detail": str(e)}

def test_all_api_keys(api_keys: List[str], model_name: str) -> List[Dict[str, Any]]:
    return [test_single_api_key(key, model_name) for key in api_keys]

def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_language: str) -> str:
    rows = [f"[{i}] {item.text.replace(chr(13), '').strip()}" for i, item in enumerate(batch, start=1)]
    joined_rows = "\n".join(rows)
    extra = f"\nREQUISITO DE ESTILO:\n{style_prompt.strip()}\n" if style_prompt.strip() else ""
    lang_instr = f"- Idioma de origem: {source_language}.\n" if source_language != "Tự động" else "- Detectar idioma automaticamente.\n"
    
    prompt = (
        f"{BASE_SYSTEM_PROMPT}\n{extra}\n"
        "TAREFA:\n"
        f"{lang_instr}"
        "- Traduzir TUDO para o PORTUGUÊS.\n"
        "- Se a linha já estiver em Português, mantenha-a.\n\n"
        "FORMATO DE RETORNO:\n"
        "- [número] tradução\n"
        "- Sem explicações.\n\n"
        "DANH SÁCH:\n"
        f"{joined_rows}"
    )
    return prompt.strip()

def parse_translated_response(batch: List[SubtitleItem], response_text: str) -> List[str]:
    mapping: Dict[int, str] = {}
    for line in response_text.splitlines():
        match = re.match(r"^\[(\d+)\]\s*(.*)$", line.strip())
        if match: mapping[int(match.group(1))] = match.group(2).strip()
    results = []
    for i in range(1, len(batch) + 1):
        txt = mapping.get(i, "").strip()
        results.append(txt if txt else batch[i - 1].text)
    return results

def try_translate_batch_with_key(api_key, model_name, batch, style_prompt, source_language):
    client = create_client(api_key)
    prompt = build_prompt(batch, style_prompt, source_language)
    for _ in range(MAX_RETRIES_PER_KEY):
        try:
            resp = client.models.generate_content(
                model=model_name, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = (resp.text or "").strip()
            if not text: raise RuntimeError("Model rỗng")
            return parse_translated_response(batch, text)
        except: time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError("Failed")

def translate_batch_with_failover(batch_id, batch, worker_slots, model_name, style_prompt, source_language):
    for api_key in worker_slots:
        try:
            trans = try_translate_batch_with_key(api_key, model_name, batch, style_prompt, source_language)
            return batch_id, True, trans, ""
        except: continue
    return batch_id, False, [item.text for item in batch], "Error"

def collect_api_keys_and_slots(keys_raw: str, batches_raw: str) -> Tuple[List[str], List[str]]:
    api_keys = [k.strip() for k in keys_raw.splitlines() if k.strip()]
    batch_vals = [b.strip() for b in batches_raw.splitlines()]
    slots = []
    for i, k in enumerate(api_keys):
        c = 1
        try:
            if i < len(batch_vals): c = int(batch_vals[i])
        except: c = 1
        for _ in range(max(1, c)): slots.append(k)
    return api_keys, slots

# =========================
# XỬ LÝ FILE - TĂNG TỐC
# =========================
def process_one_file(file_name, source_bytes, partial_bytes, worker_slots, model_name, style_prompt, batch_size, source_language, progress_callback=None):
    try:
        source_text = source_bytes.decode("utf-8-sig")
    except:
        source_text = source_bytes.decode("utf-8", errors="ignore")
    
    source_items = read_srt_content(source_text)
    if not source_items: return {"file_name": file_name, "success": False, "error": "Rỗng", "output_bytes": b"", "stats": {"total":0,"skip":0,"need":0,"done":0,"failed_batches":0,"speed":0.0}, "logs": [], "detected_lang": ""}

    det_lang = detect_dominant_language(source_items)
    source_items, skipped = prepare_items_from_source(source_items)
    
    resumed = 0
    if partial_bytes:
        p_text = partial_bytes.decode("utf-8", errors="ignore")
        source_items, resumed = merge_partial_translation(source_items, read_srt_content(p_text))

    batches = build_batches(source_items, batch_size)
    total_need = sum(len(b) for b in batches)
    logs = [f"🌐 {file_name} | Phát hiện: {LANGUAGE_LABELS.get(det_lang, det_lang)}"]
    
    done_lines = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_BATCHES, len(worker_slots))) as executor:
        future_map = {executor.submit(translate_batch_with_failover, i, b, worker_slots, model_name, style_prompt, source_language): (i, b) for i, b in enumerate(batches)}
        for future in as_completed(future_map):
            bid, b = future_map[future]
            _, ok, trans, _ = future.result()
            for item, t in zip(b, trans): item.translated_text = t
            done_lines += len(b)
            if progress_callback: progress_callback("batch_done", file_name, bid, len(batches), [t[:80] for t in trans])

    return {
        "file_name": file_name, "success": True, "error": "",
        "output_bytes": write_srt_content(source_items).encode("utf-8"),
        "stats": {"total": len(source_items), "skip": skipped+resumed, "need": total_need, "done": done_lines, "failed_batches": 0, "speed": done_lines/(time.time()-start_time)},
        "logs": logs, "detected_lang": LANGUAGE_LABELS.get(det_lang, det_lang)
    }

def build_partial_map(partial_upload) -> Dict[str, bytes]:
    res = {}
    if not partial_upload: return res
    if partial_upload.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(partial_upload.read())) as zf:
            for n in zf.namelist():
                if n.lower().endswith(".srt"): res[n.split("/")[-1]] = zf.read(n)
    else: res[partial_upload.name] = partial_upload.read()
    return res

# =========================
# SESSION STATE & UI
# =========================
def init_state():
    for k, v in {"zip_bytes": b"", "result_ready": False, "run_logs": [], "api_test_results": [], "stats": {"files": 0, "total": 0, "skip": 0, "need": 0, "done": 0, "failed_batches": 0}, "speed_text": "0 dòng/s"}.items():
        if k not in st.session_state: st.session_state[k] = v

init_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<div class="title-text">🎬 Đình Thái - SRT Portuguese Studio</div>', unsafe_allow_html=True)

left, right = st.columns([1.2, 1.5], gap="large")

with left:
    st.markdown('### 📤 Upload Files')
    uploaded_files = st.file_uploader("Chọn file .srt", type=["srt"], accept_multiple_files=True)
    partial_zip = st.file_uploader("File dịch dở (ZIP/SRT)", type=["zip", "srt"])
    
    st.markdown('### ⚙️ Cấu Hình')
    c1, c2, c3 = st.columns(3)
    model_name = c1.selectbox("MODEL", ["gemini-2.0-flash", "gemini-1.5-flash"])
    batch_size = c2.number_input("BATCH", 1, 200, DEFAULT_BATCH_SIZE)
    source_language = c3.selectbox("NGUỒN", SOURCE_LANGUAGE_OPTIONS)
    
    style_prompt = st.text_area("Prompt Style", "Dịch tự nhiên sang tiếng Bồ Đào Nha (PT-BR). Ưu tiên câu ngắn, dễ đọc.", height=100)
    
    st.markdown('### 🔑 API Keys')
    keys_text = st.text_area("Keys", placeholder="AIza...", height=120)
    batch_text = st.text_area("Slots", value="3", height=60)
    
    if st.button("🧪 Test Keys"):
        raw_keys = [k.strip() for k in keys_text.splitlines() if k.strip()]
        st.session_state["api_test_results"] = test_all_api_keys(raw_keys, model_name)

with right:
    st.markdown('### 🚀 Điều Khiển')
    run_btn = st.button("▶ Bắt Đầu Dịch", type="primary")
    
    stats = st.session_state["stats"]
    g1, g2, g3 = st.columns(3)
    g1.metric("Số file", stats["files"])
    g2.metric("Tổng dòng", stats["total"])
    g3.metric("Tốc độ", st.session_state["speed_text"])
    
    st.markdown('### 🖥 Logs')
    log_placeholder = st.empty()
    if st.session_state["run_logs"]: log_placeholder.code("\n".join(st.session_state["run_logs"][-10:]))

    if run_btn and uploaded_files and keys_text:
        api_keys, slots = collect_api_keys_and_slots(keys_text, batch_text)
        partial_map = build_partial_map(partial_zip)
        
        start_t = time.time()
        results = []
        for i, f in enumerate(uploaded_files):
            res = process_one_file(f.name, f.read(), partial_map.get(f.name), slots, model_name, style_prompt, batch_size, source_language)
            results.append(res)
            
            # Cập nhật UI
            st.session_state["stats"]["files"] = len(uploaded_files)
            st.session_state["stats"]["total"] += res["stats"]["total"]
            st.session_state["stats"]["done"] += res["stats"]["done"]
            st.session_state["speed_text"] = f"{st.session_state['stats']['done']/(time.time()-start_t):.1f} d/s"
            st.session_state["run_logs"].extend(res["logs"])
            log_placeholder.code("\n".join(st.session_state["run_logs"][-10:]))

        # Tạo ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for r in results:
                if r["output_bytes"]: zf.writestr(r["file_name"], r["output_bytes"])
        st.session_state["zip_bytes"] = zip_buffer.getvalue()
        st.session_state["result_ready"] = True

    if st.session_state["result_ready"]:
        st.download_button("📥 Tải ZIP kết quả", st.session_state["zip_bytes"], "portuguese_translated.zip", "application/zip", use_container_width=True)

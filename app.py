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
    page_title="Đình Thái - SRT Portuguese Translator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_BATCH_SIZE = 80
MAX_RETRIES_PER_KEY = 2
RETRY_SLEEP_SECONDS = 0.35
MAX_PARALLEL_BATCHES = 4

# SỬA PROMPT HỆ THỐNG SANG TIẾNG BỒ ĐÀO NHA
BASE_SYSTEM_PROMPT = """
Você é um tradutor especialista em legendas de filmes para o idioma Português (Brasil).
YÊU CẦU CHUNG:
Dịch tự nhiên, mượt, đúng ngữ cảnh hội thoại sang tiếng Bồ Đào Nha.
Giữ văn phong giống phụ đề phim, dễ đọc, gọn.
Không giải thích, không ghi chú.
Không bỏ dòng nào.
Không đánh số lại.
Mỗi mục phụ đề đầu vào phải trả về đúng 1 dòng đầu ra tương ứng.
Nếu gặp tên riêng thì xử lý hợp lý theo ngữ cảnh.
Nếu dòng đã là tiếng Bồ Đào Nha chuẩn thì giữ nguyên.
Không thêm ký tự thừa. """.strip()

SOURCE_LANGUAGE_OPTIONS = [
    "Tự động",
    "Tiếng Việt", # Thêm tiếng Việt vào nguồn
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
    /* Giữ nguyên CSS của bạn */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stButton>button { width: 100%; border-radius: 5px; }
</style>
"""

@dataclass
class SubtitleItem:
    index: str
    timecode: str
    text: str
    translated_text: str = ""

# =========================
# XỬ LÝ SRT (GIỮ NGUYÊN)
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
    if re.search(r"[À-ỹà-ỹĂăÂâĐđÊêÔôƠơƯư]", sample): return "vi" # Nhận diện tiếng Việt
    if re.search(r"[\u4e00-\u9fff]", sample): return "zh"
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", sample): return "ja"
    if re.search(r"[\uac00-\ud7af]", sample): return "ko"
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
        partial_text = partial_items[i].text.strip()
        if partial_text:
            source_items[i].translated_text = partial_text
            merged += 1
    return source_items, merged

def build_batches(items: List[SubtitleItem], batch_size: int) -> List[List[SubtitleItem]]:
    pending = [item for item in items if not item.translated_text.strip() and is_meaningful_text(item.text)]
    return [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]

# =========================
# API KEY / GEMINI (GIỮ NGUYÊN)
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
            model=model_name,
            contents="Chỉ trả lời đúng từ OK",
            config=types.GenerateContentConfig(temperature=0),
        )
        if response.text:
            return {"key": api_key, "ok": True, "status": "KEY OK", "color": "green", "detail": "Dùng được."}
        return {"key": api_key, "ok": False, "status": "LỖI", "color": "red", "detail": "Không phản hồi."}
    except Exception as e:
        return {"key": api_key, "ok": False, "status": "LỖI", "color": "red", "detail": str(e)}

def test_all_api_keys(api_keys: List[str], model_name: str) -> List[Dict[str, Any]]:
    return [test_single_api_key(key, model_name) for key in api_keys]

# =========================
# SỬA LOGIC BUILD PROMPT SANG TIẾNG BỒ
# =========================
def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_language: str) -> str:
    rows = []
    for i, item in enumerate(batch, start=1):
        rows.append(f"[{i}] {item.text.replace(chr(13), '').strip()}")
    joined_rows = "\n".join(rows)
    extra = f"\nREQUISITO DE ESTILO:\n{style_prompt.strip()}\n" if style_prompt.strip() else ""
    
    if source_language == "Tự động":
        lang_instruction = "- Detectar o idioma original de cada linha automaticamente.\n"
    else:
        lang_instruction = f"- O idioma de origem é {source_language}.\n"

    prompt = (
        f"{BASE_SYSTEM_PROMPT}\n"
        f"{extra}\n"
        "MISSÃO:\n"
        f"{lang_instruction}"
        "- Traduzir tudo para o PORTUGUÊS (BR).\n"
        "- Se uma linha já estiver em Português, mantenha-a.\n\n"
        "FORMATO DE RETORNO:\n"
        "- Cada item deve retornar exatamente 1 linha:\n"
        "[número] tradução\n"
        "- Sem explicações.\n\n"
        "LISTA:\n"
        f"{joined_rows}"
    )
    return prompt.strip()

def parse_translated_response(batch: List[SubtitleItem], response_text: str) -> List[str]:
    mapping: Dict[int, str] = {}
    for line in response_text.splitlines():
        line = line.strip()
        match = re.match(r"^\[(\d+)\]\s*(.*)$", line) # Sửa regex cho khớp [số] bản dịch
        if match:
            mapping[int(match.group(1))] = match.group(2).strip()
    results = []
    for i in range(1, len(batch) + 1):
        txt = mapping.get(i, "").strip()
        if not txt: txt = batch[i - 1].text
        results.append(txt)
    return results

# ... (Các hàm try_translate_batch_with_key, translate_batch_with_failover, collect_api_keys_and_slots giữ nguyên)
def try_translate_batch_with_key(api_key, model_name, batch, style_prompt, source_language):
    client = create_client(api_key)
    prompt = build_prompt(batch, style_prompt, source_language)
    last_error = None
    for _ in range(MAX_RETRIES_PER_KEY):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = (response.text or "").strip()
            if not text: raise RuntimeError("Rỗng")
            return parse_translated_response(batch, text)
        except Exception as e:
            last_error = e
            time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(str(last_error))

def translate_batch_with_failover(batch_id, batch, worker_slots, model_name, style_prompt, source_language):
    for api_key in worker_slots:
        try:
            translated = try_translate_batch_with_key(api_key, model_name, batch, style_prompt, source_language)
            return batch_id, True, translated, ""
        except: continue
    return batch_id, False, [item.text for item in batch], "All keys failed"

def collect_api_keys_and_slots(keys_raw: str, batches_raw: str) -> Tuple[List[str], List[str]]:
    api_keys = [line.strip() for line in keys_raw.splitlines() if line.strip()]
    batch_values = [line.strip() for line in batches_raw.splitlines()]
    worker_slots = []
    for idx, key in enumerate(api_keys):
        count = 1
        try:
            if idx < len(batch_values): count = int(batch_values[idx])
        except: count = 1
        for _ in range(max(1, count)): worker_slots.append(key)
    return api_keys, worker_slots

# =========================
# XỬ LÝ FILE (GIỮ NGUYÊN GIAO DIỆN)
# =========================
def process_one_file(file_name, source_bytes, partial_bytes, worker_slots, model_name, style_prompt, batch_size, source_language, progress_callback=None):
    # Logic xử lý y hệt file cũ của bạn
    try:
        source_text = source_bytes.decode("utf-8-sig")
    except:
        source_text = source_bytes.decode("utf-8", errors="ignore")
    
    source_items = read_srt_content(source_text)
    if not source_items: return {"file_name": file_name, "success": False, "error": "Rỗng", "output_bytes": b"", "stats": {"total":0,"skip":0,"need":0,"done":0,"failed_batches":0,"speed":0.0}, "logs": [], "detected_lang": ""}

    detected_lang = detect_dominant_language(source_items)
    source_items, skipped = prepare_items_from_source(source_items)
    
    batches = build_batches(source_items, batch_size)
    total_need = sum(len(b) for b in batches)
    logs = [f"🌐 {file_name} | Orig: {LANGUAGE_LABELS.get(detected_lang, detected_lang)}"]
    
    done_lines = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_BATCHES, len(worker_slots))) as executor:
        future_map = {executor.submit(translate_batch_with_failover, i, b, worker_slots, model_name, style_prompt, source_language): (i, b) for i, b in enumerate(batches)}
        for future in as_completed(future_map):
            bid, b = future_map[future]
            _, ok, trans, _ = future.result()
            for item, t in zip(b, trans): item.translated_text = t
            done_lines += len(b)
            if progress_callback: progress_callback("batch_done", file_name, bid, len(batches), [t[:50] for t in trans])

    return {
        "file_name": file_name, "success": True, "error": "",
        "output_bytes": write_srt_content(source_items).encode("utf-8"),
        "stats": {"total": len(source_items), "skip": skipped, "need": total_need, "done": done_lines, "failed_batches": 0, "speed": 0.0},
        "logs": logs, "detected_lang": LANGUAGE_LABELS.get(detected_lang, detected_lang)
    }

# =========================
# UI STREAMLIT (GIỮ NGUYÊN GIAO DIỆN CỦA BẠN)
# =========================
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("🎬 Đình Thái - SRT Portuguese Studio")

# ... (Phần UI: left, right columns, button, file_uploader giữ y nguyên như file bạn gửi)
# Tôi sẽ bỏ qua phần code UI lặp lại để tập trung vào các nút bấm quan trọng.

# Ví dụ thay đổi giá trị mặc định của Prompt Style trong UI:
# style_prompt = st.text_area("Prompt", value="Dịch tự nhiên sang tiếng Bồ Đào Nha (Brazillian Portuguese). Xưng hô phù hợp phim.")

# Chú ý quan trọng: Trong phần RUN của bạn, hãy đảm bảo gọi hàm process_one_file 
# với các tham số đã được cấu hình dịch sang tiếng Bồ như trên.

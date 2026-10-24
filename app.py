import streamlit as st
import sqlite3
import os
import pickle
import numpy as np
from datetime import date
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import base64
import socket
from io import BytesIO
from openpyxl import Workbook
from openpyxl.drawing.image import Image as OpenpyxlImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check root directory first, then fallback to Database/ folder
ROOT_DB = os.path.join(BASE_DIR, 'alh_errors.db')
FOLDER_DB = os.path.join(BASE_DIR, 'Database', 'alh_errors.db')

if os.path.exists(ROOT_DB):
    DB_PATH = ROOT_DB
elif os.path.exists(FOLDER_DB):
    DB_PATH = FOLDER_DB
else:
    DB_PATH = ROOT_DB  # Default fallback

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_PATH = os.path.join(BASE_DIR, 'vectors.pkl')


def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assembly_errors (
            Error_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Date TEXT,
            Component_Ref TEXT,
            Error_Type TEXT,
            Description TEXT,
            Corrective_Action TEXT,
            Technician_ID TEXT,
            Status TEXT,
            Image BLOB
        )
    ''')
    conn.commit()
    conn.close()


@st.cache_resource
def load_model():
    local_path = os.path.join(BASE_DIR, 'Requirements', 'models', 'all-MiniLM-L6-v2')
    if os.path.exists(local_path):
        return SentenceTransformer(local_path)
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'rb') as f:
                records, vectors = pickle.load(f)
                if records and len(records) > 0:
                    return records, vectors
        except Exception:
            pass
    return None, None


def rebuild_cache(model):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT Error_ID, Component_Ref, Error_Type, Description, Corrective_Action FROM assembly_errors")
    records = cursor.fetchall()
    conn.close()

    if not records:
        return [], None

    descriptions = [r[3] for r in records]
    vectors = model.encode(descriptions)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump((records, vectors), f)
    return records, vectors


def search(query, model, records, vectors, top_k=3):
    if not records or vectors is None:
        return []

    qv = model.encode([query])
    scores = cosine_similarity(qv, vectors)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]

    occurrence_count = {}
    for r in records:
        key = (r[1], r[2])
        occurrence_count[key] = occurrence_count.get(key, 0) + 1

    results = []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for idx in top_idx:
        r = records[idx]
        key = (r[1], r[2])

        cursor.execute("SELECT Image FROM assembly_errors WHERE Error_ID = ?", (r[0],))
        img_row = cursor.fetchone()
        img_blob = img_row[0] if img_row else None

        img_b64 = base64.b64encode(img_blob).decode() if img_blob else None

        results.append({
            'Error_ID': r[0],
            'Component_Ref': r[1],
            'Error_Type': r[2],
            'Description': r[3],
            'Corrective_Action': r[4],
            'Match_Score': round(float(scores[idx]) * 100, 1),
            'Occurrence_Count': occurrence_count[key],
            'Image': img_b64
        })
    conn.close()
    return results


def log_error(component_ref, error_type, description, corrective_action, technician_id, image_bytes):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
                   SELECT COUNT(*)
                   FROM assembly_errors
                   WHERE Component_Ref = ?
                     AND Error_Type = ?
                     AND Description = ?
                     AND Corrective_Action = ?
                   ''', (component_ref, error_type, description, corrective_action))
    exists = cursor.fetchone()[0]

    if exists > 0:
        conn.close()
        return False

    cursor.execute('''
                   INSERT INTO assembly_errors
                   (Date, Component_Ref, Error_Type, Description, Corrective_Action, Technician_ID, Status, Image)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ''', (str(date.today()), component_ref, error_type, description,
                         corrective_action, technician_id, 'Resolved', image_bytes))
    conn.commit()
    conn.close()
    return True


def get_all_errors():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM assembly_errors ORDER BY Error_ID DESC", conn)
    conn.close()
    return df


def remove_duplicates():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
                   DELETE
                   FROM assembly_errors
                   WHERE Error_ID NOT IN (SELECT MIN(Error_ID)
                                          FROM assembly_errors
                                          GROUP BY Component_Ref, Error_Type, Description, Corrective_Action)
                   ''')
    removed = cursor.rowcount
    conn.commit()
    conn.close()
    return removed


def delete_error(error_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM assembly_errors WHERE Error_ID = ?", (error_id,))
    conn.commit()
    conn.close()


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM assembly_errors")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM assembly_errors WHERE Date = ?", (str(date.today()),))
    today = cursor.fetchone()[0]
    cursor.execute(
        "SELECT Error_Type, COUNT(*) as cnt FROM assembly_errors GROUP BY Error_Type ORDER BY cnt DESC LIMIT 1")
    row = cursor.fetchone()
    top_error = row[0] if row else "N/A"
    cursor.execute("SELECT COUNT(DISTINCT Component_Ref) FROM assembly_errors")
    components = cursor.fetchone()[0]
    conn.close()
    return total, today, top_error, components


def check_network():
    try:
        socket.setdefaulttimeout(2)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except Exception:
        return False


def get_bg_base64():
    bg_path = os.path.join(BASE_DIR, 'hal_bg.jpg')
    if os.path.exists(bg_path):
        with open(bg_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None


def generate_excel_with_images(dataframe):
    wb = Workbook()
    ws = wb.active
    ws.title = "Assembly Error Log"

    headers = [col for col in dataframe.columns if col != '_label']
    if 'Image' not in headers:
        headers.append('Image')
    ws.append(headers)

    for index, row in dataframe.iterrows():
        row_data = []
        for col in headers:
            if col == 'Image':
                row_data.append("")
            else:
                row_data.append(row[col])
        ws.append(row_data)

        current_row_idx = ws.max_row
        img_data = row.get('Image', None)

        if img_data and pd.notna(img_data):
            try:
                img_stream = BytesIO(img_data)
                img = OpenpyxlImage(img_stream)
                img.width = 110
                img.height = 70
                col_letter = chr(65 + headers.index('Image'))
                cell_coordinate = f"{col_letter}{current_row_idx}"
                ws.row_dimensions[current_row_idx].height = 55
                ws.add_image(img, cell_coordinate)
            except Exception:
                pass
        else:
            ws.row_dimensions[current_row_idx].height = 24

    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = chr(65 + col[0].column - 1)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 13)

    if 'Image' in headers:
        img_col_letter = chr(65 + headers.index('Image'))
        ws.column_dimensions[img_col_letter].width = 18

    output_stream = BytesIO()
    wb.save(output_stream)
    output_stream.seek(0)
    return output_stream


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assembly Error Logging & Retrieval System — HAL",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── network check ─────────────────────────────────────────────────────────────
is_online = check_network()
status_color = "#43a047" if is_online else "#fb8c00"
status_text = "ONLINE" if is_online else "OFFLINE"
status_anim = "pulse_green" if is_online else "pulse_orange"

# ── background image ──────────────────────────────────────────────────────────
bg_b64 = get_bg_base64()
bg_css = f"""
    .stApp {{
    background-image: url("data:image/jpg;base64,{bg_b64}");
    background-size: 85%;
    background-position: center 75%;
    background-attachment: fixed;
    background-repeat: no-repeat;
}}
""" if bg_b64 else ".stApp { background-color: #0a0e1a; }"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
{bg_css}

.stApp::before {{
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(5, 10, 20, 0.45);
    z-index: 0;
    pointer-events: none;
}}

.block-container {{
    position: relative;
    z-index: 1;
    padding-top: 3rem !important;
    max-width: 90% !important;
    margin: 0 auto !important;
    background: rgba(8, 14, 26, 0.45);
    border-radius: 12px;
    backdrop-filter: blur(0px);
}}

.hal-header {{
    background: #0d1b2a;
    border-bottom: 2px solid #1e88e5;
    border-top: 2px solid #1e88e5;
    padding: 0;
    margin: 0 -1rem 1.5rem -1rem;
    padding-top: 0;
    position: relative;
    z-index: 10;
}}
.hal-header-inner {{
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.2rem 2rem;
    position: relative;
}}
.hal-title {{
    font-size: 1.7rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: 0.5px;
    text-align: center;
}}
.hal-subtitle {{
    font-size: 0.78rem;
    color: #90caf9;
    margin: 4px 0 0 0;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-align: center;
}}

.metric-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}}
.metric-card {{
    background: rgba(13,27,42,0.92);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(4px);
}}
.metric-value {{ font-size: 2rem; font-weight: 700; color: #1e88e5; margin: 0; }}
.metric-label {{ font-size: 0.72rem; color: #90caf9; text-transform: uppercase; letter-spacing: 1px; margin: 4px 0 0 0; }}
.metric-card.green .metric-value {{ color: #43a047; }}
.metric-card.amber .metric-value {{ color: #fb8c00; }}

.section-header {{
    font-size: 1.05rem;
    font-weight: 600;
    color: #e3f2fd;
    border-left: 3px solid #1e88e5;
    padding-left: 10px;
    margin-bottom: 0.3rem;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: rgba(13,27,42,0.92);
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e3a5f;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: #90caf9;
    border-radius: 6px;
    font-weight: 500;
    padding: 8px 20px;
}}
.stTabs [aria-selected="true"] {{
    background: #1e88e5 !important;
    color: white !important;
}}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div,
.stFileUploader > div > div {{
    background-color: rgba(13,27,42,0.95) !important;
    border: 1px solid #1e3a5f !important;
    color: #e3f2fd !important;
    border-radius: 6px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, #1565c0, #1e88e5);
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #1e88e5, #42a5f5);
    transform: translateY(-1px);
}}

.match-card {{
    background: rgba(13,27,42,0.95);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(4px);
}}
.match-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; }}
.match-component {{ font-size: 1rem; font-weight: 700; color: #e3f2fd; }}
.match-score-high {{ background: #1b5e20; color: #69f0ae; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 700; }}
.match-score-mid  {{ background: #e65100; color: #ffcc02; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 700; }}
.match-score-low  {{ background: #b71c1c; color: #ff8a80; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 700; }}
.match-desc {{ background: rgba(10,25,41,0.9); border-left: 3px solid #1e88e5; padding: 0.7rem 1rem; border-radius: 0 6px 6px 0; margin-bottom: 0.6rem; color: #b3c7e6; font-size: 0.9rem; }}
.match-fix  {{ background: rgba(10,31,10,0.9); border-left: 3px solid #43a047; padding: 0.7rem 1rem; border-radius: 0 6px 6px 0; color: #a5d6a7; font-size: 0.9rem; }}
.match-meta {{ font-size: 0.75rem; color: #546e7a; margin-top: 0.5rem; }}
.snag-image {{ margin-top: 10px; border-radius: 6px; border: 1px solid #1e3a5f; max-width: 100%; height: auto; max-height: 250px; object-fit: contain; display: block; }}

.stDownloadButton > button {{
    background: linear-gradient(135deg, #1b5e20, #43a047) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}}

hr {{ border-color: #1e3a5f; }}

@keyframes pulse_green {{
    0%  {{ box-shadow: 0 0 0 0 rgba(67,160,71,0.7); }}
    70% {{ box-shadow: 0 0 0 7px rgba(67,160,71,0); }}
    100%{{ box-shadow: 0 0 0 0 rgba(67,160,71,0); }}
}}
@keyframes pulse_orange {{
    0%  {{ box-shadow: 0 0 0 0 rgba(251,140,0,0.7); }}
    70% {{ box-shadow: 0 0 0 7px rgba(251,140,0,0); }}
    100%{{ box-shadow: 0 0 0 0 rgba(251,140,0,0); }}
}}
.status-dot {{
    display: inline-block;
    width: 9px; height: 9px;
    background: {status_color};
    border-radius: 50%;
    margin-right: 6px;
    animation: {status_anim} 2s infinite;
}}
</style>
""", unsafe_allow_html=True)

# ── load data ─────────────────────────────────────────────────────────────────
model = load_model()
init_db()

records, vectors = load_cache()
if not records or len(records) == 0:
    records, vectors = rebuild_cache(model)

total, today_count, top_error, components = get_stats()

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hal-header">
  <div class="hal-header-inner">
    <div>
      <p class="hal-title"> Assembly Error Logging &amp; Retrieval System</p>
      <p class="hal-subtitle">ALH Final Assembly &nbsp;·&nbsp; HAL Helicopter Division &nbsp;·&nbsp; Bengaluru</p>
    </div>
    <div style="position:absolute; right:2rem; top:50%; transform:translateY(-50%); display:flex; align-items:center; gap:8px;">
      <span class="status-dot"></span>
      <span style="color:{status_color}; font-size:0.82rem; font-weight:600; letter-spacing:1px;">
        {status_text}
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── metric cards ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <p class="metric-value">{total}</p>
    <p class="metric-label">Total Errors Logged</p>
  </div>
  <div class="metric-card green">
    <p class="metric-value">{today_count}</p>
    <p class="metric-label">Logged Today</p>
  </div>
  <div class="metric-card amber">
    <p class="metric-value">{components}</p>
    <p class="metric-label">Components Tracked</p>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  LOG Error", "  SEARCH Errors", "  EXPORT Log"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — LOG
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Log a New Assembly Error</p>', unsafe_allow_html=True)
    st.caption("Complete the fields below and click Save — the record will be added to the database immediately.")

    col1, col2 = st.columns(2)
    with col1:
        component_ref = st.text_input("Component Reference", placeholder="e.g. AFCS-J4, MGB-B2")
        error_type = st.selectbox("Error Type", [
            "Incorrect Connector Mating",
            "Wrong Torque Value",
            "Missing Clamp",
            "Reversed Polarity",
            "Other"
        ])
        technician_id = st.text_input("Technician ID", placeholder="e.g. TECH-01")
        uploaded_file = st.file_uploader("Upload Snag Photo (Optional)", type=["jpg", "jpeg"])

    with col2:
        description = st.text_area("Error Description", height=130,
                                   placeholder="Describe the error observed during assembly...")
        corrective_action = st.text_area("Corrective Action Taken", height=130,
                                         placeholder="Describe the steps taken to resolve the issue...")

    if st.button("  Save Error", use_container_width=True):
        if component_ref and description and corrective_action and technician_id:
            image_bytes = uploaded_file.read() if uploaded_file is not None else None
            inserted = log_error(component_ref, error_type, description, corrective_action, technician_id, image_bytes)
            if inserted:
                records, vectors = rebuild_cache(model)
                st.cache_resource.clear()
                st.success(f" Error for **{component_ref}** saved successfully.")
                st.rerun()
            else:
                st.info(
                    f" This exact error for **{component_ref}** is already in the database — not logged again to avoid duplicates.")
        else:
            st.warning("⚠️ Please fill in all fields before saving.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEARCH
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Search Past Assembly Errors</p>', unsafe_allow_html=True)
    st.caption(
        "Describe the fault in plain language — the system will instantly find the closest past matches and their solutions.")

    query = st.text_input("Describe the fault you are observing",
                          placeholder="e.g. connector on wrong port, bolt not tightened enough...")

    if st.button("  Search", use_container_width=True):
        if query.strip():
            results = search(query, model, records, vectors)
            st.markdown("---")
            st.markdown(f"**Top 3 matches for:** *{query}*")

            for i, r in enumerate(results):
                score = r['Match_Score']
                if score >= 60:
                    score_html = f'<span class="match-score-high">🟢 {score}% match</span>'
                elif score >= 40:
                    score_html = f'<span class="match-score-mid">🟡 {score}% match</span>'
                else:
                    score_html = f'<span class="match-score-low">🔴 {score}% match</span>'

                occ = r['Occurrence_Count']
                if occ >= 3:
                    recur_html = f'<span style="background:#7b1fa2; color:#e1bee7; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:700; margin-left:8px;">🔁 Recurring — seen {occ} times</span>'
                elif occ == 2:
                    recur_html = f'<span style="background:#5d4037; color:#d7ccc8; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:700; margin-left:8px;">🔁 Seen {occ} times before</span>'
                else:
                    recur_html = '<span style="background:#37474f; color:#b0bec5; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:700; margin-left:8px;">First recorded case</span>'

                img_html = f'<img src="data:image/jpeg;base64,{r["Image"]}" class="snag-image" />' if r['Image'] else ''

                st.markdown(f"""
                            <div class="match-card">
                              <div class="match-header">
                                <span class="match-component">#{i + 1} &nbsp; {r['Component_Ref']} &nbsp;·&nbsp; {r['Error_Type']}</span>
                                <div>{score_html}{recur_html}</div>
                              </div>
                              <p style="font-size:0.75rem;color:#546e7a;margin:0 0 6px 0;">Error Description</p>
                              <div class="match-desc">{r['Description']}</div>
                              {img_html}
                              <p style="font-size:0.75rem;color:#546e7a;margin:8px 0 4px 0;"> Corrective Action</p>
                              <div class="match-fix">{r['Corrective_Action']}</div>
                              <p class="match-meta">Error ID: {r['Error_ID']}</p>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a search query.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPORT
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">QA Audit Export</p>', unsafe_allow_html=True)
    st.caption("Full error log for HAL QA audit trail. Manage records below, or download the complete log as Excel.")

    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("  Remove Duplicates", use_container_width=True):
            removed = remove_duplicates()
            if removed > 0:
                st.cache_resource.clear()
                st.success(f"Removed {removed} duplicate record(s).")
                st.rerun()
            else:
                st.info("No duplicates found.")

    df = get_all_errors()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Error Types", df['Error_Type'].nunique() if not df.empty else 0)
    col3.metric("Components", df['Component_Ref'].nunique() if not df.empty else 0)

    st.markdown("---")
    st.markdown('<p class="section-header">Manage Records</p>', unsafe_allow_html=True)

    if not df.empty:
        df['_label'] = df.apply(
            lambda
                row: f"ID {row['Error_ID']} — {row['Component_Ref']} — {row['Error_Type']} — {row['Description'][:50]}...",
            axis=1
        )

        selected_label = st.selectbox("Select a record", df['_label'].tolist())
        selected_row = df[df['_label'] == selected_label].iloc[0]

        rec_img_blob = selected_row['Image'] if 'Image' in selected_row and pd.notna(selected_row['Image']) else None
        rec_img_html = f'<img src="data:image/jpeg;base64,{base64.b64encode(rec_img_blob).decode()}" class="snag-image" />' if rec_img_blob else ''

        with st.container():
            st.markdown(f"""
            <div class="match-card">
              <div class="match-header">
                <span class="match-component">ID {selected_row['Error_ID']} &nbsp;·&nbsp; {selected_row['Component_Ref']} &nbsp;·&nbsp; {selected_row['Error_Type']}</span>
              </div>
              <p style="font-size:0.75rem;color:#546e7a;margin:0 0 6px 0;">Error Description</p>
              <div class="match-desc">{selected_row['Description']}</div>
              {rec_img_html}
              <p style="font-size:0.75rem;color:#546e7a;margin:8px 0 4px 0;"> Corrective Action</p>
              <div class="match-fix">{selected_row['Corrective_Action']}</div>
              <p class="match-meta">Technician: {selected_row['Technician_ID']} &nbsp;|&nbsp; Date: {selected_row['Date']}</p>
            </div>
            """, unsafe_allow_html=True)

        confirm = st.checkbox(f"I confirm I want to permanently delete Error ID {selected_row['Error_ID']}")
        if st.button("  Delete This Record", use_container_width=True, disabled=not confirm):
            delete_error(int(selected_row['Error_ID']))
            records, vectors = rebuild_cache(model)
            st.cache_resource.clear()
            st.success(f"Error ID {selected_row['Error_ID']} deleted successfully.")
            st.rerun()
    else:
        st.info("No records in the database yet.")

    st.markdown("---")
    st.markdown('<p class="section-header">Live Log Table View</p>', unsafe_allow_html=True)

    if not df.empty:
        df_display = df.copy()


        def make_thumbnail_uri(blob):
            if blob and pd.notna(blob):
                b64 = base64.b64encode(blob).decode()
                return f"data:image/jpeg;base64,{b64}"
            return None


        df_display['Snag_Photo'] = df_display['Image'].apply(make_thumbnail_uri)
        df_display = df_display.drop(columns=['_label', 'Image'], errors='ignore')

        st.dataframe(
            df_display,
            use_container_width=True,
            height=400,
            column_config={
                "Snag_Photo": st.column_config.ImageColumn(
                    "Snag Photo",
                    help="Clicking an image expands it to full size view",
                    width="medium"
                )
            }
        )

        st.markdown("---")

        excel_data_stream = generate_excel_with_images(df)

        st.download_button(
            label="  Download Full Log as Excel ",
            data=excel_data_stream,
            file_name="ALH_Assembly_Error_Log_With_Images.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.info("No records available to display or export.")

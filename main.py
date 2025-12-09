import streamlit as st
import pandas as pd
import os
import datetime
import plotly.express as px
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import io
import hashlib
import json

# ----------------------------------------
# 🎨 APP CONFIGURATION
# ----------------------------------------
st.set_page_config(
    page_title="Cross-Domain Knowledge Mapping Dashboard",
    layout="wide",
    page_icon="🧭"
)

# ----------------------------------------
# 📁 FILE PATHS CONFIGURATION
# ----------------------------------------
EMBEDDINGS_PATH = "cross_domain_embeddings.pkl"
KNOWLEDGE_GRAPH_PATH = "knowledge_graph.html"
FEEDBACK_FILE = "feedback.csv"
USERS_FILE = "users.json"

# ----------------------------------------
# 🔐 USER AUTHENTICATION FUNCTIONS
# ----------------------------------------
def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_users(users: dict):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)


def register_user(username, password, email, role="student"):
    """Register a new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "role": role,
        "created_at": str(datetime.datetime.now()),
        "saved_graphs": [],
        "preferences": {}
    }
    save_users(users)
    return True, "Registration successful"


def authenticate_user(username, password):
    """Authenticate user credentials"""
    users = load_users()
    if username not in users:
        return False, "User not found"
    
    if users[username]["password"] == hash_password(password):
        return True, users[username]
    return False, "Invalid password"


def update_user_preferences(username, preferences):
    """Update user preferences"""
    users = load_users()
    if username in users:
        users[username]["preferences"] = preferences
        save_users(users)
        return True
    return False


def save_graph_to_profile(username, graph_name):
    """Save graph info to user profile"""
    users = load_users()
    if username in users:
        if graph_name not in [g["name"] for g in users[username].get("saved_graphs", [])]:
            users[username].setdefault("saved_graphs", []).append(
                {"name": graph_name, "saved_at": str(datetime.datetime.now())}
            )
            save_users(users)
        return True
    return False

# ----------------------------------------
# 🔐 LOGIN / REGISTRATION PAGE
# ----------------------------------------
def login_page():
    """Display login and registration page with custom styling"""
    # Custom CSS for login page
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #e0f7fa 0%, #fff9c4 50%, #f3e5f5 100%) !important;
        }
        .login-container {
            background: white;
            padding: 3rem;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            margin: 2rem auto;
            max-width: 500px;
        }
        .login-header {
            text-align: center;
            color: #2d6a4f;
            margin-bottom: 2rem;
        }
        .login-header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            color: #2d6a4f !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .login-header p {
            font-size: 1.1rem;
            color: #555;
        }
        .feature-box {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 2px solid #e0e0e0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            transition: all 0.3s;
        }
        .feature-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
            border-color: #667eea;
        }
        .feature-box h3 {
            color: #667eea !important;
            margin-bottom: 0.5rem;
            font-size: 1.3rem;
            font-weight: 700;
        }
        .feature-box p {
            color: #333;
            margin: 0;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        .stTextInput > div > div > input {
            background-color: #f8f9fa;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            padding: 12px;
            font-size: 16px;
            transition: all 0.3s;
            color: #333;
        }
        .stTextInput > div > div > input:focus {
            border: 2px solid #667eea;
            box-shadow: 0 0 15px rgba(102,126,234,0.2);
            background-color: white;
        }
        .stSelectbox > div > div > div {
            background-color: #f8f9fa;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
        }
        .icon-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            margin: 0.5rem;
            box-shadow: 0 5px 20px rgba(102,126,234,0.3);
            transition: all 0.3s;
        }
        .icon-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(102,126,234,0.4);
        }
        .icon-card h4 {
            color: white !important;
            margin: 0.5rem 0;
            font-weight: 600;
        }
        .icon-card h2 {
            filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.2));
        }
        .login-tab-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-top: 1rem;
            box-shadow: 0 5px 20px rgba(102,126,234,0.2);
        }
        .register-tab-header {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-top: 1rem;
            box-shadow: 0 5px 20px rgba(240,147,251,0.2);
        }
        div[data-baseweb="tab-list"] {
            gap: 10px;
            background: white;
            padding: 10px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        div[data-baseweb="tab"] {
            border-radius: 10px;
            font-weight: 600;
            font-size: 1.1rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 2rem !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            box-shadow: 0 5px 20px rgba(102,126,234,0.3) !important;
            transition: all 0.3s !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 30px rgba(102,126,234,0.4) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Simple Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0; background: lightblue; border-radius: 20px; 
                margin-bottom: 2rem; box-shadow: 0 5px 20px rgba(0,0,0,0.1);'>
        <h1 style='font-size: 2.8rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: green; 
                   font-weight: 900; margin: 0;'>
            🧭 Cross-Domain Knowledge Mapper
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Login / Registration in centered container
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
        # LOGIN TAB
        with tab1:
            st.markdown("""
            <div class='login-tab-header'>
                <h2 style='color: white; text-align: center; margin-bottom: 0; font-weight: 700;'>
                    Welcome Back! 👋
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            login_username = st.text_input("👤 Username", key="login_user", placeholder="Enter your username")
            login_password = st.text_input("🔒 Password", type="password", key="login_pass", placeholder="Enter your password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 LOGIN", use_container_width=True, type="primary"):
                if login_username and login_password:
                    with st.spinner("Authenticating..."):
                        success, result = authenticate_user(login_username, login_password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = login_username
                            st.session_state.user_data = result
                            st.session_state.user_role = result.get("role", "student")
                            st.success(f"✅ Welcome back, {login_username}!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"❌ {result}")
                else:
                    st.warning("⚠️ Please enter both username and password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 Don't have an account? Switch to the Register tab!")
        
        # REGISTER TAB
        with tab2:
            st.markdown("""
            <div class='register-tab-header'>
                <h2 style='color: white; text-align: center; margin-bottom: 0; font-weight: 700;'>
                    Join Us Today! 🎉
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            reg_username = st.text_input("👤 Username", key="reg_user", placeholder="Choose a unique username")
            reg_email = st.text_input("📧 Email", key="reg_email", placeholder="your.email@example.com")
            
            col_pass1, col_pass2 = st.columns(2)
            with col_pass1:
                reg_password = st.text_input("🔒 Password", type="password", key="reg_pass", placeholder="Min 6 characters")
            with col_pass2:
                reg_password_confirm = st.text_input("🔒 Confirm", type="password", key="reg_pass_confirm", placeholder="Re-enter password")
            
            reg_role = st.selectbox(
                "🎭 Select Your Role",
                ["student", "researcher", "admin"],
                key="reg_role",
                format_func=lambda x: (
                    f"🎓 {x.title()}" if x == "student"
                    else f"🔬 {x.title()}" if x == "researcher"
                    else f"👑 {x.title()}"
                )
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("✨ CREATE ACCOUNT", use_container_width=True, type="primary"):
                if reg_username and reg_email and reg_password and reg_password_confirm:
                    if reg_password != reg_password_confirm:
                        st.error("❌ Passwords do not match!")
                    elif len(reg_password) < 6:
                        st.error("❌ Password must be at least 6 characters!")
                    elif "@" not in reg_email:
                        st.error("❌ Please enter a valid email address!")
                    else:
                        with st.spinner("Creating your account..."):
                            success, message = register_user(reg_username, reg_password, reg_email, reg_role)
                            if success:
                                st.success(f"✅ {message}! Please login now.")
                                st.balloons()
                            else:
                                st.error(f"❌ {message}")
                else:
                    st.warning("⚠️ Please fill in all fields!")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 Already have an account? Switch to the Login tab!")
    
    # Bottom features section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <h2 style='text-align: center; color: #2d6a4f; font-weight: 700; font-size: 2.5rem; margin: 2rem 0;'>
        Why Choose Knowledge Mapper?
    </h2>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='feature-box'>
            <h3>🎯 Interdisciplinary Insights</h3>
            <p>Connect concepts across Science, Technology, Literature, and more</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-box'>
            <h3>⚡ AI-Powered Analysis</h3>
            <p>Advanced NLP models for entity recognition and relationship extraction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-box'>
            <h3>📊 Interactive Visualization</h3>
            <p>Explore dynamic knowledge graphs and semantic connections</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------
# 🌈 PROFESSIONAL MULTICOLOR THEME (GLOBAL)
# ----------------------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #fff9e6;
            padding-top: 1rem;
        }
        [data-testid="stSidebar"] * {
            color: #264653 !important;
            font-size: 17px !important;
            font-weight: 600;
        }
        /* Don't override .stApp here – login page uses its own bg */
        h1, h2, h3 {
            color: #2d6a4f !important;
            font-weight: 700 !important;
        }
        div.stButton > button {
            background-color: #b7e4c7;
            color: #182c25;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            transition: 0.3s;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #95d5b2;
            transform: scale(1.05);
        }
        .block-container {
            background-color: #f3f0ff !important;
        }
        .stDataFrame, .stTable {
            background-color: #fff9e6 !important;
        }
        iframe {
            border-radius: 10px;
            border: 2px solid #b7e4c7;
        }
        .upload-section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .user-info {
            background-color: #b7e4c7;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 📘 SESSION STATE INITIALIZATION
# ----------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'embeddings_generated' not in st.session_state:
    st.session_state.embeddings_generated = False

# ----------------------------------------
# 🔐 CHECK AUTHENTICATION
# ----------------------------------------
if not st.session_state.logged_in:
    login_page()
    st.stop()

# ----------------------------------------
# 📚 SIDEBAR NAVIGATION & USER INFO
# ----------------------------------------
st.sidebar.markdown(f"""
<div class="user-info">
    <b>👤 User:</b> {st.session_state.username}<br>
    <b>🎭 Role:</b> {st.session_state.user_role.title()}
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_data = None
    st.session_state.user_role = None
    st.rerun()

st.sidebar.markdown("---")

# Role-based page access
pages = [
    "📤 Upload Dataset",
    "🏠 Overview",
    "🧠 Entity & Relation Extraction",
    "🌐 Knowledge Graph",
    "🔍 Semantic Search",
    "🧩 Top 10 Sentences",
    "💬 Feedback Section",
]

# Admin-only pages
if st.session_state.user_role == "admin":
    pages.extend([
        "📈 Feedback Analysis",
        "🛠 Admin Tools",
        "👥 User Management",
    ])

pages.append("💾 Download Options")
pages.append("⚙️ User Preferences")

choice = st.sidebar.radio("📑 Navigate Pages", pages)

# ----------------------------------------
# 💬 LOAD FEEDBACK FILE
# ----------------------------------------
if os.path.exists(FEEDBACK_FILE):
    feedback_df = pd.read_csv(FEEDBACK_FILE)
else:
    feedback_df = pd.DataFrame(
        columns=["record_id", "user", "feedback_type", "comment", "status", "timestamp"]
    )

# ----------------------------------------
# 📤 UNIVERSAL UPLOAD DATASET PAGE
# ----------------------------------------
if choice == "📤 Upload Dataset":
    st.title("📤 Upload Your Dataset (Any Format Supported)")

    uploaded_file = st.file_uploader(
        "Upload CSV, Excel, or TXT file",
        type=['csv', 'xlsx', 'xls', 'txt'],
        help="Headers are optional — system auto-detects the text column."
    )

    if uploaded_file is not None:
        try:
            file_ext = uploaded_file.name.split(".")[-1].lower()

            # ======================================================
            # 📌 UNIVERSAL FILE LOADING WITH ERROR TOLERANCE
            # ======================================================
            if file_ext == "csv":
                try:
                    df = pd.read_csv(
                        uploaded_file,
                        engine="python",
                        on_bad_lines="skip",     # Skip corrupted CSV rows
                        sep=None                 # Auto-detect delimiter
                    )
                except Exception:
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file,
                        engine="python",
                        delimiter=",",
                        on_bad_lines="skip"
                    )

            elif file_ext in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)

            else:  # TXT files
                df = pd.read_csv(
                    uploaded_file, 
                    header=None,
                    names=["sentence"]
                )

            st.info("✅ File loaded successfully. Auto-detecting structure...")

            # Force column names (headerless-safe)
            df.columns = [f"col_{i}" for i in range(len(df.columns))]

            # ======================================================
            # 📌 AUTO-DETECT SENTENCE COLUMN (Longest text column)
            # ======================================================
            sentence_col = df.apply(
                lambda col: col.astype(str).str.len().mean()
            ).idxmax()

            processed_df = pd.DataFrame()
            processed_df["id"] = range(1, len(df) + 1)
            processed_df["sentence"] = df[sentence_col].astype(str)

            # ======================================================
            # 📌 AUTO-DETECT DOMAIN COLUMN
            # ======================================================
            domain_candidates = [
                col for col in df.columns
                if col != sentence_col
                and df[col].nunique() < len(df) * 0.5         # categorical-like
            ]

            if domain_candidates:
                processed_df["domain"] = df[domain_candidates[0]].astype(str)
            else:
                processed_df["domain"] = "Unknown"

            # ======================================================
            # 📌 AUTO-DETECT LABEL COLUMN
            # ======================================================
            label_candidates = [
                col for col in df.columns
                if col not in [sentence_col] + domain_candidates
            ]

            if label_candidates:
                processed_df["label"] = df[label_candidates[0]].astype(str)
            else:
                processed_df["label"] = "N/A"

            # ======================================================
            # 📌 SAVE FINAL CLEAN DATASET
            # ======================================================
            st.session_state.df = processed_df

            st.success("🎉 Dataset processed successfully!")

            st.subheader("📋 Parsed Dataset Preview")
            st.dataframe(processed_df.head(10), use_container_width=True)

            st.subheader("🔍 Auto-Detected Column Mapping")
            st.json({
                "ID Column": "Generated automatically",
                "Sentence Column": sentence_col,
                "Domain Column": domain_candidates[0] if domain_candidates else "Created",
                "Label Column": label_candidates[0] if label_candidates else "Created",
            })

            st.balloons()

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.stop()

    else:
        st.info("👆 Upload a dataset to begin — any structure is accepted now.")

# ----------------------------------------
# 🏠 OVERVIEW
# ----------------------------------------
elif choice == "🏠 Overview":

    # Ensure dataset exists
    df = st.session_state.get("df", None)

    if df is None:
        st.warning("⚠️ No dataset uploaded yet.")
        st.info("Please upload a dataset from the '📤 Upload Dataset' page.")
        st.stop()

    st.title("🧭 Dataset Overview & Insights")

    st.write(f"""
    Welcome **{st.session_state.username}**!  
    Below is a smart summary of your dataset, including domain trends, label usage,
    sentence structure, and automatic insights.
    """)

    # -----------------------------------------------------------------------------
    # 🔢 TOP METRICS
    # -----------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📌 Total Records", len(df))

    with col2:
        st.metric("🍀 Unique Domains", df["domain"].nunique())

    with col3:
        st.metric("🏷 Unique Labels", df["label"].nunique())

    with col4:
        avg_length = df["sentence"].astype(str).apply(len).mean()
        st.metric("✏️ Avg Sentence Length", f"{avg_length:.1f} chars")

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 📊 DOMAIN DISTRIBUTION
    # -----------------------------------------------------------------------------
    st.subheader("📊 Domain Distribution")
    domain_counts = df["domain"].value_counts()

    st.bar_chart(domain_counts)

    # -----------------------------------------------------------------------------
    # 🏷 LABEL DISTRIBUTION
    # -----------------------------------------------------------------------------
    st.subheader("🏷 Label Type Distribution")
    label_counts = df["label"].value_counts()
    st.bar_chart(label_counts)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 🧠 TOP WORD FREQUENCIES
    # -----------------------------------------------------------------------------
    st.subheader("🧠 Most Frequent Words in Sentences")

    import re
    from collections import Counter

    words = []

    for s in df["sentence"].astype(str):
        cleaned = re.sub(r"[^a-zA-Z ]", "", s).lower().split()
        words.extend(cleaned)

    common_words = Counter(words).most_common(10)

    st.write("### 🔝 Top 10 Most Common Words")
    st.table(pd.DataFrame(common_words, columns=["Word", "Frequency"]))

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 📑 DATASET STRUCTURE
    # -----------------------------------------------------------------------------
    st.subheader("📑 Dataset Structure")

    colA, colB = st.columns(2)
    with colA:
        st.write("### Columns in Dataset:")
        st.write(df.columns.tolist())

    with colB:
        st.write("### Missing Values per Column:")
        st.write(df.isna().sum())

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 🔍 INSIGHTS
    # -----------------------------------------------------------------------------
    st.subheader("🔍 Automated Insights")

    insights = []

    # 1) Most frequent domain
    top_domain = domain_counts.idxmax()
    insights.append(f"⭐ **Most frequent domain:** {top_domain}")

    # 2) Longest sentence
    longest_sentence = df.loc[df['sentence'].str.len().idxmax()]['sentence']
    insights.append(f"📝 **Longest sentence length:** {len(longest_sentence)} characters")

    # 3) Dataset balance
    if df["domain"].nunique() > 1:
        ratio = domain_counts.max() / domain_counts.min()
        if ratio > 3:
            insights.append("⚠️ **Dataset is imbalanced across domains.**")
        else:
            insights.append("✅ **Dataset has well-balanced domains.**")

    # Display insights
    for insight in insights:
        st.info(insight)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 👀 OPTIONAL PREVIEW
    # -----------------------------------------------------------------------------
    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ----------------------------------------
# 🧠 ENTITY & RELATION EXTRACTION
# ----------------------------------------
elif choice == "🧠 Entity & Relation Extraction":
    import spacy

    st.title("🧠 Entity & Relation Extraction")
    st.write("""
    This module extracts **Named Entities** (NER) and **Relation Triples** (SVO: Subject–Verb–Object)
    using spaCy NLP.  
    Once processed, new columns will be added to your dataset:
    - `entities`
    - `relations`
    """)

    # Try loading the spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        st.success("✅ spaCy model loaded successfully!")
    except Exception as e:
        st.error("❌ spaCy model 'en_core_web_sm' is not installed.")
        st.info("Install it using: `python -m spacy download en_core_web_sm`")
        st.stop()

    # Button to run NLP processing
    if st.button("🚀 Run Entity & Relation Extraction"):
        with st.spinner("Processing dataset... Please wait ⏳"):
            df = st.session_state.df.copy()

            all_entities = []
            all_relations = []

            for sentence in df['sentence']:
                doc = nlp(sentence)

                # ---------- Named Entity Extraction ----------
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                all_entities.append(entities)

                # ---------- Relation Extraction (SVO Triples) ----------
                triples = []
                for token in doc:
                    if token.dep_ == "ROOT" and token.pos_ == "VERB":  # find verb
                        subj = [child.text for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                        obj = [child.text for child in token.children if child.dep_ in ("dobj", "pobj")]

                        if subj and obj:
                            triples.append((subj[0], token.lemma_, obj[0]))

                all_relations.append(triples)

            # Add results to dataframe
            df["entities"] = all_entities
            df["relations"] = all_relations

            # Save back to session
            st.session_state.df = df

            st.success("✅ NLP processing completed!")
            st.balloons()

    # Display processed data (if available)
    if "entities" in st.session_state.df.columns:
        st.subheader("📘 Extracted Entities & Relations")
        st.dataframe(
            st.session_state.df[["sentence", "entities", "relations"]],
            use_container_width=True
        )
    else:
        st.warning("⚠ Dataset not processed yet. Click the button above to run NLP.")

# # ----------------------------------------
# 🌐 KNOWLEDGE GRAPH (FIXED + NEW DESIGN)
# ----------------------------------------
elif choice == "🌐 Knowledge Graph":

    # Load dataset
    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠ Please upload a dataset first.")
        st.stop()

    st.header("🌐 Interactive Knowledge Graph Visualization")

    st.write("""
    This graph is styled like your reference image:<br>
    ✔ Green = central concepts<br>
    ✔ Blue = related entities<br>
    ✔ Pink edges = strong relations<br>
    ✔ Dashed grey = normal relations<br>
    """, unsafe_allow_html=True)

    # Imports
    try:
        import networkx as nx
        from pyvis.network import Network
        import re
        import spacy
        from collections import Counter
    except Exception as e:
        st.error(f"Missing required libraries: {e}")
        st.stop()

    # Load spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        st.error("spaCy model missing. Run: python -m spacy download en_core_web_sm")
        st.stop()

    # ---------------------------------------------------
    # BUTTON
    # ---------------------------------------------------
    build = st.button("⚙️ Build Knowledge Graph")

    if build:
        with st.spinner("Generating graph..."):

            G = nx.Graph()
            freq = Counter()

            # -------- SIMPLE ENTITY EXTRACTION BACKUP --------
            def fallback_entities(text):
                parts = re.split(r"\b(uses?|affects?|contains?|requires?|forms?|drives?)\b",
                                 text, flags=re.IGNORECASE)
                ents = [p.strip() for p in parts if p.strip()]
                return ents[:2]

            # ------------- PROCESS DATASET --------------------
            for _, row in df.iterrows():

                sentence = str(row["sentence"])

                # spaCy NER
                doc = nlp(sentence)
                ents = [ent.text for ent in doc.ents]

                # fallback if spaCy fails
                if len(ents) < 2:
                    ents = fallback_entities(sentence)

                if len(ents) < 2:
                    continue  # skip sentences with <2 entities

                # use 2 entities max
                src, dst = ents[:2]

                freq[src] += 1
                freq[dst] += 1

                G.add_node(src)
                G.add_node(dst)
                G.add_edge(src, dst, label=row.get("label", ""))

            # If graph empty
            if len(G.nodes()) == 0:
                st.error("❌ No entities found — cannot build graph.")
                st.stop()

            # ---------------------------------------------------
            # BUILD PYVIS GRAPH
            # ---------------------------------------------------
            net = Network(height="750px", width="100%", bgcolor="#fff", font_color="black")

            # Keep layout stable
            net.set_options("""
            const options = {
              "nodes": { "borderWidth": 1 },
              "edges": { 
                "smooth": { "type": "continuous" } 
              },
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -2000,
                  "centralGravity": 0.30,
                  "springLength": 160
                }
              }
            }
            """)

            # Colors
            GREEN = "#6FD88F"
            BLUE = "#4A86E8"
            PINK = "#FF6AA9"
            GRAY = "#B5B5B5"

            # Add nodes
            for node in G.nodes():
                color = GREEN if freq[node] > 1 else BLUE
                size = 28 if freq[node] > 1 else 18

                net.add_node(node, label=node, color=color, size=size)

            # Add edges
            for src, dst, data in G.edges(data=True):

                label = data.get("label", "")
                strong_rels = ["affects", "causes", "leads", "increases", "reduces"]

                if any(rel in label.lower() for rel in strong_rels):
                    net.add_edge(src, dst, color=PINK, width=3)
                else:
                    net.add_edge(src, dst, color=GRAY, width=2, dashes=True)

            # Save
            net.save_graph(KNOWLEDGE_GRAPH_PATH)

            st.success("🎉 Knowledge Graph Generated Successfully!")
            st.rerun()

    # ---------------------------------------------------
    # DISPLAY GRAPH
    # ---------------------------------------------------
    if os.path.exists(KNOWLEDGE_GRAPH_PATH):
        with open(KNOWLEDGE_GRAPH_PATH, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=770, scrolling=True)

# # ----------------------------------------
# 🔍 SEMANTIC SEARCH
# ----------------------------------------
elif choice == "🔍 Semantic Search":
    st.header("🔍 Semantic Search - Explore Cross Domain Meaning")

    # --------------------------
    # 1️⃣ Load dataset first
    # --------------------------
    df = st.session_state.get("df", None)
    if df is None:
        st.warning("⚠️ Please upload a dataset first!")
        st.stop()

    # --------------------------
    # 2️⃣ Ensure embeddings exist
    # --------------------------
    if not os.path.exists(EMBEDDINGS_PATH):
        st.warning("⚠️ Embeddings not found. Generate them first.")
        st.info("""
        To generate embeddings:
        1️⃣ Install ➜ `pip install sentence-transformers`
        2️⃣ Click the button below
        """)

        if st.button("🚀 Generate Embeddings"):
            try:
                with st.spinner("Loading MiniLM model..."):
                    model = SentenceTransformer("all-MiniLM-L6-v2")

                with st.spinner("Generating embeddings... This may take a minute."):
                    embeddings = model.encode(
                        df["sentence"].tolist(),
                        show_progress_bar=True,
                        convert_to_numpy=True
                    )

                embdf = df.copy()
                embdf["embedding"] = embeddings.tolist()
                embdf.to_pickle(EMBEDDINGS_PATH)

                st.success("✅ Embeddings generated successfully!")
                st.info("Reload the Semantic Search page.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
            st.stop()

    # --------------------------
    # 3️⃣ Load embeddings
    # --------------------------
    try:
        embdf = pd.read_pickle(EMBEDDINGS_PATH)
    except Exception as e:
        st.error(f"❌ Could not load embeddings: {e}")
        st.stop()

    st.sidebar.success(f"📌 Embeddings loaded: {len(embdf)} sentences")

    # --------------------------
    # 4️⃣ Query selection
    # --------------------------
    st.write("Enter a query manually or select from frequent sentences:")

    top_sentences = embdf["sentence"].value_counts().head(3).index.tolist()
    query_options = top_sentences + ["Manual Entry"]

    selected_query = st.selectbox("Choose a Query:", query_options)

    if selected_query == "Manual Entry":
        final_query = st.text_input("Type your query here:").strip()
    else:
        final_query = selected_query.strip()

    st.write(f"**Current Query:** `{final_query if final_query else '(none)'}`")

    # --------------------------
    # 5️⃣ Cache model
    # --------------------------
    @st.cache_resource
    def load_semantic_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    # --------------------------
    # 6️⃣ Perform Search
    # --------------------------
    if st.button("🔍 Search"):
        if not final_query:
            st.warning("⚠️ Please enter a query.")
            st.stop()

        try:
            st.subheader(f"🔎 Top 3 Semantic Matches for: '{final_query}'")

            model = load_semantic_model()

            # Encode query (⚠ FIXED: no .astype)
            query_embedding = model.encode(
                final_query,
                convert_to_tensor=True
            ).float()

            # Convert stored embeddings into tensor
            stored_embeddings = np.array(embdf["embedding"].tolist(), dtype=np.float32)
            embeddings_tensor = torch.tensor(stored_embeddings, dtype=torch.float32)

            # Cosine similarity
            similarity_scores = util.cos_sim(query_embedding, embeddings_tensor)
            top_results = similarity_scores[0].topk(3)

            # -------------------------------
            # Display search results
            # -------------------------------
            for idx, score in zip(top_results.indices, top_results.values):
                row = embdf.iloc[int(idx)]

                st.markdown(f"""
                <div style="background:#b7e4c7;border-radius:8px;padding:12px;margin-bottom:10px;">
                    ✅ <b style="color:#2d6a4f;">Domain:</b> {row['domain']}<br>
                    💬 <b style="color:#2d6a4f;">Sentence:</b> {row['sentence']}<br>
                    🏷 <b style="color:#2d6a4f;">Label:</b> {row['label']}<br>
                    📈 <b style="color:#ffbe0b;">Similarity Score:</b>
                        <span style="color:#3c096c;font-weight:bold;">{float(score):.4f}</span>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error during search: {e}")
            st.info("Try deleting embeddings.pkl and regenerate again.")

# ----------------------------------------
# 🧩 TOP 10 SENTENCES
# ----------------------------------------
elif choice == "🧩 Top 10 Sentences":
    st.header("🧩 Top 10 Frequent Sentences in Dataset")

    # ✅ Safely get the dataset
    df = st.session_state.get("df", None)
    if df is None:
        st.warning("⚠️ Please upload a dataset first from '📤 Upload Dataset'.")
        st.stop()

    if "sentence" not in df.columns:
        st.error("❌ The loaded dataset has no 'sentence' column.")
        st.stop()

    # Compute top 10 most frequent sentences
    top_objects = df["sentence"].astype(str).value_counts().head(10)

    st.subheader("📋 Top Sentences List")
    for i, (sentence, count) in enumerate(top_objects.items(), start=1):
        st.markdown(
            f"""
            <div style="background:#e8f5e9;
                        padding:12px;
                        margin-bottom:8px;
                        border-radius:8px;
                        border-left:6px solid #1b5e20;">
                <b>{i}. {sentence}</b><br>
                <span style="color:#2e7d32;">Appears {count} times</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------------------
# 💬 FEEDBACK SECTION
# ----------------------------------------
elif choice == "💬 Feedback Section":
    st.header("💬 Feedback Section")

    # ------------------------------------------
    # 1️⃣ Initialize feedback storage
    # ------------------------------------------
    if "feedback_df" not in st.session_state:
        st.session_state.feedback_df = pd.DataFrame(
            columns=["feedback_id", "record_id", "user", "feedback_type",
                     "comment", "status", "timestamp"]
        )

    if "last_feedback_hash" not in st.session_state:
        st.session_state.last_feedback_hash = None

    feedback_df = st.session_state.feedback_df

    # ------------------------------------------
    # 2️⃣ Display feedback table
    # ------------------------------------------
    st.dataframe(feedback_df, use_container_width=True)

    st.markdown("---")
    st.subheader("✍️ Submit Feedback")

    # ------------------------------------------
    # 3️⃣ Load dataset
    # ------------------------------------------
    df = st.session_state.get("df", None)

    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    if "id" not in df.columns:
        st.error("❌ Dataset requires an 'id' column.")
        st.stop()

    # ------------------------------------------
    # 4️⃣ Input fields
    # ------------------------------------------
    record_id = st.number_input(
        "Enter Record ID",
        min_value=int(df["id"].min()),
        max_value=int(df["id"].max()),
        step=1,
    )

    user = st.session_state.get("username", "guest")
    feedback_type = st.selectbox("Feedback Type",
                                 ["Error", "Suggestion", "Improvement", "Other"])
    comment = st.text_area("Enter your feedback")

    # ------------------------------------------
    # 5️⃣ Submit button
    # ------------------------------------------
    if st.button("📨 Submit Feedback"):

        if not comment.strip():
            st.warning("⚠️ Comment cannot be empty.")
            st.stop()

        # 🔐 Create unique hash to prevent repeat-triggering
        import hashlib
        current_hash = hashlib.md5(
            f"{record_id}-{user}-{feedback_type}-{comment}".encode()
        ).hexdigest()

        # Prevent duplicate submission
        if st.session_state.last_feedback_hash == current_hash:
            st.info("ℹ️ Feedback already submitted. Not adding again.")
            st.stop()

        # Assign next feedback_id
        next_id = 1 if feedback_df.empty else int(feedback_df["feedback_id"].max()) + 1

        new_entry = {
            "feedback_id": next_id,
            "record_id": record_id,
            "user": user,
            "feedback_type": feedback_type,
            "comment": comment,
            "status": "Pending",
            "timestamp": pd.Timestamp.now()
        }

        # Save feedback
        st.session_state.feedback_df = pd.concat(
            [feedback_df, pd.DataFrame([new_entry])],
            ignore_index=True
        )

        # Save hash to avoid duplicate submission after rerun
        st.session_state.last_feedback_hash = current_hash

        st.success("✅ Feedback submitted successfully!")

        st.rerun()

# ----------------------------------------
# 📈 FEEDBACK ANALYSIS (ADMIN ONLY)
# ----------------------------------------
elif choice == "📈 Feedback Analysis":
    if st.session_state.user_role != "admin":
        st.error("❌ Access Denied: Admin privileges required")
        st.stop()

    st.header("📈 Feedback Analysis")
    if not feedback_df.empty:
        st.subheader("🧩 Feedback Type Distribution")
        st.bar_chart(feedback_df['feedback_type'].value_counts())
        st.subheader("🕒 Feedback Status Overview")
        st.bar_chart(feedback_df['status'].value_counts())
    else:
        st.info("No feedback available yet.")

# # ----------------------------------------
# 🛠 ADMIN TOOLS (ADMIN ONLY)
# ----------------------------------------
elif choice == "🛠 Admin Tools":
    # 1️⃣ Role check
    if st.session_state.user_role != "admin":
        st.error("❌ Access Denied: Admin privileges required")
        st.stop()

    st.header("🛠 Admin Tools")

    # 2️⃣ Dataset availability check
    df = st.session_state.get("df", None)
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset first.")
        st.stop()

    # 3️⃣ Required columns check
    missing_cols = [c for c in ["sentence", "id"] if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Dataset is missing required column(s): {', '.join(missing_cols)}")
        st.stop()

    # 4️⃣ Admin tools layout
    col1, col2 = st.columns(2)

    # --------------------------------------
    # 🔁 A) Merge Sentences (dedup / cleanup)
    # --------------------------------------
    with col1:
        st.subheader("🔁 Merge Sentences")

        all_sentences = sorted(df["sentence"].astype(str).unique().tolist())

        if not all_sentences:
            st.info("No sentences available to merge.")
        else:
            old_sentence = st.selectbox("Sentence to replace", all_sentences, key="merge_old")
            new_sentence = st.selectbox("Replace with", all_sentences, key="merge_new")

            if st.button("✅ Merge Sentences"):
                if old_sentence == new_sentence:
                    st.warning("⚠️ Old and new sentences are the same. Nothing to merge.")
                else:
                    # Only update the 'sentence' column
                    df["sentence"] = df["sentence"].replace({old_sentence: new_sentence})
                    st.session_state.df = df
                    st.success(f"✅ Merged all occurrences of:\n\n**{old_sentence}**\n\nto:\n\n**{new_sentence}**")

    # --------------------------------------
    # 🗑 B) Delete Record by ID
    # --------------------------------------
    with col2:
        st.subheader("🗑 Delete Record")

        unique_ids = sorted(df["id"].unique().tolist())
        if not unique_ids:
            st.info("No records available to delete.")
        else:
            record_id = st.selectbox("Select Record ID to delete", unique_ids, key="delete_id")

            # Show a preview of the record
            record_preview = df[df["id"] == record_id]
            st.markdown("**Record Preview:**")
            st.dataframe(record_preview, use_container_width=True)

            if st.button("🚨 Confirm Delete"):
                df = df[df["id"] != record_id].reset_index(drop=True)
                st.session_state.df = df
                st.success(f"✅ Deleted record with ID: {record_id}")
                st.rerun()

# ----------------------------------------
# 👥 USER MANAGEMENT (ADMIN ONLY)
# ----------------------------------------
elif choice == "👥 User Management":
    if st.session_state.user_role != "admin":
        st.error("❌ Access Denied: Admin privileges required")
        st.stop()
    
    st.header("👥 User Management")
    users = load_users()
    
    st.subheader("📋 Registered Users")
    if users:
        users_df = pd.DataFrame([
            {
                "Username": username,
                "Email": data.get("email", ""),
                "Role": data.get("role", ""),
                "Created At": data.get("created_at", ""),
                "Saved Graphs": len(data.get("saved_graphs", []))
            }
            for username, data in users.items()
        ])
        st.dataframe(users_df, use_container_width=True)
    else:
        st.info("No users found.")
    
    st.subheader("🔧 User Actions")
    if users:
        selected_user = st.selectbox("Select User", list(users.keys()))
        
        col1, col2 = st.columns(2)
        with col1:
            new_role = st.selectbox("Change Role", ["student", "researcher", "admin"])
            if st.button("Update Role"):
                users[selected_user]["role"] = new_role
                save_users(users)
                st.success(f"✅ Updated {selected_user}'s role to {new_role}")
        
        with col2:
            if st.button("🗑 Delete User"):
                if selected_user != st.session_state.username:
                    del users[selected_user]
                    save_users(users)
                    st.success(f"✅ Deleted user: {selected_user}")
                else:
                    st.error("❌ Cannot delete your own account")

## ----------------------------------------
# 💾 DOWNLOAD OPTIONS
# ----------------------------------------
elif choice == "💾 Download Options":
    st.header("💾 Download Data Files")

    # --- Load dataset safely ---
    df = st.session_state.get("df", None)
    if df is None:
        st.warning("⚠️ No dataset found. Please upload a dataset first.")
    else:
        st.subheader("📥 Dataset File")
        st.download_button(
            "⬇️ Download Dataset CSV",
            df.to_csv(index=False).encode("utf-8"),
            "dataset_updated.csv",
            mime="text/csv"
        )

    # --- Load feedback safely ---
    feedback_df = st.session_state.get("feedback_df", pd.DataFrame(
        columns=["record_id", "user", "feedback_type", "comment", "status", "timestamp"]
    ))

    st.subheader("📥 Feedback Records")
    st.download_button(
        "⬇️ Download Feedback CSV",
        feedback_df.to_csv(index=False).encode("utf-8"),
        "feedback.csv",
        mime="text/csv"
    )

    # --- Knowledge Graph Download ---
    st.subheader("📥 Knowledge Graph File")
    if os.path.exists(KNOWLEDGE_GRAPH_PATH):
        try:
            with open(KNOWLEDGE_GRAPH_PATH, "r", encoding="utf-8") as f:
                html_content = f.read()

            st.download_button(
                "⬇️ Download Knowledge Graph HTML",
                data=html_content,
                file_name="knowledge_graph.html",
                mime="text/html"
            )
        except Exception as e:
            st.error(f"❌ Unable to load Knowledge Graph file: {e}")
    else:
        st.info("⚠️ No Knowledge Graph has been generated yet.")


# ----------------------------------------
# ⚙️ USER PREFERENCES
# ----------------------------------------
elif choice == "⚙️ User Preferences":
    st.header("⚙️ User Preferences")

    username = st.session_state.get("username", "guest")
    st.write(f"Customize your experience, **{username}**!")

    # Ensure user_data exists
    if "user_data" not in st.session_state or st.session_state.user_data is None:
        st.session_state.user_data = {}

    # Ensure preferences dict exists with defaults
    default_prefs = {
        "theme": "Light",
        "results_per_page": 10,
        "show_tooltips": True,
        "accent_color": "Blue",
        "font_size": "Medium",
        "default_view": "🏠 Overview",
        "enable_notifications": True
    }

    current_prefs = st.session_state.user_data.get("preferences", default_prefs)

    # --------------------------------------
    # 🎨 THEME
    # --------------------------------------
    theme = st.selectbox(
        "Theme",
        ["Light", "Dark", "Auto"],
        index=["Light", "Dark", "Auto"].index(current_prefs.get("theme", "Light"))
    )

    # --------------------------------------
    # 📄 RESULTS PER PAGE
    # --------------------------------------
    results_per_page = st.slider(
        "Search Results Per Page",
        min_value=3,
        max_value=20,
        value=current_prefs.get("results_per_page", 10)
    )

    # --------------------------------------
    # 💬 SHOW TOOLTIPS
    # --------------------------------------
    show_tooltips = st.checkbox(
        "Show Tooltips",
        value=current_prefs.get("show_tooltips", True)
    )

    # --------------------------------------
    # 🌈 ACCENT COLOR
    # --------------------------------------
    accent_color = st.selectbox(
        "Accent Color",
        ["Blue", "Green", "Purple", "Orange", "Pink"],
        index=["Blue", "Green", "Purple", "Orange", "Pink"].index(
            current_prefs.get("accent_color", "Blue")
        )
    )

    # --------------------------------------
    # 🔤 FONT SIZE
    # --------------------------------------
    font_size = st.radio(
        "Font Size",
        ["Small", "Medium", "Large"],
        index=["Small", "Medium", "Large"].index(
            current_prefs.get("font_size", "Medium")
        )
    )

    # --------------------------------------
    # 📦 DEFAULT VIEW
    # --------------------------------------
    default_view = st.selectbox(
        "Default Page After Login",
        ["🏠 Overview", "🔍 Semantic Search", "🧠 Entity & Relation Extraction", "🌐 Knowledge Graph"],
        index=["🏠 Overview", "🔍 Semantic Search", "🧠 Entity & Relation Extraction", "🌐 Knowledge Graph"].index(
            current_prefs.get("default_view", "🏠 Overview")
        )
    )

    # --------------------------------------
    # 🔔 NOTIFICATION CONTROL
    # --------------------------------------
    enable_notifications = st.checkbox(
        "Enable Notifications (Success Messages, Toast Alerts)",
        value=current_prefs.get("enable_notifications", True)
    )

    st.markdown("---")

    # --------------------------------------
    # 💾 SAVE PREFERENCES BUTTON
    # --------------------------------------
    if st.button("💾 Save Preferences"):
        new_prefs = {
            "theme": theme,
            "results_per_page": results_per_page,
            "show_tooltips": show_tooltips,
            "accent_color": accent_color,
            "font_size": font_size,
            "default_view": default_view,
            "enable_notifications": enable_notifications
        }

        success = update_user_preferences(username, new_prefs)

        if success:
            st.session_state.user_data["preferences"] = new_prefs
            st.success("✅ Preferences saved successfully!")
        else:
            st.error("❌ Failed to save preferences. Please try again.")

    # --------------------------------------
    # 🧹 RESET BUTTON
    # --------------------------------------
    if st.button("🧹 Reset Preferences to Default"):
        st.session_state.user_data["preferences"] = default_prefs
        update_user_preferences(username, default_prefs)
        st.warning("♻️ Preferences reset to default.")
        st.rerun()

# Sidebar user display
st.sidebar.markdown("---")
st.sidebar.info(f"🔐 Logged in as: **{st.session_state.username}**")


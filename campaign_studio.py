import streamlit as st
import json
import time
import re
import pathlib
from io import BytesIO
from textwrap import dedent
from docx import Document
from docx.shared import Pt
import google.generativeai as genai
from openai import OpenAI

# ────────────────────────────────────────────────────────────
# 1. PAGE CONFIG & CSS
# ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Foolish Campaign Studio", page_icon="🃏", layout="wide")

st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; }
    .chat-bubble { padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .believer-bubble { background-color: #e3f6d8; border-left: 5px solid #43B02A; color: #1f1f1f; }
    .skeptic-bubble { background-color: #fce8e6; border-left: 5px solid #d93025; color: #1f1f1f; }
    h2, h3 { color: #1f1f1f; }
    .success-box { padding:15px; background:#e3f6d8; border-radius:8px; border:1px solid #43B02A; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# 2. CONFIGURATION & DATA LOADING
# ────────────────────────────────────────────────────────────

@st.cache_data
def load_configs():
    # Load Traits
    try:
        traits = json.loads(pathlib.Path("traits_config.json").read_text())
    except Exception:
        st.error("🚨 Missing 'traits_config.json'.")
        traits = {}

    # Load Personas
    try:
        # Handling the specific structure of the provided personas.json
        raw_p = json.loads(pathlib.Path("personas.json").read_text())
        flat_personas = []
        for group in raw_p.get("personas", []):
            segment = group.get("segment", "Unknown")
            for gender in ["male", "female"]:
                if gender in group:
                    p = group[gender]
                    # Add ID and Segment to the flat object
                    p["id"] = f"{p['name']} ({segment})"
                    p["segment"] = segment
                    flat_personas.append(p)
        return traits, flat_personas
    except Exception as e:
        st.error(f"🚨 Error loading 'personas.json': {e}")
        return traits, []

TRAIT_CFG, ALL_PERSONAS = load_configs()

# ────────────────────────────────────────────────────────────
# 3. AI ENGINE HANDLER (Unified)
# ────────────────────────────────────────────────────────────

def query_llm(messages, engine, temperature=0.7, json_mode=False):
    """Unified handler for OpenAI and Gemini."""
    
    # --- OPENAI ---
    if engine.startswith("OpenAI"):
        client = OpenAI(api_key=st.secrets.get("openai_api_key"))
        kwargs = {"model": "gpt-4-turbo", "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        try:
            resp = client.chat.completions.create(messages=messages, **kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            st.error(f"OpenAI Error: {e}")
            return None

    # --- GEMINI ---
    elif engine.startswith("Google"):
        api_key = st.secrets.get("google_api_key")
        if not api_key:
            st.error("Google API Key missing.")
            return None
        genai.configure(api_key=api_key)
        
        # Extract System Prompt
        sys_msg = next((m['content'] for m in messages if m['role'] == 'system'), "")
        history = [m['content'] for m in messages if m['role'] != 'system']
        prompt = "\n\n".join(history)

        config = genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain"
        )
        
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=sys_msg)
            resp = model.generate_content(prompt, generation_config=config)
            return resp.text
        except Exception as e:
            st.error(f"Gemini Error: {e}")
            return None
    return None

# ────────────────────────────────────────────────────────────
# 4. HELPER FUNCTIONS (Formatting & Logic)
# ────────────────────────────────────────────────────────────

def trait_rules_builder(scores):
    """Builds the constraint list based on slider scores."""
    rules = []
    for name, score in scores.items():
        cfg = TRAIT_CFG.get(name)
        if not cfg: continue
        if score >= cfg["high_threshold"]: rules.append(cfg["high_rule"])
        elif score <= cfg["low_threshold"]: rules.append(cfg["low_rule"])
        elif "mid_rule" in cfg: rules.append(cfg["mid_rule"])
    return rules

def create_docx(text):
    """Converts Markdown text to DOCX."""
    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    for line in text.split('\n'):
        if line.startswith('## '): doc.add_heading(line.replace('## ', ''), level=1)
        elif line.startswith('### '): doc.add_heading(line.replace('### ', ''), level=2)
        elif line.strip(): doc.add_paragraph(line.replace('**', '').replace('* ', ''))
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf

# ────────────────────────────────────────────────────────────
# 5. SESSION STATE INIT
# ────────────────────────────────────────────────────────────
defaults = {
    "current_draft": "",
    "internal_plan": "",
    "debate_transcript": [],
    "critique_summary": "",
    "optimized_copy": "",
    "active_tab": 0
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ────────────────────────────────────────────────────────────
# 6. SIDEBAR CONTROLS
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Campaign Settings")
    
    # A. Engine
    engine = st.radio("AI Engine", ["OpenAI (GPT-4)", "Google (Gemini 3)"], index=0)
    
    st.markdown("---")
    
    # B. Target Audience (Drives Tool 1 Tone & Tool 2 Personas)
    st.subheader("🎯 Target Audience")
    persona_names = [p["id"] for p in ALL_PERSONAS]
    target_persona_name = st.selectbox("Primary Avatar", persona_names, index=0 if persona_names else None)
    
    # Retrieve full persona object
    target_persona = next((p for p in ALL_PERSONAS if p["id"] == target_persona_name), None)

    st.markdown("---")

    # C. Tone Sliders (From Tool 1)
    with st.expander("🎚️ Voice & Tone Control", expanded=False):
        trait_scores = {
            "Urgency": st.slider("Urgency", 1, 10, 7),
            "Data_Richness": st.slider("Data / Stats", 1, 10, 6),
            "Social_Proof": st.slider("Social Proof", 1, 10, 6),
            "Comparative_Framing": st.slider("Metaphors", 1, 10, 5),
            "Conversational_Tone": st.slider("Conversational", 1, 10, 8),
            "FOMO": st.slider("FOMO Intensity", 1, 10, 5),
        }

# ────────────────────────────────────────────────────────────
# 7. MAIN INTERFACE
# ────────────────────────────────────────────────────────────
st.title("🃏 Foolish Campaign Studio")

tab_write, tab_test, tab_refine = st.tabs(["1️⃣ Write Draft", "2️⃣ Stress Test (Sim)", "3️⃣ Optimize"])

# ============================================================
# TAB 1: THE DRAFTER (Tool 1 Logic)
# ============================================================
with tab_write:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 The Brief")
        hook = st.text_area("Campaign Hook", "3 AI Stocks better than Nvidia. Urgent Buy Alert!", height=100)
        details = st.text_area("Product / Offer Details", "Subscription to Share Advisor. $199/year. Includes 'Top 10 Stocks' report.", height=150)
        
        c1, c2 = st.columns(2)
        length_mode = c1.selectbox("Length", ["Short (200w)", "Medium (500w)", "Long (1000w)"])
        copy_type = c2.selectbox("Format", ["📧 Email", "📝 Sales Page"])
        
        if st.button("✨ Generate First Draft", type="primary"):
            if not hook or not details:
                st.warning("Please enter a Hook and Details.")
            else:
                # 1. Build Prompt
                rules = trait_rules_builder(trait_scores)
                target_desc = f"{target_persona['name']}, {target_persona['age']}yo {target_persona['occupation']} ({target_persona['segment']})" if target_persona else "General Investor"
                
                sys_prompt = dedent(f"""
                You are a Senior Direct Response Copywriter. 
                TARGET AUDIENCE: {target_desc}.
                TONE RULES:
                {chr(10).join(rules)}
                
                STRUCTURE:
                - Use Markdown headings.
                - End with disclaimer: *Past performance is not a reliable indicator of future results.*
                """)
                
                user_prompt = dedent(f"""
                Write a {copy_type} ({length_mode}).
                HOOK: {hook}
                DETAILS: {details}
                
                OUTPUT JSON: {{ "plan": "bullet points", "copy": "the actual text" }}
                """)
                
                with st.spinner("Drafting..."):
                    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
                    res = query_llm(msgs, engine, json_mode=True)
                    
                    if res:
                        try:
                            # Clean markdown formatting if model adds it
                            clean_res = res.replace("```json", "").replace("```", "")
                            data = json.loads(clean_res)
                            st.session_state.current_draft = data["copy"]
                            st.session_state.internal_plan = data["plan"]
                            st.session_state.debate_transcript = [] # Reset subsequent steps
                            st.session_state.critique_summary = ""
                            st.rerun()
                        except:
                            st.error("Error parsing AI response. Try again.")

    with col2:
        st.subheader("📝 Draft Output")
        if st.session_state.current_draft:
            with st.expander("Show AI Plan", expanded=False):
                st.write(st.session_state.internal_plan)
            st.markdown(st.session_state.current_draft)
        else:
            st.info("👈 Enter brief and generate to see copy here.")

# ============================================================
# TAB 2: THE SIMULATOR (Tool 2 Logic)
# ============================================================
with tab_test:
    st.header("🧪 Audience Stress Test")
    
    if not st.session_state.current_draft:
        st.warning("⚠️ You need to generate a draft in Tab 1 first.")
    else:
        # Setup Participants
        c_a, c_b = st.columns(2)
        with c_a:
            # Default to the target persona from sidebar
            idx_1 = persona_names.index(target_persona_name) if target_persona_name in persona_names else 0
            p1_name = st.selectbox("Participant 1 (The Believer)", persona_names, index=idx_1, key="sim_p1")
        with c_b:
            # Default to someone else
            p2_name = st.selectbox("Participant 2 (The Skeptic)", persona_names, index=(idx_1 + 1) % len(persona_names), key="sim_p2")
            
        if st.button("▶️ Run Focus Group Simulation"):
            p1 = next(p for p in ALL_PERSONAS if p["id"] == p1_name)
            p2 = next(p for p in ALL_PERSONAS if p["id"] == p2_name)
            
            draft_text = st.session_state.current_draft
            transcript = []
            
            # --- TURN 1: The Believer ---
            prompt_base = f"You are participating in a marketing focus group. React to the text below."
            
            role_1 = f"You are {p1['name']}, {p1['age']}. Bio: {p1['narrative']}. Values: {p1['values']}. You are OPTIMISTIC and looking for opportunity."
            msg_1_prompt = f"{prompt_base}\nTEXT TO REVIEW:\n'{draft_text}'\n\nWhat excites you? What stands out?"
            
            with st.status("Simulating focus group...") as status:
                status.write(f"🎤 {p1['name']} is reading...")
                res_1 = query_llm([{"role":"system", "content": role_1}, {"role":"user", "content": msg_1_prompt}], engine)
                transcript.append({"name": p1["name"], "role": "Believer", "text": res_1})
                
                # --- TURN 2: The Skeptic ---
                status.write(f"🤔 {p2['name']} is critiquing...")
                role_2 = f"You are {p2['name']}, {p2['age']}. Bio: {p2['narrative']}. Values: {p2['values']}. You are SKEPTICAL and risk-averse."
                msg_2_prompt = f"{prompt_base}\nTEXT TO REVIEW:\n'{draft_text}'\n\n{p1['name']} just said: '{res_1}'.\nTell them why they are wrong. Find the 'catch' or the risk in the copy."
                
                res_2 = query_llm([{"role":"system", "content": role_2}, {"role":"user", "content": msg_2_prompt}], engine)
                transcript.append({"name": p2["name"], "role": "Skeptic", "text": res_2})
                
                # --- TURN 3: The Believer Rebuttal ---
                status.write(f"🗣️ {p1['name']} is responding...")
                msg_3_prompt = f"You just heard {p2['name']} say: '{res_2}'. Do you agree with their worry, or do you still want to buy?"
                
                res_3 = query_llm([{"role":"system", "content": role_1}, {"role":"user", "content": msg_3_prompt}], engine)
                transcript.append({"name": p1["name"], "role": "Believer", "text": res_3})
                
                st.session_state.debate_transcript = transcript
                status.update(label="Focus Group Complete!", state="complete", expanded=False)

        # Display Transcript
        if st.session_state.debate_transcript:
            st.divider()
            for turn in st.session_state.debate_transcript:
                style_class = "believer-bubble" if turn["role"] == "Believer" else "skeptic-bubble"
                st.markdown(f"""
                <div class="chat-bubble {style_class}">
                    <strong>{turn['name']} ({turn['role']}):</strong><br>{turn['text']}
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB 3: THE OPTIMIZER (Bridging Logic)
# ============================================================
with tab_refine:
    st.header("🚀 Optimization Station")
    
    if not st.session_state.debate_transcript:
        st.info("Run the Focus Group in Tab 2 to unlock optimization.")
    else:
        col_L, col_R = st.columns([1, 1])
        
        with col_L:
            st.subheader("📊 Strategic Analysis")
            if st.button("🧠 Analyze & Plan Fixes"):
                transcript_text = "\n".join([f"{t['role']}: {t['text']}" for t in st.session_state.debate_transcript])
                
                analysis_prompt = dedent(f"""
                You are a Marketing Strategist. Analyze this debate transcript regarding a piece of copy.
                TRANSCRIPT:
                {transcript_text}
                
                Identify:
                1. The main "Trust Gap" (what made the Skeptic doubt).
                2. The main "Hook Strength" (what the Believer liked).
                3. Three specific actionable edits to improve the copy.
                """)
                
                with st.spinner("Analyzing psychology..."):
                    st.session_state.critique_summary = query_llm([{"role":"user", "content": analysis_prompt}], engine)
            
            if st.session_state.critique_summary:
                st.success(st.session_state.critique_summary)

        with col_R:
            st.subheader("✨ Final Polish")
            if st.session_state.critique_summary:
                if st.button("✍️ Apply Fixes & Rewrite"):
                    rewrite_prompt = dedent(f"""
                    You are an Expert Editor.
                    ORIGINAL COPY:
                    {st.session_state.current_draft}
                    
                    CRITIQUE TO ADDRESS:
                    {st.session_state.critique_summary}
                    
                    TASK:
                    Rewrite the copy to address the critique while maintaining the original high-energy tone.
                    Strengthen the proof points where the Skeptic was doubtful.
                    Output ONLY the final copy.
                    """)
                    
                    with st.spinner("Polishing final draft..."):
                        st.session_state.optimized_copy = query_llm([{"role":"user", "content": rewrite_prompt}], engine)
            
            if st.session_state.optimized_copy:
                st.markdown(st.session_state.optimized_copy)
                
                # DOCX Export
                docx = create_docx(st.session_state.optimized_copy)
                st.download_button("📥 Download Final DOCX", docx, "final_campaign.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

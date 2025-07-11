# app.py - AgroSense-AI with Inline Vapi Chatbot (Persistent Sliders)
import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="🌱 AgroSense-AI: The Smart Crop Compatibility Engine",
    page_icon="🌾"
)

# --- CSS for Fixed Chat Input ---
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
        background: white;
        z-index: 999;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .main .block-container {
        padding-bottom: 100px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Model Training ---
@st.cache_resource
def load_data_and_train():
    df = pd.read_csv("Crop_Recommendation.csv")
    df = df.rename(columns={
        "Nitrogen": "N",
        "Phosphorus": "P",
        "Potassium": "K",
        "pH_Value": "pH",
    })

    # Compute global min/max for environmental sliders
    global_env_ranges = {
        "Temperature": (df["Temperature"].min(), df["Temperature"].max()),
        "Humidity":    (df["Humidity"].min(),    df["Humidity"].max()),
        "pH":          (df["pH"].min(),          df["pH"].max()),
        "Soil_Moisture": (df["Soil_Moisture"].min(), df["Soil_Moisture"].max())
    }

    le = LabelEncoder()
    df["Crop_encoded"] = le.fit_transform(df["Crop"])

    features = ["N", "P", "K", "Temperature", "Humidity", "pH", "Soil_Moisture"]
    model = RandomForestClassifier()
    model.fit(df[features], df["Crop_encoded"])

    crop_stats = {}
    for crop in le.classes_:
        subset = df[df["Crop_encoded"] == le.transform([crop])[0]]
        crop_stats[crop] = {
            "N_avg": subset["N"].mean(),
            "P_avg": subset["P"].mean(),
            "K_avg": subset["K"].mean(),
            "N_range": (subset["N"].min(), subset["N"].max()),
            "P_range": (subset["P"].min(), subset["P"].max()),
            "K_range": (subset["K"].min(), subset["K"].max()),
            "Temperature": (subset["Temperature"].min(), subset["Temperature"].max()),
            "Humidity":    (subset["Humidity"].min(),    subset["Humidity"].max()),
            "pH":          (subset["pH"].min(),          subset["pH"].max()),
            "Soil_Moisture": (subset["Soil_Moisture"].min(), subset["Soil_Moisture"].max())
        }

    return model, le, crop_stats, global_env_ranges

model, le, crop_stats, global_env_ranges = load_data_and_train()
all_crops = sorted(le.classes_)

# --- Initialize persistent sliders in session_state ---
# (Only on first run; afterwards Streamlit will remember user adjustments)
if "temp" not in st.session_state:
    lo, hi = global_env_ranges["Temperature"]
    st.session_state.temp = (lo + hi) / 2
if "humidity" not in st.session_state:
    lo, hi = global_env_ranges["Humidity"]
    st.session_state.humidity = (lo + hi) / 2
if "ph" not in st.session_state:
    lo, hi = global_env_ranges["pH"]
    st.session_state.ph = (lo + hi) / 2
if "soil_moisture" not in st.session_state:
    lo, hi = global_env_ranges["Soil_Moisture"]
    st.session_state.soil_moisture = (lo + hi) / 2

# --- Main UI ---
st.title("🌾 AgroSense-AI: The Smart Crop Compatibility Engine")
st.markdown("""
*Customize soil nutrient values (N‑P‑K) and environmental factors*  
*Get personalized crop recommendations based on your inputs*
""")

with st.sidebar:
    st.header("🌿 Crop Selection")
    selected_crop = st.selectbox("Choose your crop", all_crops)

    st.header("🧪 Soil Nutrients (ppm)")
    n_value = st.slider(
        "Nitrogen (N)", 0, 300,
        int(round(crop_stats[selected_crop]["N_avg"])),
        step=1, key="n_value"
    )
    p_value = st.slider(
        "Phosphorus (P)", 0, 300,
        int(round(crop_stats[selected_crop]["P_avg"])),
        step=1, key="p_value"
    )
    k_value = st.slider(
        "Potassium (K)", 0, 300,
        int(round(crop_stats[selected_crop]["K_avg"])),
        step=1, key="k_value"
    )

    st.header("🌦️ Environmental Factors")
    temp = st.slider(
        "Temperature (°C)",
        global_env_ranges["Temperature"][0],
        global_env_ranges["Temperature"][1],
        st.session_state.temp,
        step=0.1, key="temp"
    )
    humidity = st.slider(
        "Humidity (%)",
        global_env_ranges["Humidity"][0],
        global_env_ranges["Humidity"][1],
        st.session_state.humidity,
        step=0.1, key="humidity"
    )
    ph = st.slider(
        "Soil pH Level",
        global_env_ranges["pH"][0],
        global_env_ranges["pH"][1],
        st.session_state.ph,
        step=0.1, key="ph"
    )
    soil_moisture = st.slider(
        "Soil Moisture (%)",
        global_env_ranges["Soil_Moisture"][0],
        global_env_ranges["Soil_Moisture"][1],
        st.session_state.soil_moisture,
        step=0.1, key="soil_moisture"
    )

# --- Analysis (unchanged) ---
if st.button("🧑‍🌾 Analyze Growing Conditions"):
    st.header("🔍 Analysis Results")
    analysis_data = {
        "N": n_value, "P": p_value, "K": k_value,
        "Temperature": temp, "Humidity": humidity,
        "pH": ph, "Soil_Moisture": soil_moisture
    }

    req = crop_stats[selected_crop]
    violations = []
    for param, val in analysis_data.items():
        low, high = req[f"{param}_range"] if param in ["N","P","K"] else req[param]
        if not (low <= val <= high):
            violations.append(f"{param}: {val} (requires {low}-{high})")

    if not violations:
        st.success(f"✅ Excellent conditions for {selected_crop}!")
        st.balloons()
    else:
        st.error(f"⚠️ Potential challenges for {selected_crop}:")
        for v in violations:
            st.write(f"- {v}")
        st.warning("Consider adjusting inputs or choosing another crop.")

    st.subheader("🌱 Alternative Suitable Crops")
    suitable = [
        crop for crop in all_crops
        if all(
            (crop_stats[crop][f"{p}_range"][0] <= analysis_data[p] <= crop_stats[crop][f"{p}_range"][1])
            if p in ["N","P","K"]
            else (crop_stats[crop][p][0] <= analysis_data[p] <= crop_stats[crop][p][1])
            for p in analysis_data
        )
    ]

    if suitable:
        cols = st.columns(len(suitable))
        for col, crop in zip(cols, suitable):
            with col:
                stats = crop_stats[crop]
                st.metric(crop, f"N–P–K: {stats['N_avg']:.0f}-{stats['P_avg']:.0f}-{stats['K_avg']:.0f}")
    else:
        st.warning("No other crops perfectly match these conditions.")

# --- Requirement Table (unchanged) ---
with st.expander("📊 Crop Requirement Table"):
    df_table = pd.DataFrame([
        {
            "Crop": c,
            "N Range": f"{s['N_range'][0]:.1f}-{s['N_range'][1]:.1f}",
            "P Range": f"{s['P_range'][0]:.1f}-{s['P_range'][1]:.1f}",
            "K Range": f"{s['K_range'][0]:.1f}-{s['K_range'][1]:.1f}",
            "Temp (°C)": f"{s['Temperature'][0]:.1f}-{s['Temperature'][1]:.1f}",
            "Humidity (%)": f"{s['Humidity'][0]:.1f}-{s['Humidity'][1]:.1f}",
            "pH": f"{s['pH'][0]:.1f}-{s['pH'][1]:.1f}",
            "Moisture (%)": f"{s['Soil_Moisture'][0]:.1f}-{s['Soil_Moisture'][1]:.1f}"
        }
        for c,s in crop_stats.items()
    ])
    st.dataframe(df_table, height=400, use_container_width=True)

# --- Inline Vapi Chatbot (unchanged) ---
with st.expander("💬 Ask AgroSenseBot", expanded=True):
    st.markdown("Ask me anything about crops, soil, or growing conditions.")
    try:
        VAPI_PRIVATE_KEY = st.secrets["VAPI_PRIVATE_KEY"]
        ASSISTANT_ID = st.secrets["ASSISTANT_ID"]
    except KeyError:
        st.error("⚠️ Missing VAPI credentials in secrets.toml")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_id = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about crops, soil, or growing conditions...")
    if user_input:
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        payload = {"assistantId":ASSISTANT_ID,"input":user_input}
        if st.session_state.chat_id:
            payload["previousChatId"]=st.session_state.chat_id

        resp = requests.post(
            "https://api.vapi.ai/chat",
            json=payload,
            headers={
                "Authorization": f"Bearer {VAPI_PRIVATE_KEY}",
                "Content-Type": "application/json"
            }
        )
        bot_reply = (resp.json().get("output", [{}])[-1].get("content")
                     if resp.ok else f"Error {resp.status_code}: {resp.text}")

        st.session_state.chat_id = resp.json().get("id", st.session_state.chat_id)
        st.session_state.messages.append({"role":"assistant","content":bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

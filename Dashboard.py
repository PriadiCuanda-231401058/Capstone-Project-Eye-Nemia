import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==============================================================================
st.set_page_config(
    page_title="Eye-Nemia Final Analytics Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PERBAIKAN VISIBILITAS: Menambahkan color:#111111 (Hitam gelap) agar teks 
# selalu terlihat jelas meskipun Streamlit berada dalam Dark Mode.
st.markdown("""
    <style>
    .main-title { font-size:2.4rem; font-weight:700; color:#bc1b21; margin-bottom:0.2rem; }
    .sub-title { font-size:1.1rem; margin-bottom:1.5rem; } 
    .metric-box { 
        background-color:#f8f9fa; 
        color:#111111; 
        border-radius:8px; 
        padding:15px; 
        border-left:5px solid #bc1b21; 
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🩸 Eye-Nemia: Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem monitoring komprehensif risiko anemia menggunakan basis data terintegrasi hasil optimalisasi Feature Engineering.</div>', unsafe_allow_html=True)

# ==============================================================================
# 2. PROSES PEMUATAN DATA (DATA LOAD CACHING)
# ==============================================================================
@st.cache_data
def load_engineered_data():
    try:
        df = pd.read_csv("eye_nemia_feature_engineered.csv")
        return df
    except FileNotFoundError:
        st.error("Gagal Memuat Data: File 'eye_nemia_feature_engineered.csv' tidak ditemukan di direktori saat ini.")
        st.stop()

df_fe = load_engineered_data()

# ==============================================================================
# 3. SIDEBAR KONTROL DAN FILTERING DATA SECARA GLOBAL
# ==============================================================================
# st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2864/2864438.png", width=90)
st.sidebar.header("🎛️ Panel Kontrol Dataset")

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Populasi Pasien")

# Filter Rentang Usia
age_min, age_max = int(df_fe['Age'].min()), int(df_fe['Age'].max())
selected_age = st.sidebar.slider("Rentang Usia Target (Tahun)", age_min, age_max, (age_min, age_max))

# Filter Multiselect Status Kehamilan
preg_options = {1.0: "Ya (1.0)", 0.0: "Tidak (0.0)"}
selected_preg_vals = st.sidebar.multiselect(
    "Status Sedang Hamil:", 
    options=list(preg_options.keys()), 
    default=list(preg_options.keys()),
    format_func=lambda x: preg_options[x]
)

# Filter Multiselect Riwayat Anemia
anemia_options = {1.0: "Ya (1.0)", 0.0: "Tidak (0.0)"}
selected_anemia_vals = st.sidebar.multiselect(
    "Memiliki Riwayat Anemia:", 
    options=list(anemia_options.keys()), 
    default=list(anemia_options.keys()),
    format_func=lambda x: anemia_options[x]
)

# Eksekusi Pemotongan Filter Terhadap Dataset Utama
df_filtered = df_fe[
    (df_fe['Age'] >= selected_age[0]) & 
    (df_fe['Age'] <= selected_age[1]) &
    (df_fe['Currently_Pregnant'].isin(selected_preg_vals)) &
    (df_fe['History_Anemia'].isin(selected_anemia_vals))
]
numeric_filtered = df_filtered.select_dtypes(include=['int64', 'float64'])

# ==============================================================================
# 4. RINGKASAN METRIK KUNCI (EXECUTIVE KPI SECTION)
# ==============================================================================
st.markdown("### 📊 Ringkasan Eksekutif Populasi Pasien")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"<div class='metric-box'><b>Total Pasien</b><br><span style='font-size:1.8rem; font-weight:700;'>{len(df_filtered)}</span> / {len(df_fe)}</div>", unsafe_allow_html=True)
with kpi2:
    mean_risk = df_filtered['Risk_Score'].mean() if len(df_filtered) > 0 else 0
    st.markdown(f"<div class='metric-box'><b>Rerata Risk Score</b><br><span style='font-size:1.8rem; font-weight:700; color:#bc1b21;'>{mean_risk:.2f}</span></div>", unsafe_allow_html=True)
with kpi3:
    mean_bmi = df_filtered['BMI'].mean() if 'BMI' in df_filtered.columns and len(df_filtered) > 0 else 0
    st.markdown(f"<div class='metric-box'><b>Rerata BMI Pasien</b><br><span style='font-size:1.8rem; font-weight:700;'>{mean_bmi:.2f}</span></div>", unsafe_allow_html=True)
with kpi4:
    high_risk_count = len(df_filtered[df_filtered['Risk_Score'] >= 70]) if len(df_filtered) > 0 else 0
    st.markdown(f"<div class='metric-box'><b>Kasus Risiko Tinggi</b><br><span style='font-size:1.8rem; font-weight:700;'>{high_risk_count}</span></div>", unsafe_allow_html=True)
with kpi5:
    preg_count = len(df_filtered[df_filtered['Currently_Pregnant'] == 1.0]) if len(df_filtered) > 0 else 0
    st.markdown(f"<div class='metric-box'><b>Total Ibu Hamil</b><br><span style='font-size:1.8rem; font-weight:700;'>{preg_count}</span></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# 5. STRUKTUR NAVIGASI TAB INTERAKTIF DAN RESPONSIVE
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Pertanyaan Bisnis (Colab Match)", 
    "🔬 Analisis Transparansi Feature Engineering", 
    "📈 Eksplorasi Distribusi & Tren Medis",
    "🗂️ Matriks Korelasi & Unduhan Data"
])

# ------------------------------------------------------------------------------
# TAB 1: KONSISTENSI TOTAL DENGAN LOGIKA PERTANYAAN BISNIS GOOGLE COLAB
# ------------------------------------------------------------------------------
with tab1:
    st.header("🎯 Analisis Data-Driven Menjawab Pertanyaan Bisnis SMART")
    st.caption("Bagian ini menggunakan pemrosesan korelasi murni yang disesuaikan secara matematis dengan skrip Google Colab.")
    
    if len(numeric_filtered) > 1:
        corr_all = numeric_filtered.corr()['Risk_Score'].drop('Risk_Score').fillna(0)
        
        # PERTANYAAN BISNIS 1
        st.subheader("1. Top 5 Prediktor Terkuat Pemicu Risiko Anemia")
        top_5_all = corr_all.sort_values(ascending=False).head(5)
        
        fig_q1 = px.bar(
            x=top_5_all.values, y=top_5_all.index, orientation='h', text_auto='.2f',
            labels={'x': 'Nilai Korelasi (Pearson)', 'y': 'Fitur Prediktor'},
            color=top_5_all.values, color_continuous_scale='Reds'
        )
        fig_q1.update_traces(textposition='outside')
        fig_q1.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=350)
        st.plotly_chart(fig_q1, use_container_width=True)
        
        # PERTANYAAN BISNIS 2
        st.subheader("2. Top 3 Faktor Pola Makan & Defisit Nutrisi Pemicu Risiko Tinggi")
        diet_keywords = ['diet', 'iron', 'vitamin', 'meat', 'tea', 'coffee', 'consume']
        diet_cols = [col for col in numeric_filtered.columns if any(kw in col.lower() for kw in diet_keywords)]
        
        if diet_cols:
            corr_diet = numeric_filtered[diet_cols + ['Risk_Score']].corr()['Risk_Score'].drop('Risk_Score').fillna(0)
            top_3_diet = corr_diet.abs().sort_values(ascending=False).head(3)
            top_3_diet_actual = corr_diet.loc[top_3_diet.index].sort_values(ascending=False)
            
            fig_q2 = px.bar(
                x=top_3_diet_actual.values, y=top_3_diet_actual.index, orientation='h', text_auto='.2f',
                labels={'x': 'Nilai Korelasi (+ Pemicu, - Menurunkan Risiko)', 'y': 'Fitur Pola Makan'},
                color=top_3_diet_actual.values, color_continuous_scale='Oranges'
            )
            fig_q2.update_traces(textposition='outside')
            fig_q2.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=280)
            st.plotly_chart(fig_q2, use_container_width=True)
            
        # PERTANYAAN BISNIS 3
        st.subheader("3. Faktor Dominan Risiko Anemia Khusus Populasi Ibu Hamil")
        if 'Currently_Pregnant' in numeric_filtered.columns:
            df_pregnant = numeric_filtered[numeric_filtered['Currently_Pregnant'] == 1.0]
            
            if len(df_pregnant) > 1:
                cols_to_drop = ['Risk_Score', 'Currently_Pregnant', 'Menstruation_Days', 'Menstruation_Risk']
                corr_preg = df_pregnant.corr()['Risk_Score'].drop(cols_to_drop, errors='ignore').fillna(0)
                top_5_preg = corr_preg.sort_values(ascending=False).head(5)
                
                fig_q3 = px.bar(
                    x=top_5_preg.values, y=top_5_preg.index, orientation='h', text_auto='.2f',
                    labels={'x': 'Nilai Korelasi (Pearson)', 'y': 'Fitur Spasial Kehamilan'},
                    color=top_5_preg.values, color_continuous_scale='Purples'
                )
                fig_q3.update_traces(textposition='outside')
                fig_q3.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=350)
                st.plotly_chart(fig_q3, use_container_width=True)
            else:
                st.warning("Jumlah representasi sampel Ibu Hamil tidak mencukupi untuk memetakan kalkulasi matriks korelasi.")
    else:
        st.error("Data aktif hasil penyaringan terlalu sedikit untuk dikalkulasikan.")

# ------------------------------------------------------------------------------
# TAB 2: DETEKSI TRANSPARANSI FITUR ENGINEERED (BMI, POOR_DIET, DLL)
# ------------------------------------------------------------------------------
with tab2:
    st.header("🔬 Eksplorasi Kontribusi Komponen Feature Engineering")
    
    engineered_options = [c for c in ['BMI', 'Iron_Rich_Food_Score', 'Menstruation_Risk', 'Poor_Diet_Score', 'Iron_Inhibitor_Score'] if c in df_filtered.columns]
    
    if engineered_options:
        target_eng_feature = st.selectbox("Pilih Parameter Fitur Turunan:", engineered_options)
        
        col_eng_a, col_eng_b = st.columns(2)
        
        with col_eng_a:
            st.markdown(f"**Distribusi Sebaran Linear `{target_eng_feature}` vs Risk Score**")
            fig_scat = px.scatter(
                df_filtered, x=target_eng_feature, y="Risk_Score", 
                color="Currently_Pregnant", trendline="ols",
                color_continuous_scale="Viridis",
                labels={"Currently_Pregnant": "Status Hamil"}
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            
        with col_eng_b:
            st.markdown(f"**Segmentasi Sebaran Ragam `{target_eng_feature}` Berdasarkan Riwayat Anemia**")
            fig_box_eng = px.box(
                df_filtered, x="History_Anemia", y=target_eng_feature, 
                color="History_Anemia", notched=True
            )
            st.plotly_chart(fig_box_eng, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: ANALISIS DISTRIBUSI & TREN MULTIVARIAT (EDA MAJU)
# ------------------------------------------------------------------------------
with tab3:
    st.header("📈 Analisis Deskriptif dan Visualisasi Tren Klinis")
    
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        st.markdown("**Analisis Kepadatan Rentang Risk Score Populasi Aktif**")
        fig_density = px.histogram(
            df_filtered, x="Risk_Score", color="Currently_Pregnant", 
            marginal="rug", barmode="overlay", nbins=40
        )
        st.plotly_chart(fig_density, use_container_width=True)
        
    with row3_col2:
        st.markdown("**Perbandingan Komparatif Risk Score Terhadap Gejala Klinis Kelelahan (Fatigue)**")
        if 'Symptom_Fatigue' in df_filtered.columns:
            fig_symptom = px.violin(
                df_filtered, x="Symptom_Fatigue", y="Risk_Score", 
                color="Symptom_Fatigue", box=True, points="all"
            )
            st.plotly_chart(fig_symptom, use_container_width=True)
            
    st.markdown("---")
    st.markdown("### 🗺️ Pemetaan Klaster Pasien 3D Scatter (Deteksi Anomali Cepat)")
    
    if 'BMI' in df_filtered.columns:
        fig_3d_final = px.scatter_3d(
            df_filtered, x='Age', y='BMI', z='Risk_Score',
            color='Risk_Score', size='Risk_Score', size_max=12,
            color_continuous_scale="inferno", opacity=0.8, # DI SINI DIPERBAIKI MENJADI 'inferno'
            height=600
        )
        st.plotly_chart(fig_3d_final, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: GLOBAL MATRIX CORRELATION HEATMAP & DATA EXPORT CENTER
# ------------------------------------------------------------------------------
with tab4:
    st.header("🗂️ Pusat Manajemen Matriks Korelasi Global & Unduhan Data")
    
    if len(numeric_filtered) > 1:
        st.subheader("Matriks Heatmap Relasi Linear Pearson Teratas")
        
        global_corr = numeric_filtered.corr()['Risk_Score'].drop('Risk_Score').abs().sort_values(ascending=False)
        top_15_features = global_corr.head(15).index.tolist()
        top_15_features.append('Risk_Score')
        
        heatmap_data = numeric_filtered[top_15_features].corr()
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.columns,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=heatmap_data.values.round(2), texttemplate="%{text}",
            textfont={"color": "white"} 
        ))
        fig_heat.update_layout(height=650)
        st.plotly_chart(fig_heat, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Data Explorer Center")
    st.dataframe(df_filtered, use_container_width=True)
    
    export_csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Ekspor Hasil Filter Pasien ke CSV",
        data=export_csv_data,
        file_name='eye_nemia_filtered_output.csv',
        mime='text/csv'
    )
"""
app.py — GEOPredict (Layout Premium + CSS Protegido + Diagnóstico de Dados)
"""

import io
import os
import base64
import logging
import warnings
import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Config e esquema
from config import (
    InstrumentType, INSTRUMENT_FREQ_HOURS, INSTRUMENT_FREQ_LABEL,
    FORECAST_HORIZONS_PRODUCAO, FORECAST_HORIZONS_EXPERIMENTAL,
    FORECAST_DEFAULT_LABEL, HORIZON_RELIABILITY_MSGS,
    DEFAULT_MODEL, DEFAULT_MISSING, DEFAULT_TUNING, DEFAULT_CV,
    ModelConfig, TuningConfig, CVConfig, MissingValueConfig,
    get_feature_config_for_instrument, detect_series_frequency,
)

# Base de permeabilidade externa
from permeability_db import (
    load_permeability_db, lookup_permeability,
    get_soil_options, get_db_status,
)

# Módulos do pipeline
from schema import validate_schema, validate_business_requirements, check_external_data_coverage
from data_loader import (
    load_main_data, load_external_data,
    merge_external_data, inject_instrument_constants,
)
from preprocessing import run_preprocessing
from quality import treat_missing_values, detect_outliers, apply_outlier_filter
from features import build_features, get_feature_columns
from training import run_training
from forecasting import recursive_forecast, horizon_days_to_steps
from evaluation import (
    metrics_per_instrument, compute_historical_envelope,
    check_forecast_vs_envelope, check_physical_plausibility,
    compute_shap_values, compute_shap_for_instrument,
    compute_forecast_uncertainty,
    format_metrics_table, format_cv_results, instrument_summary,
)
from visualization import (
    plot_forecast_final, plot_full_series, plot_obs_vs_pred,
    plot_scatter, plot_residuals, plot_residual_distribution,
    plot_feature_importance, plot_metrics_per_instrument, plot_outliers,
    plot_shap_summary, plot_shap_local, plot_cv_results,
)
from sample_data import generate_sample_data

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GEOPredict",
    layout="wide", 
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Função para Forçar Tema Translúcido nos Gráficos Plotly
# ---------------------------------------------------------------------------
def apply_dark_theme(fig):
    if hasattr(fig, 'update_layout'):
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0.15)", 
            font_color="#cbd5e1", 
            title_font_color="#f8fafc", 
            legend_font_color="#cbd5e1",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.2)", tickfont=dict(color="#cbd5e1"))
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.2)", tickfont=dict(color="#cbd5e1"))
    return fig

# ---------------------------------------------------------------------------
# Função Avançada para Injetar Imagem de Fundo + Glassmorphism CSS Seguro
# ---------------------------------------------------------------------------
def set_background_and_glassmorphism(image_filename):
    css_base = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* 1. Fonte base protegida - REMOVIDA A TAG SPAN PARA SALVAR OS ÍCONES */
p, label, h1, h2, h3, h4, h5, h6, li, div[data-testid="stMarkdownContainer"] { 
    font-family: 'Inter', sans-serif !important; 
    font-size: 1.02rem !important; 
    color: #f8fafc !important; 
}
h1, h2, h3, h4, h5, h6 { font-weight: 700 !important; }

/* Força os checkboxes e radio buttons a ficarem brancos */
div[data-testid="stCheckbox"] label, div[data-testid="stRadio"] label {
    color: #f8fafc !important;
}

/* 2. Reset Global Seguro */
html, body { margin: 0; padding: 0; background-color: #0f172a !important; }
[data-testid="stHeader"] { display: none !important; }

/* 3. O Container Principal (Floating Card Premium) */
[data-testid="stMainBlockContainer"] {
    background: rgba(15, 23, 42, 0.78) !important; 
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6) !important;
    
    padding: 3rem 4rem !important;
    max-width: 1400px !important;
    margin-top: 4vh !important;
    margin-bottom: 4vh !important;
}

/* 4. Cabeçalhos Refinados */
h4 {
    font-size: 1.2rem !important; 
    font-weight: 600 !important;
    margin-top: 1rem !important; 
    margin-bottom: 0.8rem !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.15) !important; 
    padding-bottom: 0.4rem !important;
}

/* 5. Caixas de KPIs (Resultados) */
[data-testid="stMetric"] {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}
[data-testid="stMetricValue"] > div { color: #3b82f6 !important; font-weight: 700 !important; font-size: 2.2rem !important; }
[data-testid="stMetricLabel"] > div { color: #94a3b8 !important; font-size: 1rem !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }

/* 6. Inputs e Caixas de Upload */
.stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>div {
    background-color: rgba(0, 0, 0, 0.3) !important; 
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 8px !important; 
    color: #f8fafc !important;
}

[data-testid="stFileUploadDropzone"] {
    background-color: rgba(0, 0, 0, 0.3) !important; 
    border: 1px dashed rgba(255, 255, 255, 0.3) !important;
    border-radius: 12px !important; 
    padding: 1rem !important; 
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: #3b82f6 !important; background-color: rgba(0, 0, 0, 0.5) !important; }

/* 7. Botão Primário Moderno */
button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important; border: none !important;
    border-radius: 8px !important; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    color: white !important; font-weight: 600 !important; font-size: 1.1rem !important; padding: 0.8rem !important; transition: all 0.2s ease !important;
}
button[kind="primary"]:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important; }

/* Oculta lixo nativo */
footer {visibility: hidden;} #MainMenu {visibility: hidden;} header {visibility: hidden;}
</style>
"""
    if os.path.exists(image_filename):
        with open(image_filename, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        bg_css = f"""<style>.stApp {{ background-image: url("data:image/jpeg;base64,{encoded_string}"); background-size: cover !important; background-position: center top !important; background-attachment: fixed !important; background-repeat: no-repeat !important; }}</style>"""
        st.markdown(css_base + bg_css, unsafe_allow_html=True)
    else:
        st.markdown(css_base, unsafe_allow_html=True)

set_background_and_glassmorphism("fundo.jpeg")


# ---------------------------------------------------------------------------
# Inicialização do estado
# ---------------------------------------------------------------------------
def _init():
    defaults = {
        "df_raw": None, "df_pluv": None, "df_reserv": None,
        "inst_type": None, "tipo_material": None, "permeabilidade_cms": None, 
        "n_days_horizon": 30, "freq_hours": 24.0, "freq_info": None,
        "training_result": None, "all_forecasts": {}, "pipeline_done": False, "forecast_done": False,
        "enable_tuning": True, "enable_uncertainty": True, "perm_db": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

_init()
if st.session_state["perm_db"] is None: st.session_state["perm_db"] = load_permeability_db()
def _ok(msg): st.success(msg)
def _warn(msg): st.warning(msg)

# ---------------------------------------------------------------------------
# CABEÇALHO COM LOGO + SUBTÍTULO
# ---------------------------------------------------------------------------
col_logo, col_text = st.columns([1, 10])
with col_logo:
    try: 
        st.image("logo.png", width=90)
    except FileNotFoundError: 
        st.markdown("<h3 style='color: white;'>GEOPredict</h3>", unsafe_allow_html=True)
with col_text:
    st.markdown("""
        <div style="display: flex; flex-direction: column; justify-content: center; height: 100%; margin-left: -20px; margin-top: 5px;">
            <h2 style='margin: 0; padding: 0; color: #f8fafc; font-weight: 700; letter-spacing: -0.5px;'>GEOPredict</h2>
            <span style='color: #94a3b8; font-size: 1.15rem; font-weight: 400;'>Sistema Inteligente de Previsão de Instrumentos Geotécnicos</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===========================================================================
# PAINEL DE CONTROLE (3 COLUNAS)
# ===========================================================================
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("<h4>1. Upload dos dados do instrumento</h4>", unsafe_allow_html=True)
    col_btn_demo, _ = st.columns([1, 1])
    with col_btn_demo:
        use_sample = st.button("Usar demonstração", key="sample_btn", use_container_width=True)
    
    uploaded_main = st.file_uploader("Arquivo CSV:", type=["csv"], key="upload_main", label_visibility="collapsed")

    if use_sample:
        with st.spinner("Gerando dados..."): df_sample = generate_sample_data(n_instruments=3, n_days=730)
        st.session_state["df_raw"] = df_sample
        st.session_state["freq_hours"] = 24.0
        st.session_state["freq_info"] = {"freq_hours": 24.0, "confidence": "alta", "n_valid_intervals": 729}
        st.session_state["df_pluv"] = None 

    elif uploaded_main is not None:
        try:
            st.session_state["df_raw"] = load_main_data(uploaded_main.read(), uploaded_main.name)
            st.session_state["forecast_done"] = False
        except Exception as e: st.error(f"Erro: {e}")

    if st.session_state["df_raw"] is not None:
        df_raw = st.session_state["df_raw"]
        schema = validate_schema(df_raw)
        if not schema.ok: st.error("Erro: " + ", ".join(schema.missing_required))
        else:
            freq_info = detect_series_frequency(pd.to_datetime(df_raw["data"], errors="coerce"))
            st.session_state["freq_info"] = freq_info
            st.session_state["freq_hours"] = freq_info["freq_hours"]
            _ok(f"Carregado: {len(df_raw):,} registros.")

    st.markdown("<h4>2. Tipo de instrumento</h4>", unsafe_allow_html=True)
    options  = [e.value for e in InstrumentType]
    selected = st.radio("Selecione:", options, index=0, key="inst_type_radio", label_visibility="collapsed")
    st.session_state["inst_type"] = selected
    inst_enum = InstrumentType(selected)
    st.session_state["freq_hours"] = (st.session_state.get("freq_info") or {}).get("freq_hours", INSTRUMENT_FREQ_HOURS[inst_enum])

with col2:
    st.markdown("<h4>3. Tipo de solo/material</h4>", unsafe_allow_html=True)
    
    soil_options = get_soil_options(st.session_state["perm_db"])
    if not soil_options:
        soil_options = ["Argila", "Silte", "Areia Fina", "Areia Grossa", "Pedregulho", "Xisto", "Solo Residual"]
        
    c_sel, c_txt = st.columns(2)
    with c_sel: soil_selected = st.selectbox("Lista:", ["— Selecione —"] + soil_options, key="soil_sel")
    with c_txt: soil_free = st.text_input("Manual:", key="soil_free", placeholder="Ex: Xisto...")
    
    tipo_material = soil_free.strip() if soil_free.strip() else (soil_selected if soil_selected != "— Selecione —" else None)
    perm_ref = None
    
    if tipo_material:
        st.session_state["tipo_material"] = tipo_material
        perm_ref = lookup_permeability(tipo_material, st.session_state["perm_db"])
        
        if not perm_ref:
            fallbacks = {
                "Argila": {"k_cms": 1e-7, "fonte": "Padrão"}, "Silte": {"k_cms": 1e-5, "fonte": "Padrão"},
                "Areia Fina": {"k_cms": 1e-3, "fonte": "Padrão"}, "Areia Grossa": {"k_cms": 1e-2, "fonte": "Padrão"},
                "Pedregulho": {"k_cms": 1e-1, "fonte": "Padrão"}, "Xisto": {"k_cms": 1e-6, "fonte": "Padrão"},
                "Solo Residual": {"k_cms": 1e-4, "fonte": "Padrão"}
            }
            if tipo_material in fallbacks:
                perm_ref = fallbacks[tipo_material]
                
        if perm_ref: _ok(f"k ref: {perm_ref['k_cms']:.2e} cm/s")

    st.markdown("<h4>4. Permeabilidade hidráulica</h4>", unsafe_allow_html=True)
    if perm_ref and st.checkbox(f"Usar sugerido: {perm_ref['k_cms']:.2e} cm/s", value=True, key="use_suggested_perm"):
        st.session_state["permeabilidade_cms"] = perm_ref["k_cms"]
    else:
        st.session_state["permeabilidade_cms"] = st.number_input("Valor k:", min_value=1e-12, value=float(perm_ref["k_cms"]) if perm_ref else 1e-6, format="%.2e", label_visibility="collapsed")

    st.markdown("<h4>5. Dados pluviométricos</h4>", unsafe_allow_html=True)
    if st.session_state.get("df_raw") is not None and "pluviometria" in st.session_state["df_raw"].columns:
        _ok("Pluviometria na base principal.")
    else:
        uploaded_pluv = st.file_uploader("Upload chuva:", type=["csv"], key="up_pluv", label_visibility="collapsed")
        if uploaded_pluv:
            st.session_state["df_pluv"] = load_external_data(uploaded_pluv.read(), uploaded_pluv.name)
            _ok(f"Chuva: {len(st.session_state['df_pluv']):,} reg.")

with col3:
    st.markdown("<h4>6. Nível d'água do reservatório</h4>", unsafe_allow_html=True)
    if st.checkbox("Incluir nível (Opcional)", value=False):
        uploaded_reserv = st.file_uploader("Upload res:", type=["csv"], key="up_reserv", label_visibility="collapsed")
        if uploaded_reserv:
            st.session_state["df_reserv"] = load_external_data(uploaded_reserv.read(), uploaded_reserv.name)
            _ok("Reservatório carregado.")
    else:
        st.session_state["df_reserv"] = None

    st.markdown("<h4>7. Horizonte e Configuração</h4>", unsafe_allow_html=True)
    selected_prod = st.radio("Horizonte:", [h[0] for h in FORECAST_HORIZONS_PRODUCAO], index=0, horizontal=True, label_visibility="collapsed")
    st.session_state["enable_tuning"] = st.checkbox("Tuning (Optuna)", value=True)
    st.session_state["enable_uncertainty"] = st.checkbox("Calcular incerteza", value=True)
    st.session_state["n_days_horizon"] = next((h[1] for h in FORECAST_HORIZONS_PRODUCAO if h[0] == selected_prod), 30)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Gerar Previsão do Instrumento", type="primary", use_container_width=True)

# ===========================================================================
# PROCESSAMENTO E DIAGNÓSTICO
# ===========================================================================
st.markdown("---")
if run_btn:
    if st.session_state.get("df_raw") is None: st.error("Atenção: Faça o upload (Etapa 1).")
    elif st.session_state.get("tipo_material") is None: st.error("Atenção: Selecione o material (Etapa 3).")
    elif st.session_state.get("df_pluv") is None and ("pluviometria" not in st.session_state.get("df_raw", pd.DataFrame()).columns): 
        st.error("Atenção: Pluviometria necessária (Etapa 5). Lembre-se: Se usou a 'Demonstração', não faça upload de chuva real!")
    else:
        progress = st.progress(0, "Iniciando processamento...")
        try:
            # --- TRAVA SÊNIOR: ANÁLISE DE DATAS PARA DEBUG ---
            df_inst_diag = st.session_state["df_raw"]
            df_pluv_diag = st.session_state.get("df_pluv")
            
            try:
                min_inst = pd.to_datetime(df_inst_diag['data'], dayfirst=True, format='mixed').min().strftime('%d/%m/%Y')
                max_inst = pd.to_datetime(df_inst_diag['data'], dayfirst=True, format='mixed').max().strftime('%d/%m/%Y')
                diag_msg = f"🗓️ **Período do Instrumento:** {min_inst} até {max_inst}\n"
                
                if df_pluv_diag is not None and 'data' in df_pluv_diag.columns:
                    min_pluv = pd.to_datetime(df_pluv_diag['data'], dayfirst=True, format='mixed').min().strftime('%d/%m/%Y')
                    max_pluv = pd.to_datetime(df_pluv_diag['data'], dayfirst=True, format='mixed').max().strftime('%d/%m/%Y')
                    diag_msg += f"🌧️ **Período da Chuva:** {min_pluv} até {max_pluv}"
            except Exception:
                diag_msg = "Não foi possível extrair as datas automaticamente."
            # --------------------------------------------------

            df = st.session_state["df_raw"].copy()
            for col in ["leitura", "valor"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

            df = inject_instrument_constants(df, st.session_state["inst_type"], st.session_state["tipo_material"], st.session_state["permeabilidade_cms"])
            
            if st.session_state.get("df_pluv") is not None:
                df_p = st.session_state["df_pluv"].copy()
                if "pluviometria" in df_p.columns: df_p["pluviometria"] = pd.to_numeric(df_p["pluviometria"].astype(str).str.replace(',', '.'), errors='coerce')
                df, _ = merge_external_data(df, df_p)
            
            if st.session_state.get("df_reserv") is not None:
                df_r = st.session_state["df_reserv"].copy()
                if "nivel_reservatorio" in df_r.columns: df_r["nivel_reservatorio"] = pd.to_numeric(df_r["nivel_reservatorio"].astype(str).str.replace(',', '.'), errors='coerce')
                df, _ = merge_external_data(df, df_r)

            progress.progress(20, "Pré-processamento e engenharia...")
            df_proc, encoders = run_preprocessing(df).df, run_preprocessing(df).encoders
            df_proc, quality_report = treat_missing_values(df_proc, cfg=MissingValueConfig())
            df_proc = apply_outlier_filter(detect_outliers(df_proc)[0], mode="flag_only")
            
            feat_cfg = get_feature_config_for_instrument(InstrumentType(st.session_state["inst_type"]), st.session_state["freq_hours"])
            df_feat = build_features(df_proc, cfg=feat_cfg)
            feature_cols = get_feature_columns(df_feat)

            # INTERCEPTA O ERRO ANTES DO XGBOOST
            if df_feat.empty or len(df_feat) == 0:
                st.error("🚨 O Treinamento foi abortado porque a tabela de dados ficou VAZIA.")
                st.warning("Isso aconteceu porque as datas dos seus arquivos não se sobrepõem. Veja a análise abaixo:\n\n" + diag_msg)
                st.info("💡 **Solução:** Envie um arquivo de Pluviometria que tenha dados nos mesmos anos que o Piezômetro.")
                st.stop()

            progress.progress(50, "Treinando Inteligência Artificial (XGBoost)...")
            result = run_training(df_feat, feature_cols, tuning_cfg=TuningConfig(enabled=st.session_state["enable_tuning"], n_trials=20, timeout_seconds=90), cv_cfg=DEFAULT_CV, run_cv=True, run_tuning=st.session_state["enable_tuning"])

            progress.progress(80, "Calculando previsões...")
            all_forecasts = {}
            for inst in sorted(df_feat["instrumento"].unique()):
                all_forecasts[inst] = recursive_forecast(model=result.model, df_history=df_feat, feature_cols=feature_cols, instrumento=inst, n_days=st.session_state["n_days_horizon"], freq_hours=st.session_state["freq_hours"], cfg=feat_cfg, X_train=result.test_df[feature_cols])

            progress.progress(100, "Concluído!")
            st.session_state.update({"df_featured": df_feat, "feature_cols": feature_cols, "training_result": result, "all_forecasts": all_forecasts, "forecast_done": True, "shap_global": compute_shap_values(result.model, result.test_df[feature_cols].fillna(0), feature_cols)})
        except Exception as e: st.error(f"Erro no processamento: {e}")

# ===========================================================================
# NOVO DASHBOARD DE RESULTADOS (PREMIUM)
# ===========================================================================
if st.session_state.get("forecast_done"):
    st.markdown("---")
    st.markdown("""
        <div style='display: flex; align-items: center; margin-bottom: 20px;'>
            <h2 style='color: #f8fafc; font-size: 2rem; font-weight: 700; margin-right: 15px;'>📊 Dashboard de Resultados</h2>
            <div style='background: rgba(59, 130, 246, 0.2); color: #60a5fa; padding: 5px 15px; border-radius: 20px; font-weight: 600; font-size: 0.9rem;'>
                Previsão Concluída
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    df_feat = st.session_state["df_featured"]
    all_forecasts = st.session_state.get("all_forecasts", {})
    result = st.session_state["training_result"]
    
    selected_inst = st.selectbox("Selecione o Instrumento para Análise Detalhada:", sorted(df_feat["instrumento"].unique()))

    st.markdown("<br>", unsafe_allow_html=True)
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    mt = result.metrics_test
    
    with col_kpi1:
        st.metric(label="MAE (Erro Absoluto)", value=round(mt.get('MAE', 0), 4))
    with col_kpi2:
        st.metric(label="RMSE (Raiz do Erro)", value=round(mt.get('RMSE', 0), 4))
    with col_kpi3:
        st.metric(label="Acurácia R²", value=f"{round(mt.get('R²', 0), 2)}")
    with col_kpi4:
        st.metric(label="Horizonte", value=f"{st.session_state['n_days_horizon']} dias")
    
    st.markdown("<br>", unsafe_allow_html=True)

    forecast_df = all_forecasts.get(selected_inst, {}).get("forecast_df", pd.DataFrame())

    if not forecast_df.empty:
        fig_linha = px.line(forecast_df, x="data", y="previsao", markers=True)
        fig_linha.update_traces(line_color="#3b82f6", marker=dict(size=6, color="#60a5fa")) 
        
        fig_linha.update_layout(
            title=dict(text=f"Projeção Piezométrica: Instrumento {selected_inst}", font=dict(size=20, color="#f8fafc")),
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0.15)", 
            font_color="#cbd5e1", 
            title_font_color="#f8fafc", 
            legend_font_color="#cbd5e1",
            xaxis=dict(title="", showgrid=True, gridcolor="rgba(255,255,255,0.08)", showline=True, linecolor="rgba(255,255,255,0.2)"),
            yaxis=dict(title="Nível / Leitura (m)", showgrid=True, gridcolor="rgba(255,255,255,0.08)", showline=True, linecolor="rgba(255,255,255,0.2)"),
            hovermode="x unified",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_linha, use_container_width=True)

        col_down1, col_down2, _ = st.columns([1, 1, 2])
        with col_down1:
            csv_bytes = forecast_df.to_csv(index=False, sep=";", decimal=",").encode('utf-8')
            st.download_button(label="📥 Baixar Tabela de Previsão (.CSV)", data=csv_bytes, file_name=f"previsao_{selected_inst}_{datetime.date.today()}.csv", mime="text/csv", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab_shap, tab_diag = st.tabs(["🔍 Interpretabilidade (SHAP)", "⚙️ Diagnóstico do Modelo"])
    
    with tab_shap:
        if st.session_state.get("shap_global"):
            st.markdown("<h4 style='border:none; color:#f8fafc; font-size:1.3rem; margin-top:10px;'>Impacto das Variáveis (Análise SHAP)</h4>", unsafe_allow_html=True)
            st.caption("Entenda quais variáveis (chuva, dias, tempo) mais impactam as subidas e descidas na previsão do modelo.")
            fig_shap = plot_shap_summary(st.session_state["shap_global"])
            fig_shap = apply_dark_theme(fig_shap) 
            st.plotly_chart(fig_shap, use_container_width=True)
            
    with tab_diag:
        st.markdown("<h4 style='border:none; color:#f8fafc; font-size:1.3rem; margin-top:10px;'>Robustez do Treinamento</h4>", unsafe_allow_html=True)
        st.write("Resultados da Validação Cruzada Temporal (Quanto mais estáveis as dobras, mais confiável é o modelo):")
        if result.cv_results and "aggregated" in result.cv_results:
            st.dataframe(format_cv_results(result.cv_results), use_container_width=True, hide_index=True)
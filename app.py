"""
app.py — PiezoPrev Final
Interface Streamlit completa com fluxo de 8 etapas.
Execute com: streamlit run app.py
"""

import io
import logging
import warnings
import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# Config e esquema
from config import (
    InstrumentType, INSTRUMENT_FREQ_HOURS, INSTRUMENT_FREQ_LABEL,
    FORECAST_HORIZONS_PRODUCAO, FORECAST_HORIZONS_EXPERIMENTAL,
    FORECAST_DEFAULT_LABEL, HORIZON_RELIABILITY_MSGS,
    DEFAULT_MODEL, DEFAULT_MISSING, DEFAULT_TUNING, DEFAULT_CV,
    ModelConfig, TuningConfig, CVConfig,
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
    page_title="PiezoPrev",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.header{background:linear-gradient(135deg,#0F2942 0%,#1D4ED8 100%);color:white;
  padding:2rem 2.5rem;border-radius:14px;margin-bottom:2rem;}
.header h1{font-size:2rem;font-weight:600;margin:0;letter-spacing:-.5px;}
.header p{opacity:.78;margin:.4rem 0 0 0;font-size:.9rem;}
.step-badge{display:inline-block;background:#1D4ED8;color:white;border-radius:50%;
  width:28px;height:28px;text-align:center;line-height:28px;font-weight:600;
  font-size:.85rem;margin-right:.5rem;}
.step-title{font-size:1.1rem;font-weight:600;color:#1E3A5F;margin-bottom:.5rem;}
.step-card{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
  padding:1.4rem 1.6rem;margin-bottom:1.2rem;}
.info-box{background:#EFF6FF;border-left:4px solid #2563EB;padding:.7rem 1rem;
  border-radius:0 8px 8px 0;margin:.5rem 0 1rem 0;font-size:.87rem;line-height:1.6;}
.warn-box{background:#FFF7ED;border-left:4px solid #F97316;padding:.7rem 1rem;
  border-radius:0 8px 8px 0;margin:.5rem 0 .8rem 0;font-size:.87rem;}
.danger-box{background:#FFF1F2;border-left:4px solid #EF4444;padding:.7rem 1rem;
  border-radius:0 8px 8px 0;margin:.5rem 0 .8rem 0;font-size:.87rem;font-weight:500;}
.ok-box{background:#F0FDF4;border-left:4px solid #22C55E;padding:.7rem 1rem;
  border-radius:0 8px 8px 0;margin:.5rem 0 .8rem 0;font-size:.87rem;}
.kpi-card{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
  padding:1rem 1.2rem;text-align:center;}
.kpi-val{font-size:1.65rem;font-weight:600;color:#1E3A5F;
  font-family:'IBM Plex Mono',monospace;}
.kpi-lab{font-size:.73rem;color:#64748B;text-transform:uppercase;
  letter-spacing:.04em;margin-top:.2rem;}
.stButton>button{background:#2563EB;color:white;border:none;border-radius:8px;font-weight:500;}
.stButton>button:hover{background:#1D4ED8;}
hr{border:none;border-top:1px solid #E2E8F0;margin:1.5rem 0;}
/* Oculta o rodapé padrão do Streamlit */
footer {
    visibility: hidden;
}

/* Esconde rodapé e cabeçalho padrão */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Esconde qualquer classe que tenha "viewerBadge" no nome (ignora os códigos aleatórios) */
div[class*="viewerBadge"] {
    display: none !important;
}

/* Esconde o botão de deploy e barra de ferramentas nativa */
.stAppDeployButton {display: none !important;}
[data-testid="stToolbar"] {display: none !important;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Inicialização do estado da sessão
# ---------------------------------------------------------------------------

def _init():
    defaults = {
        "step": 1,
        "df_raw": None, "df_pluv": None, "df_reserv": None,
        "inst_type": None, "tipo_material": None,
        "permeabilidade_cms": None, "perm_is_estimated": False,
        "perm_fonte": "", "perm_observacao": "",
        "n_days_horizon": 30, "horizon_label": "30 dias",
        "freq_hours": 24.0, "freq_info": None,
        "df_featured": None, "feature_cols": None, "encoders": None,
        "training_result": None, "all_forecasts": {},
        "all_uncertainties": {},
        "pipeline_done": False, "training_done": False, "forecast_done": False,
        "enable_tuning": True, "enable_uncertainty": True,
        "shap_global": None, "shap_local": {},
        "perm_db": None,   # cache da base de permeabilidade carregada
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init()

# Carregar base de permeabilidade uma vez por sessão
if st.session_state["perm_db"] is None:
    st.session_state["perm_db"] = load_permeability_db()


# ---------------------------------------------------------------------------
# Helpers de UI
# ---------------------------------------------------------------------------

def _info(msg):   st.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)
def _warn(msg):   st.markdown(f'<div class="warn-box">⚠️ {msg}</div>', unsafe_allow_html=True)
def _danger(msg): st.markdown(f'<div class="danger-box">🔴 {msg}</div>', unsafe_allow_html=True)
def _ok(msg):     st.markdown(f'<div class="ok-box">✅ {msg}</div>', unsafe_allow_html=True)
def _kpi(lab, val):
    return (f'<div class="kpi-card">'
            f'<div class="kpi-val">{val}</div>'
            f'<div class="kpi-lab">{lab}</div>'
            f'</div>')


def _step_header(n, title):
    st.markdown(
        f'<div class="step-title">'
        f'<span class="step-badge">{n}</span>{title}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _nav_buttons(back_step=None, next_step=None, next_label="Continuar →"):
    if back_step and next_step:
        cols = st.columns([1, 3, 1])
        with cols[0]:
            if st.button("← Voltar", key=f"back_{back_step}"):
                st.session_state["step"] = back_step
                st.rerun()
        with cols[2]:
            if st.button(next_label, key=f"next_{next_step}", type="primary"):
                st.session_state["step"] = next_step
                st.rerun()
    elif back_step:
        if st.button("← Voltar", key=f"back_{back_step}"):
            st.session_state["step"] = back_step
            st.rerun()
    elif next_step:
        cols = st.columns([4, 1])
        with cols[1]:
            if st.button(next_label, key=f"next_{next_step}", type="primary"):
                st.session_state["step"] = next_step
                st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="header">
<h1>📡 PiezoPrev</h1>
<p>Previsão de Piezômetros e Medidores de Nível d'Água · XGBoost · SHAP · CV Temporal</p>
</div>
""", unsafe_allow_html=True)

step = st.session_state["step"]
st.progress(min(100, int((step - 1) / 8 * 100)), text=f"Etapa {step} de 8")
st.markdown("")


# ===========================================================================
# ETAPA 1 — Upload das leituras
# ===========================================================================

if step == 1:
    _step_header(1, "Upload dos dados do instrumento")
    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    _info(
        "<strong>Formato aceito:</strong> arquivo <code>.csv</code> com pelo menos 3 colunas — "
        "<em>código do instrumento</em>, <em>data da leitura</em> e <em>valor</em>. "
        "O sistema reconhece automaticamente nomes como: 'Instrumento', 'Data', "
        "'Valor', 'Leitura', 'Nível', 'Cota', 'Código do Instrumento', 'Data da Leitura' etc.<br><br>"
        "A pluviometria deve ser fornecida separadamente na Etapa 5. "
        "O nível do reservatório é opcional (Etapa 6)."
    )

    use_sample     = st.button("🔬 Carregar dados de demonstração", key="sample_btn")
    uploaded_main  = st.file_uploader(
        "Carregar arquivo de leituras (.csv):", type=["csv"], key="upload_main"
    )

    if use_sample:
        with st.spinner("Gerando dados de demonstração..."):
            df_sample = generate_sample_data(n_instruments=3, n_days=730)
        st.session_state["df_raw"]    = df_sample
        st.session_state["freq_hours"] = 24.0
        st.session_state["freq_info"]  = {"freq_hours": 24.0, "confidence": "alta",
                                           "n_valid_intervals": 729}

    elif uploaded_main is not None:
        try:
            raw_bytes = uploaded_main.read()
            df_loaded = load_main_data(raw_bytes, uploaded_main.name)
            st.session_state["df_raw"]           = df_loaded
            st.session_state["pipeline_done"]    = False
            st.session_state["training_done"]    = False
            st.session_state["forecast_done"]    = False
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")

    if st.session_state["df_raw"] is not None:
        df_raw = st.session_state["df_raw"]
        schema = validate_schema(df_raw)

        if not schema.ok:
            st.error(
                "Arquivo inválido — colunas obrigatórias ausentes: "
                + ", ".join(schema.missing_required)
                + "\n\nVerifique a estrutura do arquivo e tente novamente."
            )
        else:
            freq_info = detect_series_frequency(
                pd.to_datetime(df_raw["data"], errors="coerce")
            )
            st.session_state["freq_info"]  = freq_info
            st.session_state["freq_hours"] = freq_info["freq_hours"]

            _ok(
                f"Arquivo carregado: {len(df_raw):,} registros | "
                f"{schema.n_instruments} instrumento(s) | "
                f"Período: {schema.date_range[0]} → {schema.date_range[1]}"
            )

            conf_icon = {"alta": "✅", "média": "⚠️", "baixa": "⚠️"}.get(
                freq_info["confidence"], "ℹ️"
            )
            st.info(
                f"{conf_icon} Frequência detectada: **{freq_info['freq_hours']:.1f}h** "
                f"entre leituras — confiança: {freq_info['confidence']} "
                f"({freq_info['n_valid_intervals']} intervalos válidos)"
            )

            for w in schema.type_warnings:
                _warn(w)
            for m in schema.info_messages:
                _info(m)

            with st.expander("Ver amostra dos dados"):
                st.dataframe(df_raw.head(20), use_container_width=True)

            _nav_buttons(next_step=2, next_label="Continuar → Tipo de instrumento")

    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# ETAPA 2 — Tipo de instrumento
# ===========================================================================

elif step == 2:
    _step_header(2, "Tipo de instrumento")
    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    _info(
        "Selecione o tipo de instrumento. Isso determina a frequência temporal esperada "
        "e como os lags e janelas são calibrados no modelo.<br>"
        "A frequência detectada automaticamente na Etapa 1 tem prioridade quando "
        "a confiança for alta ou média."
    )

    options  = [e.value for e in InstrumentType]
    prev     = st.session_state.get("inst_type")
    prev_idx = options.index(prev) if prev in options else 0

    selected = st.radio(
        "Tipo do instrumento:", options, index=prev_idx, key="inst_type_radio"
    )
    st.session_state["inst_type"] = selected

    inst_enum      = InstrumentType(selected)
    freq_default   = INSTRUMENT_FREQ_HOURS[inst_enum]
    freq_info      = st.session_state.get("freq_info", {})
    freq_detected  = freq_info.get("freq_hours", freq_default) if freq_info else freq_default
    freq_confidence = freq_info.get("confidence", "baixa") if freq_info else "baixa"

    freq_used = freq_detected if freq_confidence in ("alta", "média") else freq_default
    st.session_state["freq_hours"] = freq_used

    _ok(
        f"Tipo selecionado: <strong>{selected}</strong> — "
        f"{INSTRUMENT_FREQ_LABEL[inst_enum]}<br>"
        f"Frequência a usar: <strong>{freq_used:.1f}h</strong> "
        f"({'detectada dos dados' if freq_confidence in ('alta', 'média') else 'padrão do tipo'})"
    )

    st.markdown('</div>', unsafe_allow_html=True)
    _nav_buttons(back_step=1, next_step=3, next_label="Continuar → Solo/Material")


# ===========================================================================
# ETAPA 3 — Tipo de solo/material
# ===========================================================================

elif step == 3:
    _step_header(3, "Tipo de solo/material")
    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    _info(
        "Informe o material geotécnico onde o instrumento está instalado. "
        "<strong>Obrigatório</strong> — influencia a permeabilidade e o comportamento esperado."
    )

    perm_db = st.session_state["perm_db"]
    soil_options = get_soil_options(perm_db)

    col_sel, col_txt = st.columns([1, 1])
    with col_sel:
        soil_selected = st.selectbox(
            "Selecionar da lista:",
            ["— Selecione —"] + soil_options,
            key="soil_sel",
        )
    with col_txt:
        soil_free = st.text_input(
            "Ou digitar livremente:",
            key="soil_free",
            placeholder="Ex: Xisto, Colúvio, Solo Residual...",
        )

    tipo_material = (
        soil_free.strip() if soil_free.strip()
        else (soil_selected if soil_selected != "— Selecione —" else None)
    )

    if tipo_material:
        st.session_state["tipo_material"] = tipo_material
        perm_ref = lookup_permeability(tipo_material, perm_db)
        if perm_ref:
            fonte_str = f" (Fonte: {perm_ref['fonte']})" if perm_ref.get("fonte") else ""
            obs_str   = f"<br>ℹ️ {perm_ref['observacao']}" if perm_ref.get("observacao") else ""
            _ok(
                f"Material reconhecido: <strong>{tipo_material}</strong><br>"
                f"Permeabilidade de referência: k = <strong>{perm_ref['k_cms']:.2e} cm/s</strong>"
                + (f" | Faixa: {perm_ref['k_min']:.0e} – {perm_ref['k_max']:.0e} cm/s"
                   if perm_ref.get("k_min") and perm_ref.get("k_max") else "")
                + fonte_str + obs_str
            )
        else:
            _info(
                f"Material '{tipo_material}' não encontrado na base de referência. "
                "Você informará a permeabilidade manualmente na próxima etapa."
            )
    else:
        st.warning("Selecione ou informe o tipo de material para continuar.")

    # Status da base de permeabilidade
    with st.expander("ℹ️ Status da base de permeabilidade"):
        db_status = get_db_status()
        if db_status["loaded"]:
            st.success(
                f"Base externa carregada: {db_status['n_records']} materiais disponíveis."
            )
            st.caption(f"Arquivo: {db_status['expected_path_csv']}")
        else:
            st.warning(
                "Base de permeabilidade não encontrada. "
                f"Esperado: {db_status['expected_path_csv']}. "
                "Você pode fazer upload na seção abaixo ou informar o valor manualmente."
            )
            uploaded_db = st.file_uploader(
                "Carregar base de permeabilidade (.csv ou .xlsx):",
                type=["csv", "xlsx"],
                key="upload_perm_db",
            )
            if uploaded_db:
                try:
                    perm_db_new = load_permeability_db(
                        filepath=uploaded_db.read(), force_reload=True
                    )
                    st.session_state["perm_db"] = perm_db_new
                    _ok(f"Base carregada via upload: {len(perm_db_new)} materiais.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao carregar base: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    if tipo_material:
        _nav_buttons(back_step=2, next_step=4, next_label="Continuar → Permeabilidade")
    else:
        _nav_buttons(back_step=2)


# ===========================================================================
# ETAPA 4 — Permeabilidade
# ===========================================================================

elif step == 4:
    _step_header(4, "Permeabilidade hidráulica")
    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    tipo_material = st.session_state.get("tipo_material", "")
    perm_db       = st.session_state["perm_db"]
    perm_ref      = lookup_permeability(tipo_material, perm_db) if tipo_material else None

    _info(
        "Informe a permeabilidade hidráulica (k) do material em <strong>cm/s</strong>. "
        "Caso não possua o valor medido, o sistema pode sugerir um valor de referência "
        "com base no tipo de solo informado na etapa anterior."
    )

    usar_sugerido = False
    if perm_ref:
        fonte_display = f" — Fonte: <em>{perm_ref['fonte']}</em>" if perm_ref.get("fonte") else ""
        st.markdown(
            f"**Valor de referência para '{tipo_material}':** "
            f"k = **{perm_ref['k_cms']:.2e} cm/s**"
            + (f" | Faixa: {perm_ref['k_min']:.0e} – {perm_ref['k_max']:.0e} cm/s"
               if perm_ref.get("k_min") and perm_ref.get("k_max") else "")
        )
        if fonte_display:
            st.markdown(f"📚{fonte_display}", unsafe_allow_html=True)
        _warn(
            "Este é um valor <strong>estimado</strong> por referência bibliográfica. "
            "Se você possui dados de ensaio de campo ou laboratório, prefira utilizá-los."
        )
        usar_sugerido = st.checkbox(
            f"Usar valor sugerido: k = {perm_ref['k_cms']:.2e} cm/s",
            value=True,
            key="use_suggested_perm",
        )

    if usar_sugerido and perm_ref:
        k_value = perm_ref["k_cms"]
        st.session_state["perm_is_estimated"] = True
        st.session_state["perm_fonte"]        = perm_ref.get("fonte", "Referência bibliográfica")
        st.session_state["perm_observacao"]   = perm_ref.get("observacao", "")
    else:
        default_k = float(perm_ref["k_cms"]) if perm_ref else 1e-6
        k_value = st.number_input(
            "Valor de k (cm/s):",
            min_value=1e-12,
            max_value=10.0,
            value=default_k,
            format="%.2e",
            key="perm_input",
            help=(
                "Exemplos de referência:\n"
                "Argila: ~1e-7 cm/s\n"
                "Areia: ~1e-3 cm/s\n"
                "Rocha: ~1e-6 cm/s\n"
                "Colúvio: ~1e-4 cm/s"
            ),
        )
        st.session_state["perm_is_estimated"] = False
        st.session_state["perm_fonte"]        = "Informado pelo usuário"
        st.session_state["perm_observacao"]   = ""

    st.session_state["permeabilidade_cms"] = k_value
    _ok(
        f"Permeabilidade definida: k = {k_value:.2e} cm/s "
        f"({'estimada por referência' if st.session_state['perm_is_estimated'] else 'informada pelo usuário'})"
    )

    st.markdown('</div>', unsafe_allow_html=True)
    _nav_buttons(back_step=3, next_step=5, next_label="Continuar → Pluviometria")


# ===========================================================================
# ETAPA 5 — Pluviometria (obrigatória)
# ===========================================================================

elif step == 5:
    _step_header(5, "Dados pluviométricos (obrigatório)")
    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    _info(
        "Os dados de chuva são <strong>obrigatórios</strong> para o modelo.<br><br>"
        "📌 <strong>Fontes recomendadas:</strong><br>"
        "&nbsp;&nbsp;• <a href='https://hidroweb.ana.gov.br' target='_blank'>ANA — Hidroweb</a> "
        "(estações pluviométricas nacionais)<br>"
        "&nbsp;&nbsp;• <a href='https://bdmep.inmet.gov.br' target='_blank'>INMET — BDMEP</a> "
        "(dados meteorológicos históricos)<br>"
        "&nbsp;&nbsp;• Estação meteorológica local da barragem ou do consórcio operador<br><br>"
        "<strong>Formato esperado:</strong> CSV com colunas 'data' e 'pluviometria' (mm/dia). "
        "Outros nomes aceitos: 'chuva', 'precipitacao', 'Chuva (mm)'."
    )

    df_raw   = st.session_state.get("df_raw")
    pluv_ready = False

    if df_raw is not None and "pluviometria" in df_raw.columns:
        _ok("Pluviometria já presente na base de leituras — será utilizada automaticamente.")
        st.session_state["df_pluv"] = None
        pluv_ready = True
    else:
        uploaded_pluv = st.file_uploader(
            "Arquivo de pluviometria (.csv):", type=["csv"], key="up_pluv"
        )
        if uploaded_pluv:
            try:
                df_pluv = load_external_data(uploaded_pluv.read(), uploaded_pluv.name)
                if "pluviometria" not in df_pluv.columns:
                    st.error(
                        "Coluna de pluviometria não identificada. "
                        f"Colunas encontradas: {list(df_pluv.columns)}. "
                        "Renomeie para 'pluviometria', 'chuva' ou 'precipitacao'."
                    )
                else:
                    if df_raw is not None:
                        for c in check_external_data_coverage(
                            df_raw, df_pluv, ext_cols=["pluviometria"]
                        ):
                            _warn(c)
                    st.session_state["df_pluv"] = df_pluv
                    pluv_ready = True
                    _ok(
                        f"Pluviometria carregada: {len(df_pluv):,} registros | "
                        f"Período: "
                        f"{pd.to_datetime(df_pluv['data']).min().date()} → "
                        f"{pd.to_datetime(df_pluv['data']).max().date()}"
                    )
                    with st.expander("Ver amostra"):
                        st.dataframe(df_pluv.head(15), use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao carregar pluviometria: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    if pluv_ready:
        _nav_buttons(
            back_step=4, next_step=6, next_label="Continuar → Reservatório (opcional)"
        )
    else:
        _nav_buttons(back_step=4)


# ===========================================================================
# ETAPA 6 — Nível do reservatório (opcional)
# ===========================================================================

elif step == 6:
    _step_header(6, "Nível d'água do reservatório (opcional)")
    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    _info(
        "Esta etapa é <strong>opcional</strong>. Recomendada para barragens com reservatório "
        "relevante e para instrumentos próximos ao corpo d'água.<br>"
        "Formato: CSV com 'data' e 'nivel_reservatorio' — ou: 'NA', 'cota reservatorio'."
    )

    df_raw = st.session_state.get("df_raw")
    if df_raw is not None and "nivel_reservatorio" in df_raw.columns:
        _ok("Nível do reservatório já presente na base de leituras.")
        st.session_state["df_reserv"] = None
    else:
        uploaded_reserv = st.file_uploader(
            "Arquivo do nível do reservatório (.csv) — opcional:",
            type=["csv"],
            key="up_reserv",
        )
        if uploaded_reserv:
            try:
                df_r = load_external_data(uploaded_reserv.read(), uploaded_reserv.name)
                if "nivel_reservatorio" not in df_r.columns:
                    st.error(
                        "Coluna de nível do reservatório não identificada. "
                        f"Colunas: {list(df_r.columns)}"
                    )
                else:
                    st.session_state["df_reserv"] = df_r
                    _ok(f"Nível do reservatório carregado: {len(df_r):,} registros.")
                    with st.expander("Ver amostra"):
                        st.dataframe(df_r.head(15), use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao carregar nível do reservatório: {e}")
        else:
            _info(
                "Sem arquivo de reservatório — essa variável será omitida como feature do modelo."
            )

    st.markdown('</div>', unsafe_allow_html=True)
    _nav_buttons(
        back_step=5, next_step=7, next_label="Continuar → Horizonte de previsão"
    )


# ===========================================================================
# ETAPA 7 — Horizonte de previsão e configurações do modelo
# ===========================================================================

elif step == 7:
    _step_header(7, "Horizonte de previsão")
    st.markdown('<div class="step-card">', unsafe_allow_html=True)

    freq_hours = st.session_state.get("freq_hours", 24.0)

    # Horizontes de produção
    st.markdown("#### 📊 Horizontes recomendados para uso operacional")
    _info(
        "Estes horizontes têm melhor equilíbrio entre confiabilidade e utilidade prática."
    )
    prod_labels  = [h[0] for h in FORECAST_HORIZONS_PRODUCAO]
    selected_prod = st.radio(
        "Selecione o horizonte:", prod_labels, index=0, key="horizon_prod"
    )

    # Horizonte experimental
    st.markdown("---")
    st.markdown("#### 🧪 Horizonte experimental")
    use_exp = st.checkbox(
        "Usar horizonte experimental (1 ano)", value=False, key="use_exp"
    )

    if use_exp:
        _danger(
            "MODO EXPERIMENTAL — 1 ANO: confiabilidade muito baixa. "
            "O forecast recursivo acumula erro a cada passo. "
            "Use apenas para análise exploratória de tendência de longo prazo. "
            "NÃO utilize para decisões operacionais ou de segurança de barragens."
        )
        selected_label = "1 ano (experimental)"
        n_days = 365
    else:
        selected_label = selected_prod
        h_cfg  = next(
            (h for h in FORECAST_HORIZONS_PRODUCAO if h[0] == selected_label),
            FORECAST_HORIZONS_PRODUCAO[0],
        )
        n_days = h_cfg[1]

    n_steps          = horizon_days_to_steps(n_days, freq_hours)
    readings_per_day = max(1, round(24.0 / freq_hours))
    freq_label       = f"{freq_hours:.0f}h" if freq_hours < 24 else "diária"

    st.session_state["n_days_horizon"] = n_days
    st.session_state["horizon_label"]  = selected_label

    _ok(
        f"Horizonte: <strong>{selected_label}</strong> | "
        f"Frequência: {freq_label} (~{readings_per_day} leitura(s)/dia) | "
        f"Total de passos de previsão: <strong>{n_steps:,} leituras</strong>"
    )

    rel_msg = HORIZON_RELIABILITY_MSGS.get(n_days)
    if rel_msg:
        if n_days >= 365:
            _danger(rel_msg)
        else:
            _warn(rel_msg)

    # Configurações avançadas
    st.markdown("---")
    st.markdown("#### ⚙️ Configurações do modelo")
    col_a, col_b = st.columns(2)
    with col_a:
        enable_tuning = st.checkbox(
            "Hyperparameter tuning automático (Optuna)",
            value=st.session_state.get("enable_tuning", True),
            key="tune_check",
            help=(
                "Ajusta os hiperparâmetros do XGBoost automaticamente "
                "usando Optuna com validação cruzada temporal. "
                "Adiciona ~1–2 min ao processamento."
            ),
        )
        st.session_state["enable_tuning"] = enable_tuning
    with col_b:
        enable_uncertainty = st.checkbox(
            "Calcular bandas de incerteza",
            value=st.session_state.get("enable_uncertainty", True),
            key="unc_check",
            help=(
                "Estima a propagação do erro no forecast recursivo "
                "via bootstrap. Exibe faixa P10–P90 no gráfico."
            ),
        )
        st.session_state["enable_uncertainty"] = enable_uncertainty

    if enable_tuning:
        _info(
            "O tuning usa Optuna com 20 tentativas e validação cruzada temporal interna. "
            "Tempo estimado: 1–2 minutos dependendo do volume de dados."
        )

    st.markdown('</div>', unsafe_allow_html=True)
    _nav_buttons(back_step=6, next_step=8, next_label="Gerar Previsão 🚀")


# ===========================================================================
# ETAPA 8 — Execução e Resultados
# ===========================================================================

elif step == 8:
    _step_header(8, "Gerar Previsão")

    # Resumo da configuração
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("**Resumo da configuração:**")
    for k, v in {
        "Tipo de instrumento": st.session_state.get("inst_type", "—"),
        "Solo/Material":       st.session_state.get("tipo_material", "—"),
        "Permeabilidade":      (
            f"{st.session_state.get('permeabilidade_cms', 0):.2e} cm/s "
            + ("(estimada por referência)" if st.session_state.get("perm_is_estimated")
               else "(informada pelo usuário)")
        ),
        "Fonte da permeabilidade": st.session_state.get("perm_fonte", "—"),
        "Horizonte":    st.session_state.get("horizon_label", "—"),
        "Frequência":   f"~{st.session_state.get('freq_hours', 24):.1f}h entre leituras",
        "Tuning":       "Ativo" if st.session_state.get("enable_tuning") else "Desligado",
    }.items():
        st.markdown(f"&nbsp;&nbsp;• **{k}:** {v}")
    st.markdown('</div>', unsafe_allow_html=True)

    run_btn = st.button("🚀 Gerar Previsão", type="primary", use_container_width=True)

    if run_btn:
        progress = st.progress(0, "Iniciando...")
        status   = st.empty()

        try:
            df            = st.session_state["df_raw"].copy()
            freq_hours    = st.session_state["freq_hours"]
            inst_type_str = st.session_state["inst_type"]
            tipo_material = st.session_state["tipo_material"]
            k_cms         = st.session_state["permeabilidade_cms"]
            n_days        = st.session_state["n_days_horizon"]
            enable_tuning = st.session_state.get("enable_tuning", True)
            enable_unc    = st.session_state.get("enable_uncertainty", True)

            # Injetar metadados do instrumento
            df = inject_instrument_constants(
                df, inst_type_str, tipo_material, k_cms
            )

            # Merge de dados externos
            df_pluv = st.session_state.get("df_pluv")
            if df_pluv is not None:
                df, _ = merge_external_data(df, df_pluv)

            df_reserv = st.session_state.get("df_reserv")
            if df_reserv is not None:
                df, _ = merge_external_data(df, df_reserv)

            # Validação de negócio (bloqueia se faltarem obrigatórias)
            try:
                biz_warnings = validate_business_requirements(df)
                for w in (biz_warnings or []):
                    _warn(w)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            progress.progress(10, "Pré-processamento...")
            prep = run_preprocessing(df)
            df_proc, encoders = prep.df, prep.encoders

            progress.progress(22, "Controle de qualidade e detecção de anomalias...")
            from config import MissingValueConfig
            miss_cfg = MissingValueConfig()
            df_proc, quality_report  = treat_missing_values(df_proc, cfg=miss_cfg)
            df_proc, outlier_summary = detect_outliers(df_proc)
            df_proc = apply_outlier_filter(df_proc, mode="flag_only")

            progress.progress(35, "Engenharia de atributos (identidade do instrumento)...")
            inst_enum = InstrumentType(inst_type_str)
            feat_cfg  = get_feature_config_for_instrument(inst_enum, freq_hours)
            df_feat   = build_features(df_proc, cfg=feat_cfg)
            feature_cols = get_feature_columns(df_feat)

            progress.progress(50, f"Treinamento XGBoost{' + tuning Optuna' if enable_tuning else ''}...")
            tuning_cfg = TuningConfig(enabled=enable_tuning, n_trials=20, timeout_seconds=90)
            result = run_training(
                df_feat, feature_cols,
                tuning_cfg=tuning_cfg,
                cv_cfg=DEFAULT_CV,
                run_cv=True,
                run_tuning=enable_tuning,
            )

            progress.progress(72, "Calculando SHAP values...")
            X_for_shap   = result.test_df[feature_cols].fillna(0)
            shap_global  = compute_shap_values(result.model, X_for_shap, feature_cols)
            st.session_state["shap_global"] = shap_global

            progress.progress(80, "Gerando previsões futuras...")
            instruments      = sorted(df_feat["instrumento"].unique())
            all_forecasts    = {}
            all_uncertainties = {}

            for inst in instruments:
                try:
                    fc = recursive_forecast(
                        model=result.model,
                        df_history=df_feat,
                        feature_cols=feature_cols,
                        instrumento=inst,
                        n_days=n_days,
                        freq_hours=freq_hours,
                        cfg=feat_cfg,
                        X_train=result.test_df[feature_cols],
                    )
                    all_forecasts[inst] = fc

                    if enable_unc:
                        unc = compute_forecast_uncertainty(
                            result.model, df_feat, feature_cols, inst,
                            n_steps=min(fc["n_steps"], 60),
                            n_bootstrap=15,
                        )
                        all_uncertainties[inst] = unc
                except Exception as e:
                    st.warning(f"Previsão não gerada para '{inst}': {e}")

            # SHAP por instrumento
            shap_local = {}
            for inst in instruments:
                df_inst = df_feat[df_feat["instrumento"] == inst]
                sl = compute_shap_for_instrument(result.model, df_inst, feature_cols)
                if sl:
                    shap_local[inst] = sl
            st.session_state["shap_local"] = shap_local

            progress.progress(100, "Concluído!")

            st.session_state.update({
                "df_featured": df_feat, "feature_cols": feature_cols,
                "encoders": encoders, "training_result": result,
                "all_forecasts": all_forecasts,
                "all_uncertainties": all_uncertainties,
                "pipeline_done": True, "training_done": True, "forecast_done": True,
                "feat_cfg": feat_cfg,
            })
            status.success("✅ Previsão gerada com sucesso!")

        except Exception as e:
            progress.empty()
            st.error(f"Erro durante a execução: {e}")
            import traceback
            with st.expander("Detalhes técnicos do erro"):
                st.code(traceback.format_exc(), language="python")

    # ===========================================================================
    # RESULTADOS
    # ===========================================================================

    if st.session_state.get("forecast_done"):
        result            = st.session_state["training_result"]
        df_feat           = st.session_state["df_featured"]
        feature_cols      = st.session_state["feature_cols"]
        all_forecasts     = st.session_state.get("all_forecasts", {})
        all_uncertainties = st.session_state.get("all_uncertainties", {})
        shap_global       = st.session_state.get("shap_global")
        shap_local        = st.session_state.get("shap_local", {})
        freq_hours        = st.session_state["freq_hours"]
        n_days            = st.session_state["n_days_horizon"]
        inst_type_str     = st.session_state.get("inst_type", "")

        st.divider()
        st.markdown("## 📊 Resultados")

        instruments   = sorted(df_feat["instrumento"].unique())
        selected_inst = st.selectbox(
            "Instrumento para análise:", instruments, key="res_inst"
        )

        # KPIs do instrumento
        summ = instrument_summary(df_feat, selected_inst)
        c1, c2, c3, c4 = st.columns(4)
        for col, (lab, val) in zip(
            [c1, c2, c3, c4],
            [
                ("N leituras hist.", f"{summ.get('n_leituras', '—'):,}"),
                ("Período", f"{summ.get('data_inicio', '—')} → {summ.get('data_fim', '—')}"),
                ("Média histórica", f"{summ.get('leitura_media', '—')}"),
                ("Horizonte", st.session_state.get("horizon_label", "—")),
            ],
        ):
            with col:
                st.markdown(_kpi(lab, val), unsafe_allow_html=True)

        st.markdown("")

        # Avisos do treinamento
        for w in result.warnings:
            _warn(w)

        # Métricas globais
        st.markdown("### 📐 Desempenho do Modelo (conjunto de teste)")
        mt = result.metrics_test
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, (lab, val) in zip(
            [mc1, mc2, mc3, mc4],
            [
                ("MAE", mt.get("MAE", "—")),
                ("RMSE", mt.get("RMSE", "—")),
                ("R²", mt.get("R²", "—")),
                ("lr otimizado", f"{result.best_params.get('learning_rate', 0):.3f}"
                 if result.best_params and result.tuning_enabled else "—"),
            ],
        ):
            with col:
                st.markdown(_kpi(lab, val), unsafe_allow_html=True)

        # Hiperparâmetros
        if result.best_params:
            with st.expander("🔧 Hiperparâmetros utilizados"):
                origin = "Otimizados pelo Optuna" if result.tuning_enabled else "Padrão do sistema"
                st.caption(f"Origem: {origin}")
                params_df = pd.DataFrame(
                    list(result.best_params.items()), columns=["Parâmetro", "Valor"]
                )
                st.dataframe(params_df, use_container_width=True, hide_index=True)

        # CV Temporal
        if result.cv_results and "aggregated" in result.cv_results:
            st.markdown("### 🔄 Validação Cruzada Temporal")
            agg = result.cv_results["aggregated"]
            _info(
                f"CV com {agg.get('n_folds', '?')} folds: "
                f"RMSE = **{agg.get('RMSE_mean', '—')} ± {agg.get('RMSE_std', '—')}** | "
                f"R² = **{agg.get('R2_mean', '—')} ± {agg.get('R2_std', '—')}**"
            )
            cv_disp = format_cv_results(result.cv_results)
            st.dataframe(cv_disp, use_container_width=True, hide_index=True)
            st.plotly_chart(plot_cv_results(result.cv_results), use_container_width=True)

        # Gráfico principal
        st.markdown("### 📈 Previsão")
        df_inst   = df_feat[df_feat["instrumento"] == selected_inst]
        test_inst = (
            result.test_df[result.test_df["instrumento"] == selected_inst]
            if "instrumento" in result.test_df.columns
            else result.test_df
        )
        envelope    = compute_historical_envelope(df_feat, selected_inst)
        fc_data     = all_forecasts.get(selected_inst)
        unc_data    = all_uncertainties.get(selected_inst)
        forecast_df = fc_data["forecast_df"] if fc_data else pd.DataFrame()

        # Alertas
        if fc_data:
            for w in fc_data.get("reliability_warnings", []):
                _warn(w)
            for a in fc_data.get("extrapolation_alerts", []):
                _warn(a)
        if not forecast_df.empty and envelope:
            for a in check_forecast_vs_envelope(forecast_df, envelope):
                _warn(a)
            for a in check_physical_plausibility(forecast_df, envelope):
                _warn(a)

        fig_main = plot_forecast_final(
            df_history=df_inst,
            forecast_df=forecast_df,
            instrumento=selected_inst,
            tipo_instrumento=inst_type_str,
            n_days=n_days,
            freq_hours=freq_hours,
            envelope=envelope,
            test_df=test_inst,
            uncertainty_df=unc_data,
        )
        st.plotly_chart(fig_main, use_container_width=True)

        # Tabela de previsões
        if not forecast_df.empty:
            with st.expander("📋 Tabela de Previsões Futuras"):
                fc_disp = forecast_df.copy()
                fc_disp["data"] = (
                    fc_disp["data"].dt.strftime("%d/%m/%Y %H:%M")
                    if freq_hours < 24 else fc_disp["data"].dt.date
                )
                fc_disp["previsao"] = fc_disp["previsao"].round(4)
                st.dataframe(
                    fc_disp[["passo", "dia_equivalente", "data", "previsao"]].rename(
                        columns={
                            "passo": "Passo", "dia_equivalente": "Dia equiv.",
                            "data": "Data/Hora", "previsao": "Previsão",
                        }
                    ),
                    use_container_width=True, hide_index=True,
                )

        # Análise de resíduos
        with st.expander("📉 Análise de Resíduos"):
            c_l, c_r = st.columns(2)
            with c_l:
                st.plotly_chart(
                    plot_obs_vs_pred(test_inst, selected_inst), use_container_width=True
                )
            with c_r:
                st.plotly_chart(
                    plot_scatter(test_inst, selected_inst), use_container_width=True
                )
            st.plotly_chart(
                plot_residuals(test_inst, selected_inst), use_container_width=True
            )
            st.plotly_chart(
                plot_residual_distribution(test_inst, selected_inst),
                use_container_width=True,
            )

        # SHAP
        st.markdown("### 🔍 Interpretabilidade — SHAP")
        tab_s1, tab_s2 = st.tabs(
            ["Global (todos os instrumentos)", f"Local ({selected_inst})"]
        )
        with tab_s1:
            if shap_global:
                _info(
                    "Importância SHAP: quantifica a contribuição média de cada variável "
                    "para as previsões do modelo. Mais robusto que o Gain do XGBoost "
                    "para interpretação causal."
                )
                st.plotly_chart(
                    plot_shap_summary(shap_global), use_container_width=True
                )
            else:
                _warn(
                    "SHAP não disponível. Instale com: <code>pip install shap</code>",
                )
        with tab_s2:
            sl = shap_local.get(selected_inst)
            if sl:
                _info(
                    f"SHAP local: contribuição das variáveis nas últimas "
                    f"{sl.get('n_samples', '?')} leituras do instrumento {selected_inst}."
                )
                st.plotly_chart(
                    plot_shap_local(sl, title=selected_inst), use_container_width=True
                )
            else:
                _info("SHAP local não disponível para este instrumento.")

        # Feature importance (Gain)
        with st.expander("🏆 Importância das Features — Gain XGBoost"):
            st.plotly_chart(
                plot_feature_importance(result.feature_importance),
                use_container_width=True,
            )

        # Métricas por instrumento
        with st.expander("📊 Desempenho por Instrumento"):
            per_inst = metrics_per_instrument(result.test_df)
            st.dataframe(per_inst, use_container_width=True, hide_index=True)
            st.plotly_chart(
                plot_metrics_per_instrument(per_inst), use_container_width=True
            )

        # ---- Exportação Excel ----
        st.markdown("### 💾 Exportação dos Resultados")
        _info(
            "O arquivo Excel contém: Resumo Executivo, Previsão, Histórico + Ajuste, "
            "Métricas, Validação Cruzada, Hiperparâmetros, SHAP e Alertas."
        )

        if st.button("⬇️ Gerar arquivo Excel completo", use_container_width=True):
            try:
                output    = io.BytesIO()
                ts        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                data_hoje = datetime.date.today()

                with pd.ExcelWriter(output, engine="openpyxl") as writer:

                    # 1. Resumo executivo
                    resumo = [
                        ("Sistema",               "PiezoPrev — Versão Final"),
                        ("Timestamp execução",     ts),
                        ("Instrumento analisado",  selected_inst),
                        ("Tipo de instrumento",    inst_type_str),
                        ("Solo / Material",        st.session_state.get("tipo_material", "")),
                        ("Permeabilidade (cm/s)",  st.session_state.get("permeabilidade_cms", "")),
                        ("Origem permeabilidade",  "Estimada por referência"
                         if st.session_state.get("perm_is_estimated") else "Informada pelo usuário"),
                        ("Fonte permeabilidade",   st.session_state.get("perm_fonte", "")),
                        ("Observação permeab.",    st.session_state.get("perm_observacao", "")),
                        ("Horizonte de previsão",  st.session_state.get("horizon_label", "")),
                        ("Dias previstos",         n_days),
                        ("Frequência (h)",         freq_hours),
                        ("N passos previstos",     fc_data["n_steps"] if fc_data else "—"),
                        ("MAE teste",              result.metrics_test.get("MAE")),
                        ("RMSE teste",             result.metrics_test.get("RMSE")),
                        ("R² teste",               result.metrics_test.get("R²")),
                        ("MAPE teste (%)",         result.metrics_test.get("MAPE (%)")),
                        ("Tuning ativo",           "Sim" if result.tuning_enabled else "Não"),
                        ("N amostras treino",      result.n_train),
                        ("N amostras teste",       result.n_test),
                    ]
                    pd.DataFrame(resumo, columns=["Item", "Valor"]).to_excel(
                        writer, sheet_name="Resumo_Executivo", index=False
                    )

                    # 2. Previsão futura
                    if fc_data and not fc_data["forecast_df"].empty:
                        fc_exp = fc_data["forecast_df"].copy()
                        fc_exp["instrumento"]      = selected_inst
                        fc_exp["tipo"]             = inst_type_str
                        fc_exp["horizonte"]        = st.session_state.get("horizon_label", "")
                        fc_exp["freq_horas"]       = freq_hours
                        fc_exp["tipo_material"]    = st.session_state.get("tipo_material", "")
                        fc_exp["permeab_cms"]      = st.session_state.get("permeabilidade_cms", "")
                        fc_exp["fonte_permeab"]    = st.session_state.get("perm_fonte", "")
                        fc_exp["exec_timestamp"]   = ts
                        fc_exp.to_excel(writer, sheet_name="Previsao", index=False)

                    # 3. Histórico + ajuste
                    test_exp = result.test_df.copy()
                    if "instrumento" not in test_exp.columns:
                        test_exp["instrumento"] = "—"
                    test_exp[["instrumento", "data", "leitura", "previsao"]].to_excel(
                        writer, sheet_name="Historico_Ajuste", index=False
                    )

                    # 4. Métricas globais
                    format_metrics_table(result.metrics_test).to_excel(
                        writer, sheet_name="Metricas_Modelo", index=False
                    )

                    # 5. Métricas por instrumento
                    metrics_per_instrument(result.test_df).to_excel(
                        writer, sheet_name="Metricas_por_Instrumento", index=False
                    )

                    # 6. Validação cruzada temporal
                    cv_df = format_cv_results(result.cv_results)
                    if not cv_df.empty:
                        cv_df.to_excel(
                            writer, sheet_name="Validacao_Cruzada", index=False
                        )

                    # 7. Hiperparâmetros
                    if result.best_params:
                        params_exp = pd.DataFrame(
                            list(result.best_params.items()),
                            columns=["Parâmetro", "Valor"],
                        )
                        params_exp["origem"] = (
                            "Optuna" if result.tuning_enabled else "Padrão"
                        )
                        params_exp.to_excel(
                            writer, sheet_name="Hiperparametros", index=False
                        )

                    # 8. SHAP
                    if shap_global and "shap_df" in shap_global:
                        shap_global["shap_df"].to_excel(
                            writer, sheet_name="SHAP_Global", index=False
                        )
                    elif result.feature_importance is not None:
                        result.feature_importance.to_excel(
                            writer, sheet_name="Feature_Importance", index=False
                        )

                    # 9. Alertas e mensagens
                    alertas = []
                    for w in result.warnings:
                        alertas.append({"Tipo": "Modelo", "Mensagem": w})
                    if fc_data:
                        for w in fc_data.get("reliability_warnings", []):
                            alertas.append({
                                "Tipo": "Confiabilidade forecast", "Mensagem": w
                            })
                        for a in fc_data.get("extrapolation_alerts", []):
                            alertas.append({"Tipo": "Extrapolação", "Mensagem": a})
                    if envelope:
                        for a in check_forecast_vs_envelope(forecast_df, envelope):
                            alertas.append({
                                "Tipo": "Envelope histórico", "Mensagem": a
                            })
                        for a in check_physical_plausibility(forecast_df, envelope):
                            alertas.append({
                                "Tipo": "Plausibilidade física", "Mensagem": a
                            })
                    if alertas:
                        pd.DataFrame(alertas).to_excel(
                            writer, sheet_name="Alertas", index=False
                        )

                output.seek(0)
                fname = (
                    f"piezoprev_{selected_inst}_{n_days}d_{data_hoje}.xlsx"
                )
                st.download_button(
                    "📥 Baixar Excel",
                    output.getvalue(),
                    fname,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(f"Erro ao gerar arquivo Excel: {e}")
                import traceback
                with st.expander("Detalhes"):
                    st.code(traceback.format_exc())

        st.divider()
        if st.button("🔄 Nova análise (reiniciar)", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    _nav_buttons(back_step=7)

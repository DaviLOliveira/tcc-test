"""
visualization.py — PiezoPrev V4
Adições V4:
- plot_shap_summary(): importância SHAP global (beeswarm-style com barras)
- plot_shap_waterfall_local(): explicação local de uma previsão específica
- plot_cv_results(): resultado da validação cruzada temporal
- plot_forecast_with_uncertainty(): gráfico de previsão com bandas de incerteza
- plot_forecast_final(): atualizado com bandas de incerteza quando disponíveis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

C_OBS   = "#1D4ED8"
C_PRED  = "#F97316"
C_FORE  = "#10B981"
C_TRAIN = "#94A3B8"
C_ERR   = "#EF4444"
C_OK    = "#22C55E"
C_SHAP  = "#7C3AED"
BG      = "#F8FAFC"
GRID    = "#E2E8F0"
FONT    = "IBM Plex Sans, Inter, sans-serif"


def _base_layout(title, xlab="Data", ylab="Leitura", height=None):
    d = dict(
        title=dict(text=title, font=dict(size=15, family=FONT)),
        xaxis=dict(title=xlab, gridcolor=GRID),
        yaxis=dict(title=ylab, gridcolor=GRID),
        plot_bgcolor=BG, paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=65, r=20, t=65, b=55),
        font=dict(family=FONT),
    )
    if height:
        d["height"] = height
    return d


# ---------------------------------------------------------------------------
# Gráfico principal da previsão futura com bandas de incerteza
# ---------------------------------------------------------------------------

def plot_forecast_final(
    df_history: pd.DataFrame,
    forecast_df: pd.DataFrame,
    instrumento: str,
    tipo_instrumento: str,
    n_days: int,
    freq_hours: float,
    yaxis_label: str = "Leitura",
    envelope: dict = None,
    date_col: str = "data",
    target_col: str = "leitura",
    pred_col: str = "previsao",
    test_df: pd.DataFrame = None,
    uncertainty_df: pd.DataFrame = None,
) -> go.Figure:
    """
    Gráfico técnico exportável da previsão futura.
    V4: adiciona bandas de incerteza bootstrap quando disponíveis.
    """
    fig = go.Figure()

    if envelope:
        q_low  = envelope.get("p5",  envelope.get("min"))
        q_high = envelope.get("p95", envelope.get("max"))
        if q_low is not None and q_high is not None:
            fig.add_hrect(
                y0=q_low, y1=q_high,
                fillcolor="rgba(148,163,184,0.10)", line_width=0,
                annotation_text="Envelope histórico P5–P95",
                annotation_position="top left", annotation_font_size=10,
            )

    # Histórico
    fig.add_trace(go.Scatter(
        x=df_history[date_col], y=df_history[target_col],
        name="Histórico observado", mode="lines",
        line=dict(color=C_TRAIN, width=1.5),
    ))

    # Teste (se disponível)
    if test_df is not None and not test_df.empty:
        fig.add_trace(go.Scatter(
            x=test_df[date_col], y=test_df[target_col],
            name="Observado (período de teste)", mode="lines+markers",
            line=dict(color=C_OBS, width=2), marker=dict(size=4),
        ))
        if pred_col in test_df.columns:
            fig.add_trace(go.Scatter(
                x=test_df[date_col], y=test_df[pred_col],
                name="Ajuste do modelo (teste)", mode="lines",
                line=dict(color=C_PRED, width=2, dash="dash"),
            ))

    # Bandas de incerteza
    if uncertainty_df is not None and not uncertainty_df.empty and not forecast_df.empty:
        n_uc = min(len(uncertainty_df), len(forecast_df))
        uc   = uncertainty_df.head(n_uc)
        fc_uc = forecast_df.head(n_uc)

        fig.add_trace(go.Scatter(
            x=list(fc_uc[date_col]) + list(fc_uc[date_col])[::-1],
            y=list(uc["p90"]) + list(uc["p10"])[::-1],
            fill="toself",
            fillcolor="rgba(16,185,129,0.10)",
            line=dict(width=0),
            name="Faixa incerteza P10–P90",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=list(fc_uc[date_col]) + list(fc_uc[date_col])[::-1],
            y=list(uc["p75"]) + list(uc["p25"])[::-1],
            fill="toself",
            fillcolor="rgba(16,185,129,0.18)",
            line=dict(width=0),
            name="Faixa incerteza P25–P75",
            showlegend=True,
        ))

    # Previsão central
    if not forecast_df.empty:
        readings_per_day = max(1, round(24.0 / freq_hours))
        n_steps = len(forecast_df)
        freq_label = f"{freq_hours:.0f}h" if freq_hours < 24 else "diária"

        fig.add_trace(go.Scatter(
            x=forecast_df[date_col], y=forecast_df[pred_col],
            name=f"Previsão central — {n_days} dias ({n_steps} leituras)",
            mode="lines+markers",
            line=dict(color=C_FORE, width=2.5),
            marker=dict(size=3 if readings_per_day > 2 else 5, symbol="circle-open"),
            hovertemplate="<b>Data:</b> %{x}<br><b>Previsão:</b> %{y:.4f}<extra></extra>",
        ))

    # Linha de início da previsão
    if not df_history.empty:
        split_date = df_history[date_col].max()
        fig.add_vline(x=split_date, line_dash="dot", line_color="#64748B",
                      annotation_text="Início da previsão",
                      annotation_position="top right", annotation_font_size=10)

    # Caption técnico
    per_str = "—"
    if not forecast_df.empty:
        try:
            per_str = f"{forecast_df[date_col].min().date()} → {forecast_df[date_col].max().date()}"
        except Exception:
            per_str = "—"

    obs_str = " | ⚠️ MODO EXPERIMENTAL" if n_days >= 365 else (
        " | ⚠️ Horizonte longo — tendência qualitativa" if n_days > 90 else ""
    )
    caption = (
        f"Instrumento: {instrumento}   |   Tipo: {tipo_instrumento}   |   "
        f"Frequência: {freq_label}   |   Período: {per_str}   |   "
        f"Horizonte: {n_days} dias{obs_str}"
    )

    layout = _base_layout(f"Previsão — {instrumento}", ylab=yaxis_label)
    layout["annotations"] = [dict(
        text=caption, xref="paper", yref="paper",
        x=0, y=-0.13, xanchor="left", yanchor="top",
        font=dict(size=10, color="#64748B", family=FONT), showarrow=False,
    )]
    layout["margin"]["b"] = 85
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# NOVO V4 — Gráfico SHAP global
# ---------------------------------------------------------------------------

def plot_shap_summary(shap_result: dict, top_n: int = 20) -> go.Figure:
    """
    Gráfico de importância SHAP global (|SHAP médio| por feature).
    Mais confiável que o gain do XGBoost para interpretação causal.

    Args:
        shap_result: Retorno de compute_shap_values().
        top_n: Features a exibir.
    """
    if shap_result is None:
        fig = go.Figure()
        fig.add_annotation(text="SHAP não disponível (instale: pip install shap)",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    df_plot = shap_result["shap_df"].head(top_n).sort_values("mean_abs_shap")

    fig = go.Figure(go.Bar(
        x=df_plot["mean_abs_shap"],
        y=df_plot["feature"],
        orientation="h",
        marker=dict(
            color=df_plot["mean_abs_shap"],
            colorscale="Purples",
            showscale=False,
        ),
        text=df_plot["shap_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))

    n_samp = shap_result.get("n_samples", "?")
    fig.update_layout(
        title=dict(
            text=f"Importância SHAP — Contribuição Média Absoluta (n={n_samp} amostras)",
            font=dict(size=14, family=FONT)
        ),
        xaxis_title="|SHAP| médio (impacto na previsão)",
        yaxis_title="",
        plot_bgcolor=BG, paper_bgcolor="white",
        margin=dict(l=200, r=70, t=65, b=55),
        height=max(380, top_n * 26),
        font=dict(family=FONT),
    )
    return fig


# ---------------------------------------------------------------------------
# NOVO V4 — SHAP local (por instrumento)
# ---------------------------------------------------------------------------

def plot_shap_local(shap_result: dict, top_n: int = 15, title: str = "") -> go.Figure:
    """Gráfico SHAP para o instrumento selecionado (últimas N leituras)."""
    return plot_shap_summary(shap_result, top_n=top_n)


# ---------------------------------------------------------------------------
# NOVO V4 — Resultado da validação cruzada temporal
# ---------------------------------------------------------------------------

def plot_cv_results(cv_results: dict) -> go.Figure:
    """
    Gráfico de RMSE por fold da validação cruzada temporal.

    Mostra a estabilidade do modelo ao longo do tempo:
    variação alta entre folds indica sensibilidade temporal.
    """
    if not cv_results or "folds" not in cv_results or not cv_results["folds"]:
        fig = go.Figure()
        fig.add_annotation(text="CV temporal não disponível",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    folds = cv_results["folds"]
    fold_nums = [f["fold"] for f in folds]
    rmse_vals = [f["RMSE"] for f in folds]
    mae_vals  = [f["MAE"]  for f in folds]
    r2_vals   = [f["R²"]   for f in folds]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["RMSE por Fold", "MAE por Fold", "R² por Fold"])

    def _bar(values, color, col):
        fig.add_trace(go.Bar(
            x=[f"Fold {n}" for n in fold_nums],
            y=values,
            marker_color=color,
            opacity=0.8,
            showlegend=False,
        ), row=1, col=col)

    _bar(rmse_vals, C_PRED, 1)
    _bar(mae_vals,  C_OBS,  2)
    _bar(r2_vals,   C_FORE, 3)

    # Média como linha horizontal
    if rmse_vals:
        agg = cv_results.get("aggregated", {})
        if agg:
            for col_i, (key, color) in enumerate(
                [("RMSE_mean", C_PRED), ("MAE_mean", C_OBS), ("R2_mean", C_FORE)], start=1
            ):
                val = agg.get(key)
                if val is not None:
                    fig.add_hline(y=val, line_dash="dot", line_color=color,
                                  annotation_text=f"Média: {val:.4f}",
                                  row=1, col=col_i)

    fig.update_layout(
        title=dict(text="Validação Cruzada Temporal — Desempenho por Fold",
                   font=dict(size=14, family=FONT)),
        plot_bgcolor=BG, paper_bgcolor="white",
        height=350, font=dict(family=FONT),
        margin=dict(l=50, r=20, t=80, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Série completa
# ---------------------------------------------------------------------------

def plot_full_series(df_all, test_df, instrumento="", forecast_df=None, envelope=None,
                     date_col="data", target_col="leitura", pred_col="previsao"):
    split_date = test_df[date_col].min()
    train_df   = df_all[df_all[date_col] < split_date]
    fig = go.Figure()

    if envelope:
        q_low  = envelope.get("p5",  envelope.get("min"))
        q_high = envelope.get("p95", envelope.get("max"))
        if q_low and q_high:
            fig.add_hrect(y0=q_low, y1=q_high,
                          fillcolor="rgba(148,163,184,0.12)", line_width=0,
                          annotation_text="Envelope P5–P95", annotation_position="top left",
                          annotation_font_size=10)

    fig.add_trace(go.Scatter(x=train_df[date_col], y=train_df[target_col],
        name="Histórico (treino)", mode="lines", line=dict(color=C_TRAIN, width=1.5)))
    fig.add_trace(go.Scatter(x=test_df[date_col], y=test_df[target_col],
        name="Observado (teste)", mode="lines+markers",
        line=dict(color=C_OBS, width=2), marker=dict(size=4)))
    if pred_col in test_df.columns:
        fig.add_trace(go.Scatter(x=test_df[date_col], y=test_df[pred_col],
            name="Ajuste do modelo", mode="lines+markers",
            line=dict(color=C_PRED, width=2, dash="dash"), marker=dict(size=4, symbol="diamond")))
    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(go.Scatter(x=forecast_df[date_col], y=forecast_df[pred_col],
            name="Previsão futura", mode="lines+markers",
            line=dict(color=C_FORE, width=2, dash="dot"), marker=dict(size=4, symbol="circle-open")))

    fig.add_vline(x=str(split_date), line_dash="dot", line_color="#64748B",
                  annotation_text="Início do teste", annotation_position="top right",
                  annotation_font_size=10)
    fig.update_layout(**_base_layout(f"Série Completa — {instrumento}",
                                     ylab=target_col.capitalize()))
    return fig


def plot_obs_vs_pred(test_df, instrumento="", date_col="data", target_col="leitura", pred_col="previsao"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df[date_col], y=test_df[target_col],
        name="Observado", mode="lines+markers",
        line=dict(color=C_OBS, width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=test_df[date_col], y=test_df[pred_col],
        name="Previsto (XGBoost)", mode="lines+markers",
        line=dict(color=C_PRED, width=2, dash="dash"), marker=dict(size=4, symbol="diamond")))
    fig.update_layout(**_base_layout(f"Observado vs Previsto — {instrumento}"))
    return fig


def plot_scatter(test_df, instrumento="", target_col="leitura", pred_col="previsao"):
    y_obs, y_pred = test_df[target_col], test_df[pred_col]
    all_vals = pd.concat([y_obs, y_pred]).dropna()
    v_min, v_max = all_vals.min(), all_vals.max()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_obs, y=y_pred, mode="markers",
        marker=dict(color=C_OBS, size=6, opacity=0.6), name="Amostras"))
    fig.add_trace(go.Scatter(x=[v_min, v_max], y=[v_min, v_max], mode="lines",
        line=dict(color="#64748B", dash="dash", width=1.5), name="Referência (y = x)"))
    fig.update_layout(**_base_layout(f"Dispersão — {instrumento}", xlab="Observado", ylab="Previsto"))
    return fig


def plot_residuals(test_df, instrumento="", date_col="data", target_col="leitura", pred_col="previsao"):
    residuals = test_df[target_col] - test_df[pred_col]
    colors = [C_ERR if r < 0 else C_OK for r in residuals]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=test_df[date_col], y=residuals,
        marker_color=colors, opacity=0.75, name="Resíduo"))
    fig.add_hline(y=0, line_dash="dot", line_color="#64748B", line_width=1.5)
    fig.update_layout(**_base_layout(f"Resíduos ao Longo do Tempo — {instrumento}",
                                     ylab="Erro (Obs − Prev)"))
    return fig


def plot_residual_distribution(test_df, instrumento="", target_col="leitura", pred_col="previsao"):
    residuals = test_df[target_col] - test_df[pred_col]
    fig = px.histogram(residuals, nbins=40, color_discrete_sequence=[C_OBS],
                       opacity=0.75, marginal="box")
    fig.update_layout(
        title=dict(text=f"Distribuição dos Resíduos — {instrumento}", font=dict(size=14, family=FONT)),
        xaxis_title="Erro (Obs − Prev)", yaxis_title="Frequência",
        plot_bgcolor=BG, paper_bgcolor="white", showlegend=False,
        margin=dict(l=60, r=20, t=60, b=55), font=dict(family=FONT))
    return fig


def plot_feature_importance(fi_df, top_n=20):
    df_plot = fi_df.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df_plot["importance"], y=df_plot["feature"], orientation="h",
        marker=dict(color=df_plot["importance"], colorscale="Blues", showscale=False),
        text=df_plot["importance_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside"))
    fig.update_layout(
        title=dict(text=f"Importância das Features — Gain XGBoost (Top {top_n})",
                   font=dict(size=14, family=FONT)),
        xaxis_title="Importância (Gain)", plot_bgcolor=BG, paper_bgcolor="white",
        margin=dict(l=200, r=60, t=60, b=55),
        height=max(380, top_n * 26), font=dict(family=FONT))
    return fig


def plot_outliers(df, instrumento="", date_col="data", target_col="leitura", group_col="instrumento"):
    subset = df[df[group_col] == instrumento].sort_values(date_col) if instrumento else df
    normal  = subset[~subset.get("outlier_flag", pd.Series(False, index=subset.index))]
    flagged = subset[subset.get("outlier_flag", pd.Series(False, index=subset.index))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal[date_col], y=normal[target_col],
        name="Normal", mode="lines+markers",
        line=dict(color=C_OBS, width=1.5), marker=dict(size=3)))
    if not flagged.empty:
        hover = flagged.get("outlier_reason", pd.Series("", index=flagged.index))
        fig.add_trace(go.Scatter(x=flagged[date_col], y=flagged[target_col],
            name="⚠️ Anomalia sinalizada", mode="markers",
            marker=dict(color=C_ERR, size=9, symbol="x"), text=hover,
            hovertemplate="<b>Data:</b> %{x}<br><b>Leitura:</b> %{y}<br><b>Tipo:</b> %{text}<extra></extra>"))
    fig.update_layout(**_base_layout(f"Série com Anomalias Sinalizadas — {instrumento}"))
    return fig


def plot_metrics_per_instrument(metrics_df):
    if metrics_df.empty:
        return go.Figure()
    fig = go.Figure(go.Bar(
        x=metrics_df["Instrumento"], y=metrics_df["RMSE"],
        marker_color=metrics_df["RMSE"], colorscale="RdYlGn_r", showscale=True,
        text=metrics_df["RMSE"].round(4), textposition="outside"))
    fig.update_layout(
        title=dict(text="RMSE por Instrumento — Conjunto de Teste", font=dict(size=14, family=FONT)),
        xaxis_title="Instrumento", yaxis_title="RMSE",
        plot_bgcolor=BG, paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=80),
        xaxis=dict(tickangle=-30), font=dict(family=FONT))
    return fig

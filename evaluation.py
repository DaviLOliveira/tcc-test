"""
evaluation.py — PiezoPrev V4
Mudanças V4:
- compute_shap_values(): calcula SHAP values para interpretabilidade técnica
- compute_forecast_uncertainty(): estima propagação de incerteza por horizonte
- check_physical_plausibility(): validação física/geotécnica do resultado
- check_forecast_vs_envelope(): expandido com severidade
- format_cv_results(): formata resultado da CV para exibição
- Funções existentes preservadas e aprimoradas
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Métricas por instrumento
# ---------------------------------------------------------------------------

def metrics_per_instrument(
    test_df: pd.DataFrame,
    target_col: str = "leitura",
    pred_col: str = "previsao",
    group_col: str = "instrumento",
) -> pd.DataFrame:
    rows = []
    for inst, grp in test_df.groupby(group_col):
        y_true = grp[target_col].dropna()
        y_pred = grp.loc[y_true.index, pred_col]
        if len(y_true) < 2:
            continue
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
        mask = y_true.abs() > 1e-9
        mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                if mask.any() else np.nan)
        rows.append({
            "Instrumento": inst,
            "N (teste)": len(y_true),
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R²": round(float(r2), 4) if not np.isnan(r2) else None,
            "MAPE (%)": round(float(mape), 2) if not np.isnan(mape) else None,
        })
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Envelope histórico
# ---------------------------------------------------------------------------

def compute_historical_envelope(
    df: pd.DataFrame,
    instrumento: str,
    target_col: str = "leitura",
    group_col: str = "instrumento",
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> dict:
    subset = df[df[group_col] == instrumento][target_col].dropna()
    if len(subset) == 0:
        return {}
    return {
        "min":  float(subset.min()),
        "max":  float(subset.max()),
        "mean": float(subset.mean()),
        "std":  float(subset.std()),
        f"p{int(quantile_low*100)}":  float(subset.quantile(quantile_low)),
        f"p{int(quantile_high*100)}": float(subset.quantile(quantile_high)),
        "n": int(len(subset)),
    }


def check_forecast_vs_envelope(
    forecast_df: pd.DataFrame,
    envelope: dict,
    pred_col: str = "previsao",
) -> list:
    """
    V4: alertas com severidade e quantificação mais clara.
    """
    alerts = []
    if not envelope or pred_col not in forecast_df.columns:
        return alerts

    q_low  = envelope.get("p5",  envelope.get("min"))
    q_high = envelope.get("p95", envelope.get("max"))
    mean_h = envelope.get("mean", None)
    std_h  = envelope.get("std",  None)
    preds  = forecast_df[pred_col].dropna()
    n_total = len(preds)

    if n_total == 0:
        return alerts

    n_below = int((preds < q_low).sum())
    n_above = int((preds > q_high).sum())

    if n_below > 0:
        pct = 100 * n_below / n_total
        alerts.append(
            f"⚠️ {n_below}/{n_total} previsões ({pct:.0f}%) abaixo do percentil P5 histórico "
            f"(limiar: {q_low:.4f}). Verifique consistência com comportamento observado."
        )
    if n_above > 0:
        pct = 100 * n_above / n_total
        alerts.append(
            f"⚠️ {n_above}/{n_total} previsões ({pct:.0f}%) acima do percentil P95 histórico "
            f"(limiar: {q_high:.4f}). Verifique consistência com comportamento observado."
        )

    # Verificar deriva significativa (previsão média muito distante da média histórica)
    if mean_h is not None and std_h is not None and std_h > 0:
        pred_mean = preds.mean()
        n_sigma = abs(pred_mean - mean_h) / std_h
        if n_sigma > 2:
            alerts.append(
                f"⚠️ Nível médio das previsões ({pred_mean:.4f}) desvia {n_sigma:.1f}σ "
                f"da média histórica ({mean_h:.4f}). Possível extrapolação de tendência."
            )

    return alerts


# ---------------------------------------------------------------------------
# NOVO V4 — SHAP values
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X: pd.DataFrame,
    feature_cols: list,
    max_samples: int = 500,
) -> Optional[dict]:
    """
    Calcula SHAP values para o modelo XGBoost.

    SHAP (SHapley Additive exPlanations) permite explicar quão cada feature
    contribui para cada previsão individual — essencial para defensabilidade
    técnica em engenharia geotécnica.

    Implementa:
    - shap_values: matriz (n_amostras × n_features) com contribuições individuais
    - mean_abs_shap: importância SHAP média (mais confiável que feature importance XGBoost)
    - shap_df: DataFrame formatado para exibição

    Args:
        model: Modelo XGBoost treinado.
        X: Features (subconjunto para cálculo — usa max_samples para performance).
        feature_cols: Lista de features.
        max_samples: Máximo de amostras para SHAP (para performance).

    Returns:
        Dict com shap_values, mean_abs_shap, shap_df, ou None se shap não instalado.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap não instalado. Execute: pip install shap")
        return None

    # Limitar amostras para performance
    X_sample = X[feature_cols].fillna(0)
    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(max_samples, random_state=42)

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)

        if shap_vals is None:
            return None

        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_df = (
            pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_df["shap_pct"] = (shap_df["mean_abs_shap"] / shap_df["mean_abs_shap"].sum() * 100).round(2)

        logger.info(f"SHAP calculado: {len(X_sample)} amostras")
        return {
            "shap_values": shap_vals,
            "X_sample": X_sample,
            "mean_abs_shap": mean_abs,
            "shap_df": shap_df,
            "feature_cols": feature_cols,
            "n_samples": len(X_sample),
        }
    except Exception as e:
        logger.warning(f"Erro ao calcular SHAP: {e}")
        return None


def compute_shap_for_instrument(
    model,
    df_instrument: pd.DataFrame,
    feature_cols: list,
    n_last: int = 50,
) -> Optional[dict]:
    """
    SHAP local: explica as últimas N previsões de um instrumento específico.

    Útil para responder: "por que o modelo previu esse valor para esse instrumento?"

    Args:
        model: Modelo treinado.
        df_instrument: DataFrame do instrumento (já com features).
        feature_cols: Features usadas.
        n_last: Número de amostras recentes a explicar.

    Returns:
        Dict com shap_values e contexto.
    """
    try:
        import shap
    except ImportError:
        return None

    X_inst = df_instrument[feature_cols].fillna(0).tail(n_last)
    if len(X_inst) < 2:
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_inst)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        shap_df   = (
            pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
        return {
            "shap_values": shap_vals,
            "X_sample": X_inst,
            "shap_df": shap_df,
            "n_samples": len(X_inst),
        }
    except Exception as e:
        logger.warning(f"SHAP por instrumento falhou: {e}")
        return None


# ---------------------------------------------------------------------------
# NOVO V4 — Incerteza do forecast recursivo
# ---------------------------------------------------------------------------

def compute_forecast_uncertainty(
    model,
    df_history: pd.DataFrame,
    feature_cols: list,
    instrumento: str,
    n_steps: int,
    target_col: str = "leitura",
    group_col: str = "instrumento",
    n_bootstrap: int = 20,
) -> Optional[pd.DataFrame]:
    """
    Estima incerteza do forecast recursivo via bootstrap nas leituras recentes.

    Estratégia:
    - Perturba as últimas leituras com o resíduo histórico do modelo
    - Para cada perturbação, gera um forecast independente
    - Calcula percentis das previsões ao longo do horizonte

    Isso quantifica como o erro se propaga recursivamente e
    permite exibir uma faixa de incerteza crescente no gráfico.

    Args:
        model: Modelo treinado.
        df_history: Histórico do instrumento (com features).
        feature_cols: Features usadas.
        instrumento: Nome do instrumento.
        n_steps: Número de passos a prever.
        target_col: Coluna alvo.
        group_col: Coluna de instrumento.
        n_bootstrap: Número de replicações bootstrap.

    Returns:
        DataFrame com colunas: passo, p10, p25, p50, p75, p90
        ou None se não for possível calcular.
    """
    from forecasting import _build_step_features
    from config import DEFAULT_FEATURES

    hist = df_history[df_history[group_col] == instrumento].copy()
    hist = hist.sort_values("data").reset_index(drop=True)

    if len(hist) < 30:
        return None

    # Estimar desvio dos resíduos com base nas últimas previsões do teste
    # (usa std da série recente como proxy de erro do modelo)
    recent = hist[target_col].tail(30).dropna()
    sigma_residuo = float(recent.diff().dropna().std())
    if sigma_residuo == 0 or np.isnan(sigma_residuo):
        return None

    freq_td = pd.Timedelta(hours=DEFAULT_FEATURES.freq_hours)
    last_date = hist["data"].max()
    cfg = DEFAULT_FEATURES

    all_series = []

    for _ in range(n_bootstrap):
        working = hist.copy()
        series_preds = []

        for step in range(1, n_steps + 1):
            next_date = last_date + freq_td * step
            try:
                X_step = _build_step_features(
                    working, feature_cols, target_col, group_col,
                    instrumento, next_date, None, cfg
                )
                pred = float(model.predict(X_step)[0])
            except Exception:
                pred = working[target_col].iloc[-1]

            # Adiciona ruído crescente com o horizonte (simula propagação de erro)
            # Escala: ruído proporcional a sqrt(step) para modelar difusão
            noise = np.random.normal(0, sigma_residuo * np.sqrt(step / 10))
            pred_perturbed = pred + noise

            new_row = working.iloc[-1].copy()
            new_row["data"] = next_date
            new_row[target_col] = pred_perturbed
            working = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)
            series_preds.append(pred_perturbed)

        all_series.append(series_preds)

    arr = np.array(all_series)  # (n_bootstrap, n_steps)
    uncertainty_df = pd.DataFrame({
        "passo": range(1, n_steps + 1),
        "p10": np.percentile(arr, 10, axis=0),
        "p25": np.percentile(arr, 25, axis=0),
        "p50": np.percentile(arr, 50, axis=0),
        "p75": np.percentile(arr, 75, axis=0),
        "p90": np.percentile(arr, 90, axis=0),
    })

    return uncertainty_df


# ---------------------------------------------------------------------------
# NOVO V4 — Validação física
# ---------------------------------------------------------------------------

def check_physical_plausibility(
    forecast_df: pd.DataFrame,
    envelope: dict,
    pred_col: str = "previsao",
    max_rate_of_change: Optional[float] = None,
) -> list:
    """
    Verifica plausibilidade física/geotécnica das previsões.

    Checks implementados:
    1. Valores dentro da faixa min/max histórica absoluta (não apenas P5-P95)
    2. Taxa de variação entre leituras consecutivas (gradiente excessivo)
    3. Tendência monotônica suspeita (só sobe ou só desce por muitos passos)

    Args:
        forecast_df: DataFrame de previsões.
        envelope: Envelope histórico do instrumento.
        pred_col: Coluna de previsão.
        max_rate_of_change: Variação máxima aceitável entre passos consecutivos.
                            Se None, usa 3×std histórico.

    Returns:
        Lista de alertas com linguagem técnica.
    """
    alerts = []
    if not envelope or pred_col not in forecast_df.columns:
        return alerts

    preds = forecast_df[pred_col].dropna()
    if len(preds) == 0:
        return alerts

    hist_min = envelope.get("min", None)
    hist_max = envelope.get("max", None)
    hist_std = envelope.get("std", None)

    # 1. Fora do range histórico absoluto
    if hist_min is not None and hist_max is not None:
        n_below_abs = int((preds < hist_min).sum())
        n_above_abs = int((preds > hist_max).sum())
        if n_below_abs > 0:
            alerts.append(
                f"🔴 {n_below_abs} previsões abaixo do mínimo histórico absoluto "
                f"({hist_min:.4f}). Fisicamente improvável sem causa identificada."
            )
        if n_above_abs > 0:
            alerts.append(
                f"🔴 {n_above_abs} previsões acima do máximo histórico absoluto "
                f"({hist_max:.4f}). Fisicamente improvável sem causa identificada."
            )

    # 2. Taxa de variação excessiva
    if hist_std and hist_std > 0:
        max_rc = max_rate_of_change or (3.0 * hist_std)
        diffs = preds.diff().abs().dropna()
        n_steep = int((diffs > max_rc).sum())
        if n_steep > 0:
            alerts.append(
                f"⚠️ {n_steep} transições com variação > {max_rc:.4f} entre passos consecutivos. "
                "Taxa de variação pode ser fisicamente implausível — revisar."
            )

    # 3. Tendência monotônica suspeita (> 20 passos só subindo ou só descendo)
    if len(preds) > 20:
        diffs_sign = np.sign(preds.diff().dropna())
        run_length = (diffs_sign != diffs_sign.shift()).cumsum()
        max_run = run_length.value_counts().max()
        if max_run > 20:
            direction = "crescente" if diffs_sign.iloc[-1] > 0 else "decrescente"
            alerts.append(
                f"ℹ️ Previsão apresenta tendência monotônica {direction} "
                f"por {max_run} passos consecutivos. "
                "Verifique se isso é fisicamente coerente com o contexto da barragem."
            )

    return alerts


# ---------------------------------------------------------------------------
# Formatação de resultados de CV
# ---------------------------------------------------------------------------

def format_cv_results(cv_results: dict) -> pd.DataFrame:
    """
    Formata resultados da validação cruzada temporal para exibição.

    Args:
        cv_results: Dict retornado por run_temporal_cv().

    Returns:
        DataFrame com métricas por fold + linha de média.
    """
    if not cv_results or "folds" not in cv_results:
        return pd.DataFrame()

    folds = cv_results["folds"]
    if not folds:
        return pd.DataFrame()

    rows = []
    for f in folds:
        rows.append({
            "Fold": f.get("fold", "—"),
            "N Treino": f.get("n_train", "—"),
            "N Validação": f.get("n_val", "—"),
            "MAE": f.get("MAE", None),
            "RMSE": f.get("RMSE", None),
            "R²": f.get("R²", None),
        })

    df = pd.DataFrame(rows)

    if "aggregated" in cv_results:
        agg = cv_results["aggregated"]
        media_row = {
            "Fold": "Média ± Std",
            "N Treino": "—",
            "N Validação": "—",
            "MAE":  f"{agg.get('MAE_mean','—')} ± {agg.get('MAE_std','—')}",
            "RMSE": f"{agg.get('RMSE_mean','—')} ± {agg.get('RMSE_std','—')}",
            "R²":   f"{agg.get('R2_mean','—')} ± {agg.get('R2_std','—')}",
        }
        df = pd.concat([df, pd.DataFrame([media_row])], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Formatação de métricas
# ---------------------------------------------------------------------------

def format_metrics_table(metrics: dict) -> pd.DataFrame:
    label_map = {
        "MAE": "MAE — Erro Médio Absoluto",
        "RMSE": "RMSE — Raiz do Erro Quadrático Médio",
        "R²": "R² — Coeficiente de Determinação",
        "MAPE (%)": "MAPE — Erro Percentual Médio Absoluto (%)",
        "n": "N — Amostras Avaliadas",
    }
    return pd.DataFrame([
        {"Métrica": label_map.get(k, k), "Valor": v}
        for k, v in metrics.items() if v is not None
    ])


# ---------------------------------------------------------------------------
# Resumo do instrumento
# ---------------------------------------------------------------------------

def instrument_summary(
    df: pd.DataFrame,
    instrumento: str,
    target_col: str = "leitura",
    group_col: str = "instrumento",
) -> dict:
    subset = df[df[group_col] == instrumento].sort_values("data")
    if subset.empty:
        return {"erro": f"Instrumento '{instrumento}' não encontrado."}
    last = subset.iloc[-1]
    return {
        "instrumento":      instrumento,
        "tipo":             str(last.get("tipo_instrumento", "N/D")),
        "material":         str(last.get("tipo_material", "N/D")),
        "profundidade_m":   last.get("profundidade", None),
        "cota_instalacao_m": last.get("cota_instalacao", None),
        "permeabilidade_m_s": last.get("permeabilidade", None),
        "n_leituras":       int(len(subset)),
        "data_inicio":      str(subset["data"].min().date()),
        "data_fim":         str(subset["data"].max().date()),
        "leitura_min":      round(float(subset[target_col].min()), 4),
        "leitura_max":      round(float(subset[target_col].max()), 4),
        "leitura_media":    round(float(subset[target_col].mean()), 4),
        "leitura_std":      round(float(subset[target_col].std()), 4),
        "pct_ausentes":     round(100 * subset[target_col].isna().sum() / max(len(subset), 1), 2),
    }

"""
training.py — PiezoPrev Final (corrigido)

Correções aplicadas nesta versão:
─────────────────────────────────────────────────────────────────────────────
1. temporal_split()
   - Valida se o DataFrame está vazio ANTES de calcular qualquer índice
   - Verifica se todas as feature_cols existem no DataFrame
   - Adapta test_ratio automaticamente quando há poucos dados
   - Mensagem de erro diagnóstica: mostra n_total, test_ratio, n_treino, n_teste
   - Limiar mínimo de treino configurável (padrão: 30 amostras)

2. run_training()
   - Auditoria de linhas em CADA ETAPA antes do split:
       • total de linhas no df recebido
       • colunas de features faltantes
       • linhas com target nulo
       • linhas com todas as features nulas
   - Emite DataPipelineError com diagnóstico completo quando o df está vazio
   - Adapta test_ratio automaticamente para datasets pequenos (< 200 amostras)
   - Logs claros em cada ponto de controle

3. Classe DataPipelineError
   - Exceção específica para falhas de pipeline de dados
   - Carrega um dicionário de diagnóstico para exibição no Streamlit
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from config import DEFAULT_MODEL, DEFAULT_TUNING, DEFAULT_CV, ModelConfig, TuningConfig, CVConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceção específica de pipeline de dados
# ---------------------------------------------------------------------------

class DataPipelineError(ValueError):
    """
    Exceção lançada quando os dados chegam ao treinamento em estado inválido.
    Carrega um dicionário 'diagnostics' para exibição detalhada no Streamlit.
    """
    def __init__(self, message: str, diagnostics: dict = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


# ---------------------------------------------------------------------------
# Resultado de treinamento
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    model: XGBRegressor
    feature_cols: list
    metrics_test: dict
    metrics_train: dict
    feature_importance: pd.DataFrame
    test_df: pd.DataFrame
    train_period: tuple
    test_period: tuple
    n_train: int                   = 0
    n_test: int                    = 0
    warnings: list                 = field(default_factory=list)
    cv_results: Optional[dict]     = None
    best_params: Optional[dict]    = None
    tuning_enabled: bool           = False


# ---------------------------------------------------------------------------
# Split temporal — com diagnóstico completo
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "leitura",
    date_col: str = "data",
    test_ratio: float = 0.2,
    min_test_samples: int = 15,
    min_train_samples: int = 30,
) -> tuple:
    """
    Divide treino/teste estritamente por ordem temporal.

    Melhorias desta versão:
    - Valida DataFrame vazio antes de qualquer cálculo
    - Verifica colunas de features ausentes
    - Adapta test_ratio se necessário para não zerar o treino
    - Mensagem de erro diagnóstica com todos os valores relevantes

    Args:
        df: DataFrame com features, target e data.
        feature_cols: Colunas de features.
        target_col: Coluna alvo.
        date_col: Coluna de data.
        test_ratio: Fração para teste (últimos N%).
        min_test_samples: Mínimo de amostras no teste.
        min_train_samples: Mínimo absoluto de amostras no treino.

    Returns:
        Tuple (X_train, X_test, y_train, y_test, test_df_raw).

    Raises:
        DataPipelineError: Se os dados forem insuficientes, com diagnóstico completo.
    """
    # ── Guarda 1: DataFrame vazio ──────────────────────────────────────────
    if df is None or len(df) == 0:
        raise DataPipelineError(
            "O DataFrame chegou VAZIO ao treinamento. "
            "O problema está em alguma etapa anterior do pipeline "
            "(merge, feature engineering ou dropna).",
            diagnostics={
                "n_total": 0,
                "causa_provavel": "DataFrame vazio — verifique merge e feature engineering",
            },
        )

    # ── Guarda 2: colunas de features ausentes ─────────────────────────────
    missing_feat_cols = [c for c in feature_cols if c not in df.columns]
    if missing_feat_cols:
        raise DataPipelineError(
            f"Features solicitadas não existem no DataFrame: {missing_feat_cols}. "
            "Verifique se o feature engineering foi executado corretamente.",
            diagnostics={
                "n_total": len(df),
                "features_ausentes": missing_feat_cols,
                "colunas_existentes": list(df.columns),
            },
        )

    # ── Ordenar por data ───────────────────────────────────────────────────
    df = df.sort_values(date_col).reset_index(drop=True)
    n_total = len(df)

    # ── Guarda 3: target completamente nulo ────────────────────────────────
    n_target_null = int(df[target_col].isna().sum())
    if n_target_null == n_total:
        raise DataPipelineError(
            f"A coluna alvo '{target_col}' está completamente nula. "
            "Verifique o carregamento e o merge dos dados.",
            diagnostics={"n_total": n_total, "n_target_null": n_target_null},
        )

    # ── Calcular split ─────────────────────────────────────────────────────
    split_idx = int(n_total * (1.0 - test_ratio))

    # Garantir mínimo no teste
    if n_total - split_idx < min_test_samples:
        split_idx = max(0, n_total - min_test_samples)

    # Garantir mínimo no treino — adaptar test_ratio se necessário
    if split_idx < min_train_samples:
        # Tenta reduzir o teste para salvar o treino
        split_idx = min_train_samples
        n_test_adjusted = n_total - split_idx
        if n_test_adjusted < 5:
            # Dados realmente insuficientes — diagnóstico completo
            raise DataPipelineError(
                f"Dados insuficientes para treino.\n"
                f"  • Total de amostras: {n_total}\n"
                f"  • test_ratio aplicado: {test_ratio:.0%}\n"
                f"  • Amostras de treino calculadas: {split_idx} "
                f"(mínimo exigido: {min_train_samples})\n"
                f"  • Amostras de teste calculadas: {n_total - split_idx} "
                f"(mínimo exigido: {min_test_samples})\n\n"
                f"Causas mais prováveis:\n"
                f"  1. Merge com dados externos gerou muitos NaN e o dropna() "
                f"eliminou quase todos os registros\n"
                f"  2. Lags longos (ex: lag_14) eliminam as primeiras N linhas de cada instrumento\n"
                f"  3. A série histórica do instrumento é curta (< 60 leituras úteis)\n"
                f"  4. test_ratio muito alto para o volume de dados disponível\n\n"
                f"Sugestão: verifique o número de linhas após cada etapa do pipeline "
                f"usando os logs de diagnóstico.",
                diagnostics={
                    "n_total": n_total,
                    "test_ratio": test_ratio,
                    "split_idx": split_idx,
                    "n_train": split_idx,
                    "n_test": n_total - split_idx,
                    "min_train_samples": min_train_samples,
                    "min_test_samples": min_test_samples,
                },
            )
        else:
            logger.warning(
                f"test_ratio={test_ratio:.0%} foi reduzido automaticamente para "
                f"garantir mínimo de {min_train_samples} amostras no treino. "
                f"Treino efetivo: {split_idx} | Teste efetivo: {n_test_adjusted}"
            )

    train_df    = df.iloc[:split_idx]
    test_df_raw = df.iloc[split_idx:].copy()

    logger.info(
        f"Split temporal: n_total={n_total} | "
        f"Treino={len(train_df)} ({train_df[date_col].min().date()} → "
        f"{train_df[date_col].max().date()}) | "
        f"Teste={len(test_df_raw)} ({test_df_raw[date_col].min().date()} → "
        f"{test_df_raw[date_col].max().date()})"
    )

    return (
        train_df[feature_cols],
        test_df_raw[feature_cols],
        train_df[target_col],
        test_df_raw[target_col],
        test_df_raw,
    )


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, label: str = "") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = y_true.abs() > 1e-9
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.any() else np.nan
    )
    metrics = {
        "MAE":      round(float(mae), 4),
        "RMSE":     round(float(rmse), 4),
        "R²":       round(float(r2), 4),
        "MAPE (%)": round(float(mape), 2) if not np.isnan(mape) else None,
        "n":        int(len(y_true)),
    }
    logger.info(f"Métricas [{label}]: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Treinamento XGBoost base
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
    cfg: ModelConfig = DEFAULT_MODEL,
) -> XGBRegressor:
    """Treina XGBoost. Aceita params dict diretamente (do tuning) ou fallback para cfg."""
    if params is None:
        params = {
            "n_estimators":          cfg.n_estimators,
            "learning_rate":         cfg.learning_rate,
            "max_depth":             cfg.max_depth,
            "min_child_weight":      cfg.min_child_weight,
            "subsample":             cfg.subsample,
            "colsample_bytree":      cfg.colsample_bytree,
            "reg_alpha":             cfg.reg_alpha,
            "reg_lambda":            cfg.reg_lambda,
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "random_state":          cfg.random_state,
            "n_jobs":                cfg.n_jobs,
            "eval_metric":           cfg.eval_metric,
        }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_iter = getattr(model, "best_iteration", params.get("n_estimators", 500))
    logger.info(f"Treinamento concluído | Melhor iteração: {best_iter}")
    return model


# ---------------------------------------------------------------------------
# Validação cruzada temporal (TimeSeriesSplit)
# ---------------------------------------------------------------------------

def run_temporal_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: ModelConfig = DEFAULT_MODEL,
    cv_cfg: CVConfig = DEFAULT_CV,
    params: Optional[dict] = None,
) -> dict:
    """Validação cruzada temporal usando TimeSeriesSplit."""
    tscv = TimeSeriesSplit(
        n_splits=cv_cfg.n_splits,
        test_size=cv_cfg.test_size,
        gap=cv_cfg.gap,
    )

    fold_results = []
    logger.info(f"CV temporal: {cv_cfg.n_splits} folds")

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_val   = X.iloc[val_idx]
        y_fold_val   = y.iloc[val_idx]

        if len(X_fold_train) < 20 or len(X_fold_val) < 5:
            continue

        try:
            fold_params = (params or {}).copy()
            fold_params["early_stopping_rounds"] = fold_params.get("early_stopping_rounds", 30)
            model_fold = XGBRegressor(**{
                "n_estimators":          fold_params.get("n_estimators", cfg.n_estimators),
                "learning_rate":         fold_params.get("learning_rate", cfg.learning_rate),
                "max_depth":             fold_params.get("max_depth", cfg.max_depth),
                "min_child_weight":      fold_params.get("min_child_weight", cfg.min_child_weight),
                "subsample":             fold_params.get("subsample", cfg.subsample),
                "colsample_bytree":      fold_params.get("colsample_bytree", cfg.colsample_bytree),
                "reg_alpha":             fold_params.get("reg_alpha", cfg.reg_alpha),
                "reg_lambda":            fold_params.get("reg_lambda", cfg.reg_lambda),
                "early_stopping_rounds": fold_params["early_stopping_rounds"],
                "random_state": cfg.random_state,
                "n_jobs": cfg.n_jobs,
                "eval_metric": cfg.eval_metric,
            })
            model_fold.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False,
            )
            y_pred_fold = model_fold.predict(X_fold_val)
            metrics_fold = compute_metrics(y_fold_val, y_pred_fold, f"Fold {fold_idx+1}")
            metrics_fold["fold"]    = fold_idx + 1
            metrics_fold["n_train"] = len(X_fold_train)
            metrics_fold["n_val"]   = len(X_fold_val)
            fold_results.append(metrics_fold)
        except Exception as e:
            logger.warning(f"Fold {fold_idx+1} falhou: {e}")

    if not fold_results:
        return {"error": "Nenhum fold completado com sucesso.", "folds": []}

    folds_df = pd.DataFrame(fold_results)
    aggregated = {
        "MAE_mean":  round(float(folds_df["MAE"].mean()), 4),
        "MAE_std":   round(float(folds_df["MAE"].std()), 4),
        "RMSE_mean": round(float(folds_df["RMSE"].mean()), 4),
        "RMSE_std":  round(float(folds_df["RMSE"].std()), 4),
        "R2_mean":   round(float(folds_df["R²"].mean()), 4),
        "R2_std":    round(float(folds_df["R²"].std()), 4),
        "n_folds":   len(fold_results),
    }

    logger.info(f"CV concluído: RMSE={aggregated['RMSE_mean']:.4f} ± {aggregated['RMSE_std']:.4f}")
    return {"folds": fold_results, "aggregated": aggregated, "folds_df": folds_df}


# ---------------------------------------------------------------------------
# Hyperparameter tuning com Optuna
# ---------------------------------------------------------------------------

def run_hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: ModelConfig = DEFAULT_MODEL,
    tuning_cfg: TuningConfig = DEFAULT_TUNING,
    cv_cfg: CVConfig = DEFAULT_CV,
) -> dict:
    """Ajuste automático de hiperparâmetros com Optuna usando folds temporais."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna não instalado. Usando hiperparâmetros padrão.")
        return {
            "best_params": None, "best_rmse": None,
            "n_trials_completed": 0, "error": "optuna_not_installed",
        }

    tscv = TimeSeriesSplit(
        n_splits=tuning_cfg.cv_folds,
        test_size=cv_cfg.test_size,
        gap=cv_cfg.gap,
    )

    def objective(trial):
        params = {
            "n_estimators":          trial.suggest_int("n_estimators", 200, 800),
            "learning_rate":         trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "max_depth":             trial.suggest_int("max_depth", 3, 8),
            "min_child_weight":      trial.suggest_int("min_child_weight", 3, 15),
            "subsample":             trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":             trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":            trial.suggest_float("reg_lambda", 0.5, 5.0),
            "early_stopping_rounds": 25,
            "random_state": cfg.random_state,
            "n_jobs": cfg.n_jobs,
            "eval_metric": cfg.eval_metric,
        }

        rmse_folds = []
        for train_idx, val_idx in tscv.split(X_train):
            Xf_tr = X_train.iloc[train_idx]
            yf_tr = y_train.iloc[train_idx]
            Xf_vl = X_train.iloc[val_idx]
            yf_vl = y_train.iloc[val_idx]
            if len(Xf_tr) < 15 or len(Xf_vl) < 3:
                continue
            try:
                m = XGBRegressor(**params)
                m.fit(Xf_tr, yf_tr, eval_set=[(Xf_vl, yf_vl)], verbose=False)
                preds = m.predict(Xf_vl)
                rmse_folds.append(np.sqrt(mean_squared_error(yf_vl, preds)))
            except Exception:
                rmse_folds.append(1e9)

        return float(np.mean(rmse_folds)) if rmse_folds else 1e9

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(
            objective,
            n_trials=tuning_cfg.n_trials,
            timeout=tuning_cfg.timeout_seconds,
            show_progress_bar=False,
        )

    best = study.best_params
    best["early_stopping_rounds"] = 40
    best["random_state"] = cfg.random_state
    best["n_jobs"] = cfg.n_jobs
    best["eval_metric"] = cfg.eval_metric

    logger.info(f"Tuning: {len(study.trials)} trials | RMSE CV: {study.best_value:.4f}")
    return {
        "best_params":          best,
        "best_rmse":            round(float(study.best_value), 4),
        "n_trials_completed":   len(study.trials),
        "all_trials":           len(study.trials),
    }


# ---------------------------------------------------------------------------
# Importância das features
# ---------------------------------------------------------------------------

def get_feature_importance(model, feature_cols, top_n=30):
    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    fi_df["importance_pct"] = (
        fi_df["importance"] / fi_df["importance"].sum() * 100
    ).round(2)
    return fi_df


# ---------------------------------------------------------------------------
# Verificação de extrapolação
# ---------------------------------------------------------------------------

def check_extrapolation(X_train, X_pred, feature_cols, threshold_pct=0.2):
    alerts = []
    out_of_range = []
    for col in feature_cols:
        if col not in X_train.columns or col not in X_pred.columns:
            continue
        pred_vals = X_pred[col].dropna()
        if len(pred_vals) == 0:
            continue
        if pred_vals.min() < X_train[col].min() or pred_vals.max() > X_train[col].max():
            out_of_range.append(col)
    if out_of_range:
        pct = len(out_of_range) / max(len(feature_cols), 1)
        severity = "⚠️" if pct > threshold_pct else "ℹ️"
        alerts.append(
            f"{severity} {len(out_of_range)} feature(s) fora do intervalo de treino "
            f"({pct:.0%} das features). "
            + ("Extrapolação significativa — confiabilidade reduzida."
               if pct > threshold_pct
               else f"Extrapolação leve: {out_of_range[:3]}")
        )
    return alerts


# ---------------------------------------------------------------------------
# Auditoria de dados antes do split
# ---------------------------------------------------------------------------

def _audit_dataframe(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    date_col: str,
) -> dict:
    """
    Produz relatório diagnóstico do DataFrame antes do split.
    Chamado por run_training() para log detalhado.

    Returns:
        Dict com contagens e situação de cada coluna relevante.
    """
    n = len(df)
    audit = {
        "n_total":              n,
        "n_target_null":        int(df[target_col].isna().sum()) if target_col in df.columns else "coluna ausente",
        "n_target_valid":       int(df[target_col].notna().sum()) if target_col in df.columns else 0,
        "feature_cols_total":   len(feature_cols),
        "feature_cols_missing": [c for c in feature_cols if c not in df.columns],
        "date_min":             str(df[date_col].min().date()) if date_col in df.columns and n > 0 else "N/D",
        "date_max":             str(df[date_col].max().date()) if date_col in df.columns and n > 0 else "N/D",
        "instrumentos":         (
            list(df["instrumento"].unique()) if "instrumento" in df.columns else "coluna ausente"
        ),
    }

    # Contar nulos por feature
    null_counts = {}
    for col in feature_cols:
        if col in df.columns:
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                null_counts[col] = {"n_null": n_null, "pct": round(100 * n_null / max(n, 1), 1)}
    audit["features_com_nulos"] = null_counts

    # Linhas onde TODAS as features são nulas (linha inútil)
    if feature_cols and all(c in df.columns for c in feature_cols):
        all_null_rows = int(df[feature_cols].isna().all(axis=1).sum())
        audit["linhas_todas_features_nulas"] = all_null_rows

    return audit


# ---------------------------------------------------------------------------
# Pipeline completo de treinamento — com auditoria pré-split
# ---------------------------------------------------------------------------

def run_training(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "leitura",
    date_col: str = "data",
    cfg: ModelConfig = DEFAULT_MODEL,
    tuning_cfg: TuningConfig = DEFAULT_TUNING,
    cv_cfg: CVConfig = DEFAULT_CV,
    run_cv: bool = True,
    run_tuning: bool = None,
    save_path: Optional[str] = None,
) -> TrainingResult:
    """
    Pipeline completo de treinamento com auditoria de dados pré-split.

    Etapas:
    1. Auditoria do DataFrame recebido (log diagnóstico)
    2. Adaptação automática de test_ratio para datasets pequenos
    3. Split temporal
    4. Hyperparameter tuning (opcional)
    5. Validação cruzada temporal (opcional)
    6. Treinamento final
    7. Avaliação e métricas
    """
    logger.info("=== Pipeline de treinamento iniciado ===")
    result_warnings = []
    do_tuning = run_tuning if run_tuning is not None else tuning_cfg.enabled

    # ── Auditoria pré-split ────────────────────────────────────────────────
    audit = _audit_dataframe(df, feature_cols, target_col, date_col)
    logger.info(
        f"[AUDITORIA PRÉ-SPLIT]\n"
        f"  • Linhas totais:        {audit['n_total']}\n"
        f"  • Período:              {audit['date_min']} → {audit['date_max']}\n"
        f"  • Target válidos:       {audit['n_target_valid']}\n"
        f"  • Target nulos:         {audit['n_target_null']}\n"
        f"  • Features solicitadas: {audit['feature_cols_total']}\n"
        f"  • Features ausentes:    {audit['feature_cols_missing']}\n"
        f"  • Instrumentos:         {audit['instrumentos']}"
    )

    if audit["n_total"] == 0:
        raise DataPipelineError(
            "O DataFrame chegou VAZIO ao treinamento. "
            "Revise as etapas de merge, feature engineering e dropna().",
            diagnostics=audit,
        )

    if audit["feature_cols_missing"]:
        raise DataPipelineError(
            f"Features ausentes no DataFrame: {audit['feature_cols_missing']}. "
            "O feature engineering pode não ter sido executado.",
            diagnostics=audit,
        )

    # ── Adaptação automática de test_ratio ─────────────────────────────────
    # Para datasets pequenos, reduz automaticamente o test_ratio para preservar
    # linhas suficientes no treino.
    effective_test_ratio = cfg.test_ratio
    n_total = audit["n_total"]

    if n_total < 200:
        # Com poucos dados, garante ao menos 60% para treino
        max_test_ratio = max(0.10, 1.0 - (80 / max(n_total, 1)))
        if effective_test_ratio > max_test_ratio:
            effective_test_ratio = round(max_test_ratio, 2)
            result_warnings.append(
                f"Dataset pequeno ({n_total} amostras): test_ratio reduzido de "
                f"{cfg.test_ratio:.0%} para {effective_test_ratio:.0%} "
                "para garantir treino mínimo."
            )
            logger.warning(
                f"test_ratio adaptado: {cfg.test_ratio:.0%} → {effective_test_ratio:.0%} "
                f"(n_total={n_total})"
            )

    if audit.get("features_com_nulos"):
        top_nulls = sorted(
            audit["features_com_nulos"].items(),
            key=lambda x: x[1]["n_null"],
            reverse=True,
        )[:5]
        logger.warning(
            f"Features com mais nulos (top 5): "
            + ", ".join(f"{c}={v['pct']}%" for c, v in top_nulls)
        )

    # ── Split temporal ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, test_df_raw = temporal_split(
        df, feature_cols, target_col, date_col,
        effective_test_ratio, cfg.min_test_samples,
    )

    logger.info(
        f"[SPLIT] Treino={len(X_train)} | Teste={len(X_test)} | "
        f"test_ratio_efetivo={effective_test_ratio:.0%}"
    )

    # Verificar cobertura de instrumentos no teste
    if "instrumento" in test_df_raw.columns:
        test_insts  = set(test_df_raw["instrumento"].unique())
        train_insts = set(df["instrumento"].unique())
        missing     = train_insts - test_insts
        if missing:
            result_warnings.append(
                f"Instrumento(s) sem amostras no conjunto de teste: {missing}. "
                "Métricas finais não cobrem esses instrumentos."
            )

    # ── Hyperparameter tuning ──────────────────────────────────────────────
    best_params   = None
    if do_tuning and len(X_train) >= 50:
        logger.info(
            f"Tuning Optuna: {tuning_cfg.n_trials} trials, "
            f"timeout={tuning_cfg.timeout_seconds}s ..."
        )
        tuning_result = run_hyperparameter_tuning(X_train, y_train, cfg, tuning_cfg, cv_cfg)
        best_params   = tuning_result.get("best_params")
        if best_params:
            logger.info("Tuning concluído — usando parâmetros otimizados.")
        else:
            result_warnings.append("Tuning não encontrou melhoria — usando hiperparâmetros padrão.")
    elif do_tuning:
        result_warnings.append(
            f"Dados insuficientes para tuning ({len(X_train)} amostras). "
            "Usando hiperparâmetros padrão."
        )

    # ── Validação cruzada temporal ─────────────────────────────────────────
    cv_results = None
    if run_cv and len(X_train) >= 50:
        cv_results = run_temporal_cv(X_train, y_train, cfg, cv_cfg, best_params)
        if "aggregated" in cv_results:
            agg = cv_results["aggregated"]
            rmse_m = agg.get("RMSE_mean", None)
            rmse_s = agg.get("RMSE_std",  None)
            if rmse_s and rmse_m and (rmse_s / max(rmse_m, 1e-9)) > 0.30:
                result_warnings.append(
                    f"CV temporal: alta variabilidade entre folds "
                    f"(RMSE_std/RMSE_mean = {rmse_s/rmse_m:.1%}). "
                    "O modelo pode ser instável ao longo do tempo."
                )

    # ── Treinamento final ──────────────────────────────────────────────────
    model = train_xgboost(X_train, y_train, X_test, y_test, best_params, cfg)

    # ── Métricas ───────────────────────────────────────────────────────────
    y_pred_test  = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    metrics_test  = compute_metrics(y_test,  y_pred_test,  "Teste Final")
    metrics_train = compute_metrics(y_train, y_pred_train, "Treino")

    if metrics_train["R²"] - metrics_test["R²"] > 0.15:
        result_warnings.append(
            f"Possível overfitting: R² treino={metrics_train['R²']:.3f}, "
            f"R² teste={metrics_test['R²']:.3f}. "
            "Considere aumentar regularização ou reduzir max_depth."
        )

    fi_df   = get_feature_importance(model, feature_cols)
    test_df = test_df_raw.copy()
    test_df["previsao"] = y_pred_test

    # Períodos
    train_sorted = df.sort_values(date_col).iloc[:len(X_train)]
    train_period = (
        str(train_sorted[date_col].min().date()),
        str(train_sorted[date_col].max().date()),
    )
    test_period = (
        str(test_df[date_col].min().date()),
        str(test_df[date_col].max().date()),
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": model, "feature_cols": feature_cols, "best_params": best_params},
            save_path,
        )

    logger.info(
        f"=== Treinamento concluído | Treino={len(X_train)} | Teste={len(X_test)} | "
        f"RMSE={metrics_test['RMSE']} | R²={metrics_test['R²']} ==="
    )

    return TrainingResult(
        model=model,
        feature_cols=feature_cols,
        metrics_test=metrics_test,
        metrics_train=metrics_train,
        feature_importance=fi_df,
        test_df=test_df,
        train_period=train_period,
        test_period=test_period,
        n_train=len(X_train),
        n_test=len(X_test),
        warnings=result_warnings,
        cv_results=cv_results,
        best_params=best_params or _cfg_to_params(cfg),
        tuning_enabled=do_tuning,
    )


def _cfg_to_params(cfg: ModelConfig) -> dict:
    return {
        "n_estimators":   cfg.n_estimators,
        "learning_rate":  cfg.learning_rate,
        "max_depth":      cfg.max_depth,
        "min_child_weight": cfg.min_child_weight,
        "subsample":      cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_alpha":      cfg.reg_alpha,
        "reg_lambda":     cfg.reg_lambda,
    }


def load_model(path: str):
    payload = joblib.load(path)
    return payload["model"], payload["feature_cols"]

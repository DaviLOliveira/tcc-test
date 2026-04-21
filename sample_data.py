"""
sample_data.py
--------------
Gerador de base de dados sintética para testes e demonstrações.
Simula comportamento realista de piezômetros e medidores de nível d'água.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIPOS = ["Piezômetro", "Medidor de Nível d'Água"]
MATERIAIS = ["Argila", "Areia", "Rocha", "Argila Siltosa", "Cascalho", "Aterro Compactado"]
PERM_MAP = {
    "Argila": 1e-9,
    "Areia": 1e-4,
    "Rocha": 1e-7,
    "Argila Siltosa": 1e-8,
    "Cascalho": 1e-3,
    "Aterro Compactado": 5e-8,
}


def generate_sample_data(
    n_instruments: int = 6,
    n_days: int = 730,
    start_date: str = "2022-01-01",
    seed: int = 42,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gera base de dados sintética de instrumentação geotécnica.

    Simula:
    - Resposta do piezômetro ao nível do reservatório (com lag)
    - Resposta à pluviometria acumulada (com amortecimento por material)
    - Sazonalidade anual
    - Tendência de longo prazo suave
    - Ruído de sensor controlado
    - Lacunas e valores ausentes realistas (~2%)

    Args:
        n_instruments: Número de instrumentos a simular.
        n_days: Número de dias de dados.
        start_date: Data de início da série.
        seed: Semente aleatória para reprodutibilidade.
        save_path: Se fornecido, salva como CSV.

    Returns:
        DataFrame no formato consolidado esperado pelo sistema.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    t = np.arange(n_days)

    # Variáveis externas (comuns a todos os instrumentos)
    pluviometria = _generate_rainfall(rng, n_days)
    nivel_reservatorio = _generate_reservoir(t, rng, n_days)

    configs = _generate_instrument_configs(rng, n_instruments)
    records = []

    for cfg in configs:
        leitura = _simulate_reading(t, pluviometria, nivel_reservatorio, cfg, rng, n_days)

        # Introduzir ausentes (~2%)
        missing_mask = rng.random(n_days) < 0.02
        leitura[missing_mask] = np.nan

        for j, date in enumerate(dates):
            records.append({
                "data": date,
                "instrumento": cfg["instrumento"],
                "tipo_instrumento": cfg["tipo_instrumento"],
                "leitura": round(float(leitura[j]), 4) if not np.isnan(leitura[j]) else np.nan,
                "pluviometria": round(float(pluviometria[j]), 1),
                "nivel_reservatorio": round(float(nivel_reservatorio[j]), 3),
                "permeabilidade": cfg["permeabilidade"],
                "tipo_material": cfg["tipo_material"],
                "profundidade": cfg["profundidade"],
                "cota_instalacao": cfg["cota_instalacao"],
            })

    df = pd.DataFrame(records).sort_values(["instrumento", "data"]).reset_index(drop=True)
    logger.info(
        f"Base sintética gerada: {len(df):,} registros | "
        f"{n_instruments} instrumentos | {n_days} dias"
    )

    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Base salva em: {save_path}")

    return df


def _generate_rainfall(rng: np.random.Generator, n_days: int) -> np.ndarray:
    """Simula série de pluviometria com episódios de chuva sazonais."""
    rain = np.zeros(n_days)
    n_events = int(n_days / 12)
    for _ in range(n_events):
        start = rng.integers(0, n_days)
        duration = int(rng.integers(1, 6))
        intensity = float(rng.exponential(18))
        rain[start:min(start + duration, n_days)] += intensity
    return np.clip(rain, 0, 180)


def _generate_reservoir(t: np.ndarray, rng: np.random.Generator, n_days: int) -> np.ndarray:
    """Simula nível do reservatório com sazonalidade anual e tendência."""
    return (
        450
        + 18 * np.sin(2 * np.pi * t / 365 - np.pi / 3)
        + 0.004 * t
        + rng.normal(0, 0.4, n_days)
    )


def _generate_instrument_configs(
    rng: np.random.Generator,
    n_instruments: int,
) -> list[dict]:
    """Gera configurações individuais por instrumento."""
    configs = []
    for i in range(n_instruments):
        tipo = TIPOS[i % len(TIPOS)]
        material = str(rng.choice(MATERIAIS))
        base_perm = PERM_MAP[material]
        cfg = {
            "instrumento": f"INST-{i+1:03d}",
            "tipo_instrumento": tipo,
            "tipo_material": material,
            "permeabilidade": base_perm * float(rng.uniform(0.9, 1.1)),
            "profundidade": round(float(rng.uniform(5, 45)), 1),
            "cota_instalacao": round(float(rng.uniform(180, 620)), 2),
            "base_nivel": float(rng.uniform(8, 55)),
            "amplitude": float(rng.uniform(2, 9)),
            "noise_std": float(rng.uniform(0.08, 0.45)),
            "rain_sensitivity": float(rng.uniform(0.008, 0.12)),
            "reserv_sensitivity": float(rng.uniform(0.03, 0.12)),
        }
        configs.append(cfg)
    return configs


def _simulate_reading(
    t: np.ndarray,
    pluviometria: np.ndarray,
    nivel_reservatorio: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    n_days: int,
) -> np.ndarray:
    """Simula série de leituras de um instrumento."""
    chuva_acum_7 = pd.Series(pluviometria).rolling(7, min_periods=1).sum().values
    return (
        cfg["base_nivel"]
        + cfg["reserv_sensitivity"] * (nivel_reservatorio - 450)
        + cfg["rain_sensitivity"] * chuva_acum_7
        + cfg["amplitude"] * np.sin(2 * np.pi * t / 365)
        + rng.normal(0, cfg["noise_std"], n_days)
    )

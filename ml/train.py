"""
ml/train.py — LightGBM 3モデルの学習・評価・保存

モデル構成:
  model_A: 釣果数回帰（count 予測）
  model_B: 釣行推奨二値分類（count >= 5 → 1）
  model_C: 魚種別出現確率（魚種ごとの二値分類）

実行:
  python ml/train.py
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    roc_auc_score, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml.feature_builder import (
    build_features, load_csv,
    get_feature_cols, CATEGORICAL_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path('ml/models')


# ------------------------------------------------------------------
# モデル A: 釣果数回帰
# ------------------------------------------------------------------

def train_model_a(df: pd.DataFrame) -> LGBMRegressor:
    """count を予測する LightGBM Regressor を学習。"""
    logger.info('=== Model A: 釣果数回帰 ===')

    feat_cols = get_feature_cols(include_species=True)
    target = 'count'

    sub = df[df[target].notna()].copy()
    logger.info('学習データ: %d 件', len(sub))

    X = sub[feat_cols]
    y = sub[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        categorical_feature=CATEGORICAL_FEATURES,
        eval_set=[(X_test, y_test)],
        callbacks=[],
    )

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    logger.info('Model A — MAE: %.2f  R²: %.3f', mae, r2)

    path = MODELS_DIR / 'model_a.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info('Model A 保存: %s', path)

    return model


# ------------------------------------------------------------------
# モデル B: 釣行推奨二値分類
# ------------------------------------------------------------------

def train_model_b(df: pd.DataFrame) -> LGBMClassifier:
    """count >= 5 を「釣れる日（1）」とする二値分類モデルを学習。"""
    logger.info('=== Model B: 釣行推奨分類 ===')

    feat_cols = get_feature_cols(include_species=False)
    sub = df[df['count'].notna()].copy()
    sub['label'] = (sub['count'] >= 5).astype(int)

    logger.info('学習データ: %d 件（正例: %d, 負例: %d）',
                len(sub), sub['label'].sum(), (sub['label'] == 0).sum())

    X = sub[feat_cols]
    y = sub['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        categorical_feature=CATEGORICAL_FEATURES,
        eval_set=[(X_test, y_test)],
    )

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc  = roc_auc_score(y_test, y_proba)
    logger.info('Model B — AUC: %.3f', auc)
    logger.info('\n%s', classification_report(y_test, y_pred, target_names=['不釣れ', '釣れる']))

    path = MODELS_DIR / 'model_b.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info('Model B 保存: %s', path)

    return model


# ------------------------------------------------------------------
# モデル C: 魚種別出現確率（魚種ごとに二値分類）
# ------------------------------------------------------------------

def train_model_c(df: pd.DataFrame) -> dict[str, LGBMClassifier]:
    """魚種ごとに「その日その磯で釣れたか」を予測する分類モデルを学習。

    Returns:
        {species_name: LGBMClassifier} の辞書
    """
    logger.info('=== Model C: 魚種別出現確率 ===')

    feat_cols = get_feature_cols(include_species=False)

    # 日×磯単位で集計（同日同磯の複数レコードは1件に集約）
    daily_spot = (
        df.groupby(['date', 'spot_enc'])
        .agg(
            **{col: (col, 'first') for col in feat_cols if col != 'spot_enc'},
        )
        .reset_index()
    )

    # 各魚種が釣れた日×磯を展開
    species_records = (
        df[df['species'].notna()]
        .groupby(['date', 'spot_enc'])['species']
        .apply(lambda x: set(x.dropna().astype(str)))
        .reset_index()
        .rename(columns={'species': 'caught_species'})
    )

    base = daily_spot.merge(species_records, on=['date', 'spot_enc'], how='left')
    base['caught_species'] = base['caught_species'].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    target_species = [
        s for s in df['species'].dropna().unique()
        if df[df['species'] == s].shape[0] >= 30  # 30件以上ある魚種のみ
    ]
    logger.info('対象魚種: %s', target_species)

    models: dict[str, LGBMClassifier] = {}
    metrics: dict[str, dict] = {}

    for sp in target_species:
        base[f'target_{sp}'] = base['caught_species'].apply(lambda s: int(sp in s))
        y = base[f'target_{sp}']

        if y.sum() < 20:
            logger.info('  %s: 正例 %d 件（少なすぎるためスキップ）', sp, y.sum())
            continue

        X = base[feat_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        m = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_child_samples=10,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
        )
        m.fit(X_train, y_train,
              categorical_feature=CATEGORICAL_FEATURES)

        y_proba = m.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = float('nan')

        logger.info('  %s — AUC: %.3f  (n=%d, pos=%d)',
                    sp, auc, len(y), y.sum())
        metrics[sp] = {'auc': auc, 'n': int(len(y)), 'pos': int(y.sum())}
        models[sp] = m

    # 保存
    path = MODELS_DIR / 'model_c.pkl'
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    logger.info('Model C 保存: %s  (%d 魚種)', path, len(models))

    metrics_path = MODELS_DIR / 'model_c_metrics.json'
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    return models


# ------------------------------------------------------------------
# 特徴量重要度の保存
# ------------------------------------------------------------------

def _save_feature_importance(model_a: LGBMRegressor,
                              model_b: LGBMClassifier,
                              feat_cols_a: list[str],
                              feat_cols_b: list[str]) -> None:
    rows = []
    for col, imp in zip(feat_cols_a, model_a.feature_importances_):
        rows.append({'model': 'A', 'feature': col, 'importance': int(imp)})
    for col, imp in zip(feat_cols_b, model_b.feature_importances_):
        rows.append({'model': 'B', 'feature': col, 'importance': int(imp)})

    fi_df = pd.DataFrame(rows).sort_values(['model', 'importance'], ascending=[True, False])
    fi_path = MODELS_DIR / 'feature_importance.csv'
    fi_df.to_csv(fi_path, index=False, encoding='utf-8-sig')
    logger.info('特徴量重要度保存: %s', fi_path)


# ------------------------------------------------------------------
# メイン
# ------------------------------------------------------------------

def main(skip_if_no_new: bool = False) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info('CSV 読み込み中...')
    raw_df = load_csv()
    logger.info('レコード数（生）: %d', len(raw_df))

    # 新規データなしは skip
    if skip_if_no_new and raw_df.empty:
        logger.info('新規データなし。学習をスキップします。')
        return

    logger.info('特徴量構築中...')
    df = build_features(raw_df)
    logger.info('特徴量構築完了: %d レコード', len(df))

    # 特徴量をParquetで保存（デバッグ用）
    feat_path = Path('ml/features.parquet')
    df.to_parquet(feat_path, index=False)
    logger.info('特徴量 Parquet 保存: %s', feat_path)

    model_a = train_model_a(df)
    model_b = train_model_b(df)
    train_model_c(df)

    _save_feature_importance(
        model_a, model_b,
        get_feature_cols(include_species=True),
        get_feature_cols(include_species=False),
    )

    logger.info('=== 学習完了 ===')


if __name__ == '__main__':
    main()

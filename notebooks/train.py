import gc
from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, backend as K
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer 

# --- 設定 ---
RANDOM_STATE = 42
BASE_DIR = Path("/workspace/competitions/nfl-big-data-bowl-2026-prediction")
DATA_DIR = BASE_DIR / "data/raw"
OUTPUT_DIR = BASE_DIR / "notebooks/artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- グローバル変数 ---
RAW_FEATURES = [
    'rel_x', 'rel_y', 's', 'a', 
    'dir_sin', 'dir_cos', 'o_sin', 'o_cos',
    'player_bmi', 'player_age_years', 
    'is_defense', 'is_offense', 'is_ball' 
]
MAX_FRAMES = 100      
MAX_NODES = 23        
PADDING_VALUE = 0.0   
TRAIN_WEEKS = [f"{week:02d}" for week in range(1, 19)]
TARGET_KEYS = ["game_id", "play_id", "nfl_id", "frame_id"] 
SELECT_COLUMNS = [
    "game_id", "play_id", "nfl_id", "frame_id", "player_to_predict",
    "player_height", "player_weight", "player_birth_date", "player_position",
    "player_side", "player_role", "x", "y", "s", "a", "dir", "o",
    "num_frames_output"
]

# --- ヘルパー関数 ---
def _parse_height(heights: pd.Series) -> pd.Series:
    parts = heights.fillna("0-0").str.split("-", expand=True)
    feet = pd.to_numeric(parts[0], errors="coerce")
    inches = pd.to_numeric(parts[1], errors="coerce")
    return feet * 12 + inches

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["player_position"] = data["player_position"].fillna("unknown")
    data["player_side"] = data["player_side"].fillna("unknown")
    data["player_role"] = data["player_role"].fillna("unknown")

    data["height_inches"] = _parse_height(data.get("player_height", pd.Series(index=data.index, dtype=str)))
    data["player_weight"] = pd.to_numeric(data.get("player_weight"), errors="coerce")
    height_m = data["height_inches"] * 0.0254
    weight_kg = data["player_weight"] * 0.45359237
    data["player_bmi"] = weight_kg / np.square(height_m)
    data.loc[~np.isfinite(data["player_bmi"]), "player_bmi"] = np.nan

    birth_dates = pd.to_datetime(data.get("player_birth_date"), errors="coerce")
    game_dates = pd.to_datetime(data["game_id"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    data["player_age_years"] = (game_dates - birth_dates).dt.days / 365.25
    
    data["is_defense"] = (data["player_side"].str.lower() == "defense").astype("int8")
    data["is_offense"] = (data["player_side"].str.lower() == "offense").astype("int8")
    data["is_ball"] = (data["player_role"].str.lower() == "football").astype("int8")

    # ここで x -> track_x, y -> track_y に変更している
    data = data.rename(columns={"x": "track_x", "y": "track_y"}, errors="ignore")
    
    for angle_col in ("dir", "o"):
        if angle_col in data.columns:
            radians = np.deg2rad(data[angle_col].fillna(0)) 
            data[f"{angle_col}_sin"] = np.sin(radians)
            data[f"{angle_col}_cos"] = np.cos(radians)
            data.loc[data[angle_col].isna(), [f"{angle_col}_sin", f"{angle_col}_cos"]] = np.nan
        else:
            data[f"{angle_col}_sin"] = np.nan
            data[f"{angle_col}_cos"] = np.nan
            
    return data

def load_week(week: str) -> pd.DataFrame:
    input_path = DATA_DIR / "train" / f"input_2023_w{week}.csv"
    output_path = DATA_DIR / "train" / f"output_2023_w{week}.csv"
    
    use_cols = SELECT_COLUMNS.copy()
    features = pd.read_csv(input_path, usecols=use_cols)
    targets = pd.read_csv(output_path).rename(columns={"x": "target_x", "y": "target_y"})
    
    features = features.merge(targets, on=TARGET_KEYS, how="left", suffixes=('', '_target'))
    print(f"Week {week}: {len(features):,} rows loaded.")
    return features

def load_training_data() -> pd.DataFrame:
    frames = []
    for week in TRAIN_WEEKS:
        frames.append(load_week(week))
    train_df = pd.concat(frames, ignore_index=True)
    
    print("Running feature engineering...")
    train_df = engineer_features(train_df)
    
    train_df['is_target_player'] = (
        train_df['player_to_predict'] & 
        (train_df['frame_id'] <= train_df['num_frames_output'])
    )
    return train_df

# --- 【修正】相対座標計算用関数 (track_x / track_y を使用) ---
def add_relative_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating Relative Coordinates (Centering)...")
    df['play_seq_id'] = df['game_id'].astype(str) + "_" + df['play_id'].astype(str)
    
    # engineer_features でリネーム済みのカラム名を使用
    x_col = 'track_x' 
    y_col = 'track_y' 
    
    # まだリネームされていない場合のフォールバック
    if x_col not in df.columns and 'x' in df.columns:
        x_col = 'x'
    if y_col not in df.columns and 'y' in df.columns:
        y_col = 'y'

    # 重心計算
    x_clean = df[x_col].fillna(df[x_col].mean())
    y_clean = df[y_col].fillna(df[y_col].mean())
    
    # play_seq_id と frame_id でグルーピングして平均を算出
    mean_x = df.assign(temp_x=x_clean).groupby(['play_seq_id', 'frame_id'])['temp_x'].transform('mean')
    mean_y = df.assign(temp_y=y_clean).groupby(['play_seq_id', 'frame_id'])['temp_y'].transform('mean')
    
    # 相対座標の生成
    df['rel_x'] = df[x_col] - mean_x
    df['rel_y'] = df[y_col] - mean_y
    
    # ターゲットも相対化
    df['rel_target_x'] = df['target_x'] - mean_x
    df['rel_target_y'] = df['target_y'] - mean_y
    
    return df

# --- シーケンス作成 ---
def create_sequences_from_df(df: pd.DataFrame, feature_cols: list[str], max_frames: int, max_nodes: int) -> tuple:
    print("Grouping by play and padding...")
    grouped_plays = df.groupby('play_seq_id')
    
    X_plays, y_plays = [], []
    groups_cv = [] 
    
    play_ids = df['play_seq_id'].unique()
    
    for play_id in play_ids:
        play_df = grouped_plays.get_group(play_id)
        play_df = play_df.sort_values(['frame_id', 'nfl_id'])
        
        grouped_frames = play_df.groupby('frame_id')
        X_frames, y_frames = [], []
        frame_ids = sorted(play_df['frame_id'].unique())
        
        for frame_id in frame_ids:
            frame_df = grouped_frames.get_group(frame_id)
            
            # Features
            frame_X = frame_df[feature_cols].values
            pad_width_X = ((0, max_nodes - len(frame_X)), (0, 0))
            frame_X_padded = np.pad(frame_X, pad_width_X, 'constant', constant_values=PADDING_VALUE)
            X_frames.append(frame_X_padded)
            
            # Targets
            frame_y_targets = frame_df[['rel_target_x', 'rel_target_y']].values
            frame_mask = frame_df['is_target_player'].values
            frame_y = np.where(frame_mask[:, None], frame_y_targets, np.nan) 
            
            pad_width_y = ((0, max_nodes - len(frame_y)), (0, 0))
            frame_y_padded = np.pad(frame_y, pad_width_y, 'constant', constant_values=np.nan)
            y_frames.append(frame_y_padded)
        
        if len(X_frames) > max_frames:
            X_frames = X_frames[:max_frames]
            y_frames = y_frames[:max_frames]
        
        pad_len = max(0, max_frames - len(X_frames))
        pad_width_frames = ((0, pad_len), (0, 0), (0, 0))
        
        X_play_padded = np.pad(X_frames, pad_width_frames, 'constant', constant_values=PADDING_VALUE)
        y_play_padded = np.pad(y_frames, pad_width_frames, 'constant', constant_values=np.nan)
        y_play_padded = np.nan_to_num(y_play_padded, nan=PADDING_VALUE) 
        
        X_plays.append(X_play_padded)
        y_plays.append(y_play_padded)
        groups_cv.append(play_id) 
        
    return (
        np.array(X_plays, dtype='float32'),
        np.array(y_plays, dtype='float32'),
        np.array(groups_cv)
    )

# --- Transformerモデル定義 ---
def masked_mae_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PADDING_VALUE), tf.float32)
    abs_error = tf.abs(y_true - y_pred)
    masked_error = abs_error * mask
    total_error = tf.reduce_sum(masked_error)
    num_non_zero = tf.reduce_sum(mask)
    return tf.math.divide_no_nan(total_error, num_non_zero)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "rate": self.dropout1.rate
        })
        return config

def build_transformer_model(num_features):
    inputs = layers.Input(shape=(MAX_FRAMES, MAX_NODES, num_features))
    
    x = layers.TimeDistributed(layers.TimeDistributed(layers.Dense(64, activation='gelu')))(inputs)
    
    # Spatial Attention
    x_spatial = layers.Reshape((-1, MAX_NODES, 64))(x)
    x_spatial = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128)(x_spatial)
    
    # Temporal Attention
    x = layers.Reshape((MAX_FRAMES, MAX_NODES, 64))(x_spatial)
    x_temporal = layers.Permute((2, 1, 3))(x)
    x_temporal = layers.Reshape((-1, MAX_FRAMES, 64))(x_temporal)
    
    x_temporal = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128)(x_temporal)
    
    x_temporal = layers.Reshape((-1, MAX_NODES, MAX_FRAMES, 64))(x_temporal)
    x = layers.Permute((2, 1, 3))(x_temporal)
    
    output = layers.TimeDistributed(layers.TimeDistributed(layers.Dense(2)))(x)
    
    model = models.Model(inputs=inputs, outputs=output)
    
    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=10000
    )
    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule), loss=masked_mae_loss)
    return model

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading Data...")
    df = load_training_data()
    
    # 1. 相対座標計算 (修正版)
    df = add_relative_coordinates(df)
    
    # 2. スケーリング
    print("Fitting Scaler/Imputer...")
    imputer = SimpleImputer(strategy='median')
    scaler = RobustScaler()
    
    df[RAW_FEATURES] = imputer.fit_transform(df[RAW_FEATURES])
    df[RAW_FEATURES] = scaler.fit_transform(df[RAW_FEATURES])
    
    print(f"Saving preprocessors to {OUTPUT_DIR}...")
    joblib.dump(imputer, OUTPUT_DIR / 'imputer.pkl')
    joblib.dump(scaler, OUTPUT_DIR / 'scaler.pkl')
    
    # 3. シーケンス作成
    print("Creating Sequences...")
    X_all, y_all, groups = create_sequences_from_df(df, RAW_FEATURES, MAX_FRAMES, MAX_NODES)
    
    print(f"X shape: {X_all.shape}, y shape: {y_all.shape}")
    
    # 5-Fold Training
    gkf = GroupKFold(n_splits=5)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=groups)):
        print(f"\n--- Training Fold {fold + 1}/5 ---")
        
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        
        model = build_transformer_model(len(RAW_FEATURES))
        
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20, 
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        save_path = OUTPUT_DIR / f'model_fold_{fold}.keras'
        model.save(save_path)
        print(f"Saved: {save_path}")
        
        del model, X_train, y_train, X_val, y_val
        gc.collect()
        tf.keras.backend.clear_session()

    print("\nAll folds trained successfully.")
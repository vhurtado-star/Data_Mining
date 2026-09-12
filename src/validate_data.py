import os
import pandas as pd
import numpy as np

# 1. Rutas dinámicas con __file__
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'raw', 'solar_telemetry_corrupted.csv')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)

# 2. Carga del archivo CSV
if os.path.exists(RAW_DATA_PATH):
    df = pd.read_csv(RAW_DATA_PATH)
    print('Ingestion Successful! Loaded:', df.shape[0], 'rows and', df.shape[1], 'columns.\n')
    print('Initial data preview (first 3 rows):')
    print(df.head(3))
    print('-' * 60)
else:
    print('Error: Dataset file not found at:', RAW_DATA_PATH)
    exit() # Corrección: Indentado dentro del else

# 3. Audit A: Primary Key Duplicates
duplicate_mask = df.duplicated(subset=['timestamp', 'panel_id'], keep=False)
df_duplicates = df[duplicate_mask]

print('--- AUDIT A: PRIMARY KEY DUPLICATE SCAN ---')
print(f'Detected {len(df_duplicates)} duplicate rows.')
if len(df_duplicates) > 0:
    print('Duplicate rows identified (displaying first 4):')
    print(df_duplicates[['timestamp', 'panel_id', 'voltage_v', 'current_a']].head(4))
    print('-' * 60)

# 4. Audit B: Negative Electrical Values
voltage_violations = df[df['voltage_v'] < 0]
current_violations = df[df['current_a'] < 0]

print('--- AUDIT B: ELECTRICAL VIOLATION SCAN ---')
print(f'Detected {len(voltage_violations)} rows with negative voltage.')
print(f'Detected {len(current_violations)} rows with negative current.')
if len(voltage_violations) > 0:
    print('Example negative voltage records:')
    print(voltage_violations[['timestamp', 'panel_id', 'voltage_v']].head(3))
if len(current_violations) > 0:
    print('Example negative current records:')
    print(current_violations[['timestamp', 'panel_id', 'current_a']].head(3))
print('-' * 60)

# 5. Audit C: Efficiency Bounds Check
efficiency_violations = df[(df['efficiency_pct'] < 0) | (df['efficiency_pct'] > 100)]

print('--- AUDIT C: MECHANICAL RANGE VIOLATION SCAN ---')
print(f'Detected {len(efficiency_violations)} rows with out-of-bounds efficiency.')
if len(efficiency_violations) > 0:
    print('Example efficiency violations:')
    print(efficiency_violations[['timestamp', 'panel_id', 'efficiency_pct']].head(3))
print('-' * 60)

# 6. Audit D: Temporal Sensor Spike Check
df_sorted = df.sort_values(by=['panel_id', 'timestamp']).copy()
df_sorted['temp_change'] = df_sorted.groupby('panel_id')['temperature_c'].diff().abs()
temp_violations = df_sorted[df_sorted['temp_change'] > 30.0]

print('--- AUDIT D: TEMPORAL SENSOR SPIKE SCAN ---')
print(f'Detected {len(temp_violations)} rapid hourly temperature fluctuations (> 30°C/hr).')
if len(temp_violations) > 0:
    print('Example malfunctioning sensor records:')
    print(temp_violations[['timestamp', 'panel_id', 'temperature_c', 'temp_change']].head(3))
print('-' * 60)

# 7. Aislamiento y Exportación de Datos
corrupted_indices = set()
corrupted_indices.update(duplicate_mask[duplicate_mask].index)
corrupted_indices.update(voltage_violations.index)
corrupted_indices.update(current_violations.index)
corrupted_indices.update(efficiency_violations.index)
corrupted_indices.update(temp_violations.index)

df_corrupted = df.loc[list(corrupted_indices)].copy()
df_clean = df.drop(index=list(corrupted_indices)).copy()

print('=== AUTOMATED DATA QUALITY GATE SUMMARY ===')
print(f'Total Rows Audited: {len(df)}')
print(f'Approved Clean Rows: {len(df_clean)} ({len(df_clean)/len(df)*100:.2f}%)')
print(f'Isolated Corrupted Rows: {len(df_corrupted)} ({len(df_corrupted)/len(df)*100:.2f}%)')
print('============================================\n')

CLEAN_OUT_PATH = os.path.join(PROCESSED_DIR, 'solar_telemetry_clean.csv')
ANOMALY_OUT_PATH = os.path.join(PROCESSED_DIR, 'solar_telemetry_anomalies.csv')

df_clean.to_csv(CLEAN_OUT_PATH, index=False)
df_corrupted.to_csv(ANOMALY_OUT_PATH, index=False)

print(f'Successfully exported clean data to: {CLEAN_OUT_PATH}')
print(f'Successfully exported anomalies subset to: {ANOMALY_OUT_PATH}')

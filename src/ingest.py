import os
import pandas as pd
# ============================================================================
# STEP 1: PORTABLE PATH RESOLUTION (DYNAMIC)
# ============================================================================
# __file__ is a special variable representing the file location of this script.
# os.path.abspath(__file__) resolves the complete, absolute path of this ingest.py file.
# os.path.dirname(...) strips away 'ingest.py', leaving only the folder path (e.g., /Data_Mining/src/).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# To point to our raw data, we must:
# 1. Use '..' to back out of our current folder (/src/) and enter the parent root (/Data_Mining/)
# 2. Navigate inside our 'data' folder, then enter the 'raw' subfolder
# 3. Target our dataset name 'solar_telemetry_raw.csv'
RAW_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data', 'raw',
'solar_telemetry_raw.csv'))
print('=' * 70)
print(f'DEBUG: Active Python Script Directory: {SCRIPT_DIR}')
print(f'DEBUG: Resolved Absolute Path to Data: {RAW_DATA_PATH}')
print('=' * 70)
# ============================================================================
# STEP 2: VERIFICATION & DATA LOADING
# ============================================================================
# Check if the operating system can actually locate our file at the calculated path
if os.path.exists(RAW_DATA_PATH):
    print('\n[SUCCESS] File located successfully! Starting ingestion pipeline...')
    # Ingest the raw data table into a Pandas DataFrame object
    df = pd.read_csv(RAW_DATA_PATH)

    # Print a structured report to our screen
    print('\n' + '=' * 45)
    print(' ENTERPRISE DATA INGESTION REPORT')
    print('=' * 45)
    print(f'Total Row Records (Samples): {df.shape[0]}')
    print(f'Total Column Variables (Features): {df.shape[1]}')
    print('-' * 45)
    print('Detected Column Variables & Data Types:')
    for col in df.columns:
        print(f' - {col:<20} ({df[col].dtype})')
    print('=' * 45 + '\n')

    # Preview the top records to verify matrix loading
    print('Previewing top 3 rows of loaded data:')
    print(df.head(3))
else:
    print('\n[CRITICAL ERROR] Python could not find your dataset!')
    print(f'Attempted Lookup Path: "{RAW_DATA_PATH}"')
    print('\n--- Troubleshooting Checklists ---')
    print(' 1. Check if your file is placed inside "Data_Mining/data/raw/".')
    print(' 2. Verify that the spelling is exactly "solar_telemetry_raw.csv".')
    print(' 3. Make sure you opened the parent folder "Data_Mining/" in VS Code.')
    print('=' * 70)

 
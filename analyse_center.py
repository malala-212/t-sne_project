import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'merged_all_results.csv' 
COL_FEATURES = 'features'
COL_CENTER = 'center'
METRICS_TO_PLOT = ['nh_score', 'trustworthiness', 'continuity', 'spearman']
FEATURES_TO_PLOT = [17, 50, 90]

try:
    data = pd.read_csv(file_path, encoding='latin-1') 
    data.columns = data.columns.str.strip().str.lower().str.replace('[^a-zA-Z0-9_]', '', regex=True)
    data[COL_CENTER] = data[COL_CENTER].astype(str)
    data[COL_FEATURES] = pd.to_numeric(data[COL_FEATURES], errors='coerce').astype(float) 
    
    
except Exception as e:
    print(f"Erreur lors du chargement/nettoyage du fichier : {e}")
    exit()

for metric in METRICS_TO_PLOT:
    
    COL_QUALITY = metric

    data[COL_QUALITY] = pd.to_numeric(data[COL_QUALITY], errors='coerce')
    data_filtered = data[data[COL_FEATURES].isin(FEATURES_TO_PLOT)].copy()
    data_filtered.dropna(subset=[COL_QUALITY], inplace=True)
    
    #calcul de la moyenne
    summary_data = data_filtered.groupby([COL_CENTER, COL_FEATURES])[COL_QUALITY].mean().reset_index()

    counts = summary_data.groupby(COL_CENTER)[COL_FEATURES].count()
    centers_with_data = counts[counts > 0].index.tolist()
    summary_data = summary_data[summary_data[COL_CENTER].isin(centers_with_data)]

    if summary_data.empty:
        print(f"Attention : Aucune donnée trouvée pour la métrique '{metric}' dans les features {FEATURES_TO_PLOT}.")
        continue

    # graphiques
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=summary_data, 
        x=COL_FEATURES, 
        y=COL_QUALITY, 
        hue=COL_CENTER,
        marker='o',
        palette='tab10',
        linewidth=2
    )
    
    plt.title(f"Impact du 'Center' sur  {COL_QUALITY.capitalize()} (Moyenne)", fontsize=14)
    plt.xlabel("Nombre de Features (17, 50, 90)", fontsize=12)
    plt.ylabel(COL_QUALITY.capitalize() + " (Moyenne)", fontsize=12)


    if metric in ['trustworthiness', 'continuity', 'spearman', 'nh_score']:
         plt.ylim(summary_data[COL_QUALITY].min() * 0.9, summary_data[COL_QUALITY].max() * 1.05)
    
    plt.xticks(FEATURES_TO_PLOT)

    plt.legend(title='Center ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show() 

print("\n--- 🖼️ Les quatre graphiques ont été générés successivement. ---")
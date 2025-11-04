import os
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from zadu import zadu as zadu_lib
    HAS_ZADU = True
except ImportError:
    HAS_ZADU = False
    _ZADU_IMPORT_ERR = "La librairie ZADU n'est pas installée. Installez-la avec : python -m pip install zadu"


ROOT_DATA_DIR = "./data/"
ROOT_TSNE_DIR = "./data_tsne/"
TC_RESULTS_DIR = "./analysis_results/trustworthiness_continuity/"
K_NEIGHBORS = 30
CLUSTERS = range(1, 7) 
PERPLEXITIES = [5, 15, 20, 50, 90]
FEATURES_LIST = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 50, 60, 70, 80, 90, 100]


def zadu_tnc(X_hd, X_ld, k):
    """
    Calcule les scores de Trustworthiness (fiabilité) et Continuity (continuité) en utilisant la méthode ZADU.
    
    Arguments :
        X_hd (np.ndarray) : Données originales (hautes dimensions).
        X_ld (np.ndarray) : Données après réduction de dimension (embarquées, t-SNE par exemple).
        k (int) : Le nombre de voisins à utiliser pour le calcul.

    Retour :
        dict : Dictionnaire contenant les scores 'trustworthiness' et 'continuity'.
    """

    if not HAS_ZADU:
        raise RuntimeError(_ZADU_IMPORT_ERR)
        
    spec = [{"id": "tnc", "params": {"k": int(k)}}]
    
    z = zadu_lib.ZADU(spec, X_hd, return_local=False) 
    out = z.measure(X_ld)
    
    g = out[0]
    
    global_scores = {
        "trustworthiness": float(g.get("trustworthiness", np.nan)),
        "continuity": float(g.get("continuity", np.nan)),
    }
    return global_scores

def save_tc_results(df_results, file_path):
    """
    Sauvegarde les résultats de Trustworthiness et Continuity dans un fichier CSV.
    
    Arguments :
        df_results (pd.DataFrame) : DataFrame contenant les résultats à sauvegarder.
        file_path (str ou Path) : Le chemin du fichier où les résultats seront enregistrés.
    
    Retour :
        None
    """
    
    df_results.sort_values(by=['Features'], inplace=True)
    df_results.to_csv(file_path, index=False)
    print(f"\n[OK] Résultats T&C sauvegardés dans: {file_path}")

def run_trustworthiness_analysis(cluster_range, features_list, perplexities, k_neighbors,
    root_data_dir, root_tsne_dir, tc_results_dir):
    """
    Effectue l'analyse de la Trustworthiness et de la Continuity pour les données originales et t-SNE 
    et sauvegarde les résultats dans un fichier CSV.

    Arguments :
        cluster_range (range) : Plage des clusters à traiter (exemple: range(1, 4)).
        features_list (list) : Liste des différentes configurations de caractéristiques (features).
        perplexities (list) : Liste des perplexités pour lesquelles t-SNE sera appliqué.
        k_neighbors (int) : Nombre de voisins à utiliser pour le calcul.
        root_data_dir (str ou Path) : Le répertoire contenant les données originales.
        root_tsne_dir (str ou Path) : Le répertoire contenant les résultats t-SNE.
        tc_results_dir (str ou Path) : Le répertoire où les résultats seront sauvegardés.

    Retour :
        None
    """
    
    if not HAS_ZADU:
        print(_ZADU_IMPORT_ERR)
        return

    root_data_dir_p = Path(root_data_dir)
    root_tsne_dir_p = Path(root_tsne_dir)
    tc_results_dir_p = Path(tc_results_dir)
    
    if not tc_results_dir_p.exists():
        tc_results_dir_p.mkdir(parents=True, exist_ok=True)
        
    print(f"Analyse Trustworthiness & Continuity (K={k_neighbors}) en cours...")
    print(f"Les résultats seront enregistrés dans : {tc_results_dir}")
    
    try:
        for cluster in cluster_range:
            tc_results_list = []
            tc_results_file = tc_results_dir_p / f"tc_results_center{cluster}.csv" 
            
            print(f"\n Début du Cluster {cluster} ")
            
            for feat in features_list:
                
                data_file = root_data_dir_p / f'center{cluster}' / f"features{feat}.csv"
                
                if not data_file.exists():
                    print(f"[!] Fichier de données HD non trouvé: {data_file}. Saut pour Features {feat}.")
                    continue
                    
                df_hd = pd.read_csv(data_file, header=0)
                X_hd = df_hd.drop(columns=['Cluster_Label']).values    
                
                tc_results_entry = {
                    'Cluster': cluster,
                    'Features': feat,
                    'K_Neighbors': k_neighbors,
                }
                
                tsne_sub_dir = root_tsne_dir_p / f'center{cluster}' / f'features{feat}'
                
                for perplexity in perplexities:
                    tsne_file = tsne_sub_dir / f"features{feat}_perplexity{perplexity}.csv"
                    
                    col_trust = f'Trustworthiness_P{perplexity}'
                    col_cont = f'Continuity_P{perplexity}'

                    if tsne_file.exists():
                        df_tsne = pd.read_csv(tsne_file, header=0)
                        X_tsne = df_tsne.drop(columns=['Cluster_Label']).values 
                        
                        global_scores = zadu_tnc(X_hd, X_tsne, k_neighbors)
                        
                        tc_results_entry[col_trust] = global_scores['trustworthiness']
                        tc_results_entry[col_cont] = global_scores['continuity']         
                        
                    else:
                        tc_results_entry[col_trust] = np.nan
                        tc_results_entry[col_cont] = np.nan
                        
                tc_results_list.append(tc_results_entry)
            
            if tc_results_list:
                df_cluster_results = pd.DataFrame(tc_results_list)
                save_tc_results(df_cluster_results, tc_results_file)

    except Exception as e :
        print (f"\n[ERREUR] Erreur globale dans le traitement T&C: {e}")



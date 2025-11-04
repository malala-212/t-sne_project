# Fichier : spearman_analysis.py

import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import pairwise_distances
import os

def pairwise_distance_matrix(data: npt.NDArray) -> npt.NDArray:
    """
    Calcule la matrice des distances euclidiennes entre toutes les paires de points.

    Arguments :
        data (npt.NDArray) : Un tableau numpy de forme (n_samples, n_features) représentant les données.

    Retour :
        npt.NDArray : Une matrice carrée de taille (n_samples, n_samples) où chaque élément [i, j] représente la distance euclidienne entre les points i et j.
    """

    return pairwise_distances(data, metric='euclidean')


def measure(orig: npt.NDArray, emb: npt.NDArray, distance_matrices: tuple | None = None) -> dict:
    """
    Calcule le coefficient de corrélation de rang de Spearman (Shepard's correlation)
    entre les matrices de distance des données originales (HD) et des données embarquées (LD).

    Arguments :
        orig (npt.NDArray) : Un tableau numpy représentant les données originales.
        emb (npt.NDArray) : Un tableau numpy représentant les données embarquées (après réduction de dimension).
        distance_matrices (tuple, optional) : Un tuple contenant les matrices de distance (orig_distance_matrix, emb_distance_matrix).
                                              Si None, les matrices de distance sont calculées à partir des données.

    Retour :
        dict : Un dictionnaire contenant le coefficient de corrélation de Spearman sous la clé 'spearman_rho'.
    """

    if distance_matrices is None: 
        orig_distance_matrix = pairwise_distance_matrix(orig)
        emb_distance_matrix = pairwise_distance_matrix(emb)
    else:
        orig_distance_matrix, emb_distance_matrix = distance_matrices

   
    N = orig_distance_matrix.shape[0]


    if N < 2 or orig_distance_matrix.shape != emb_distance_matrix.shape:
        return {"spearman_rho": np.nan}


    mask = np.triu_indices(N, k=1)
    
    orig_flat = orig_distance_matrix[mask]
    emb_flat = emb_distance_matrix[mask]

    rho, p_value = spearmanr(orig_flat, emb_flat)

    return {
        "spearman_rho": rho,
    }

def save_spearman_results(df_results, file_path):
    """
    Sauvegarde le DataFrame des résultats de la corrélation de Spearman dans un fichier CSV.

    Arguments :
        df_results (pd.DataFrame) : DataFrame contenant les résultats de l'analyse de la corrélation de Spearman.
        file_path (str ou Path) : Le chemin où le fichier CSV sera sauvegardé. 

    Retour :
        None
    """
    
    try:
        df_results.to_csv(file_path, index=False)
        print(f"Résultats de Spearman sauvegardés dans: {file_path}")
    except Exception as e:
        print(f"Erreur de sauvegarde des résultats de Spearman: {e}")

def run_spearman_analysis(cluster_range, features_list, perplexities, root_data_dir="./data/", root_tsne_dir="./data_tsne/", results_dir="./analysis_results/"):
    """
    Effectue l'analyse de la corrélation de Spearman entre les données originales (HD) et les données embarquées (LD)
    pour différents clusters, caractéristiques et perplexités de t-SNE, puis sauvegarde les résultats dans un fichier CSV.

    Arguments :
        cluster_range (iterable) : Plage des clusters à traiter (par exemple, une liste de clusters).
        features_list (iterable) : Liste des différentes configurations de caractéristiques (features) à tester.
        perplexities (iterable) : Liste des perplexités pour lesquelles t-SNE sera appliqué.
        root_data_dir (str ou Path, optionnel) : Le répertoire contenant les données originales. Par défaut, "./data/".
        root_tsne_dir (str ou Path, optionnel) : Le répertoire contenant les résultats de t-SNE. Par défaut, "./data_tsne/".
        results_dir (str ou Path, optionnel) : Le répertoire où les résultats seront sauvegardés. Par défaut, "./analysis_results/".

    Retour :
        None
    """

    root_data_dir = Path(root_data_dir)
    root_tsne_dir = Path(root_tsne_dir)
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
        
    print(f"Analyse de la corrélation de Spearman en cours.")
    
    try:
        for cluster in cluster_range:
            spearman_results_list = []
            spearman_results_file = results_dir / f"spearman_results_center{cluster}.csv" 
            
            for feat in features_list:
                data_file = root_data_dir / f'center{cluster}' / f"features{feat}.csv"
                
                if not data_file.exists():
                    print(f"Fichier de données HD non trouvé: {data_file}. Saut pour Features {feat}.")
                    continue
                    
                df_hd = pd.read_csv(data_file, header=0)
                X_hd = df_hd.drop(columns=['Cluster_Label']).values

                rho_original_dict = measure(orig=X_hd, emb=X_hd)
                
                
                spearman_results_entry = {
                    'Cluster': cluster,
                    'Features': feat,
                    'Spearman_Original': rho_original_dict['spearman_rho']
                }

                tsne_sub_dir = root_tsne_dir / f'center{cluster}' / f'features{feat}'
                
                for perplexity in perplexities:
                    tsne_file = tsne_sub_dir / f"features{feat}_perplexity{perplexity}.csv"
                    column_name = f'Spearman_TSNE_P{perplexity}'

                    if tsne_file.exists():
                        df_tsne = pd.read_csv(tsne_file, header=0)
                        X_tsne = df_tsne.drop(columns=['Cluster_Label']).values 
                        
                        rho_tsne_dict = measure(orig=X_hd, emb=X_tsne)
                        spearman_results_entry[column_name] = rho_tsne_dict['spearman_rho']
                        
                    else:
                        spearman_results_entry[column_name] = np.nan
                        
                spearman_results_list.append(spearman_results_entry)
            
            if spearman_results_list:
                df_cluster_results = pd.DataFrame(spearman_results_list)
                save_spearman_results(df_cluster_results, spearman_results_file)

    except Exception as e :
        print (f"Erreur globale dans le traitement Spearman: {e}")



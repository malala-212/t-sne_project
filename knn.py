import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors 
import os

def measure(emb: npt.NDArray,label: npt.NDArray,k: int = 20,
    knn_emb_info: tuple | None = None,
) -> dict:
    """
    Calcule le Neighborhood Hit (NH) des données embarquées (emb).

    Le Neighborhood Hit mesure la proportion d'échantillons dans les k plus proches voisins 
    d'un point qui appartiennent au même cluster que ce point. 
    Cette mesure est utilisée pour évaluer la qualité de la représentation des données après 
    une réduction de dimension, telle que t-SNE, en termes de la préservation des clusters.

    Arguments :
        emb (npt.NDArray) : Un tableau numpy contenant les données embarquées (représentation réduite des points).
        label (npt.NDArray) : Un tableau numpy contenant les étiquettes de cluster des points.
        k (int, optional) : Le nombre de voisins à considérer pour le calcul du Neighborhood Hit. Par défaut, 20.
        knn_emb_info (tuple, optional) : Un tuple contenant les indices des k voisins pour chaque point. Si ce paramètre 
                                          est fourni, les voisins sont utilisés directement au lieu de les recalculer.

    Retour :
        dict : Un dictionnaire contenant le résultat sous la clé 'neighborhood_hit' avec la valeur du calcul.
    """
    
    if emb.shape[0] < k + 1:
        return {"neighborhood_hit": np.nan}
    
    if knn_emb_info is None:
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean')
        nn.fit(emb)
        distances, indices = nn.kneighbors(emb)
        emb_knn_indices = indices[:, 1:] 
    else:
        emb_knn_indices = knn_emb_info

    points_num = emb.shape[0]
    nh_list = []
    
    emb_knn_indices = emb_knn_indices.astype(int)

    for i in range(points_num):
        emb_knn_index = emb_knn_indices[i]
        emb_knn_index_label = label[emb_knn_index]
        
        nh_list.append(np.sum((emb_knn_index_label == label[i]).astype(int)))

    nh_list = np.array(nh_list)
    nh = np.mean(nh_list) / k

    return {"neighborhood_hit": nh}


def save_nh_results(df_nh, nh_results_file):
    """
    Sauvegarde le DataFrame de résultats du Neighborhood Hit dans un fichier CSV.

    Cette fonction prend un DataFrame contenant les résultats du Neighborhood Hit pour plusieurs 
    configurations et le sauvegarde dans un fichier CSV.

    Arguments :
        df_nh (pandas.DataFrame) : Le DataFrame contenant les résultats du Neighborhood Hit.
        nh_results_file (str or Path) : Le chemin du fichier CSV dans lequel les résultats seront sauvegardés.

    Retour :
        None
    """
    nh_file_path = Path(nh_results_file)
    df_nh.to_csv(nh_file_path, index=False)
    print(f"Neighborhood Hit results saved/overwritten to {nh_results_file}")


def compare_and_save_nh_results(cluster_range,features_list,perplexities,k_neighbors,root_data_dir="./data/",
                                    root_tsne_dir="./data_tsne/", nh_results_base_name="nh_results_center"):
    """
    Compare les fichiers HD et les fichiers t-SNE, calcule le Neighborhood Hit pour chacun, 
    et sauvegarde les résultats par cluster.

    Cette fonction itère sur chaque cluster et chaque ensemble de caractéristiques, 
    calcule le Neighborhood Hit (NH) pour les données originales (HD) et pour les données transformées par t-SNE, 
    puis sauvegarde les résultats dans un fichier CSV pour chaque cluster.

    Arguments :
        cluster_range (iterable) : Plage des clusters à traiter (par exemple, une liste de clusters).
        features_list (iterable) : Liste des différentes configurations de caractéristiques (features) à tester.
        perplexities (iterable) : Liste des perplexités pour lesquelles t-SNE sera appliqué.
        k_neighbors (int) : Le nombre de voisins à utiliser pour le calcul du Neighborhood Hit.
        root_data_dir (str or Path, optional) : Le répertoire contenant les données HD (données originales). Par défaut, "./data/".
        root_tsne_dir (str or Path, optional) : Le répertoire contenant les résultats t-SNE. Par défaut, "./data_tsne/".
        nh_results_base_name (str, optional) : Le nom de base pour les fichiers de résultats de Neighborhood Hit. Par défaut, "nh_results_center".

    Retour :
        None
    """

    print(f"\n--- Démarrage de la comparaison et du calcul NH (K={k_neighbors}) ---")
    
    try:
        for cluster in cluster_range:
            
            nh_results_list = []
            nh_results_file = f"{nh_results_base_name}{cluster}.csv"
            print(f"\nProcessing Cluster: {cluster}. Output file: {nh_results_file}")
            
            for feat in features_list:
                
                original_file_path = Path(root_data_dir) / f'center{cluster}' / f"features{feat}.csv"
                
                if not original_file_path.exists():
                    print(f"Original file not found at {original_file_path}. Skipping analysis for feat {feat}.")
                    continue
                    
                df_original = pd.read_csv(original_file_path, header=0)
                X_hd = df_original.drop(columns=['Cluster_Label']).values 
                Y_labels = df_original['Cluster_Label'].values
                
                nh_original_result = measure(emb=X_hd, label=Y_labels, k=k_neighbors)
                nh_original = nh_original_result.get('neighborhood_hit', np.nan)
                
                nh_results_entry = {
                    'Cluster': cluster,
                    'Features': feat,
                    'K_Neighbors': k_neighbors,
                    'NH_Original': nh_original
                }
                
                tsne_sub_dir = Path(root_tsne_dir) / f'center{cluster}' / f'features{feat}'
                
                for perplexity in perplexities:
                    tsne_file = tsne_sub_dir / f"features{feat}_perplexity{perplexity}.csv"
                    
                    if tsne_file.exists():
                        df_tsne = pd.read_csv(tsne_file, header=0)
                        X_tsne = df_tsne.drop(columns=['Cluster_Label']).values 
                        
                        nh_tsne_result = measure(emb=X_tsne, label=Y_labels, k=k_neighbors)
                        nh_tsne = nh_tsne_result.get('neighborhood_hit', np.nan)
                        
                        nh_results_entry[f'NH_TSNE_P{perplexity}'] = nh_tsne
                    else:
                        nh_results_entry[f'NH_TSNE_P{perplexity}'] = np.nan
                        

                nh_results_list.append(nh_results_entry)
            
            if nh_results_list:
                df_cluster_results = pd.DataFrame(nh_results_list)
                save_nh_results(df_cluster_results, nh_results_file)

    except Exception as e :
        print (f"Global Error in NH processing: {e}")


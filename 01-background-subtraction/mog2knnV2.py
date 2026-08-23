# -*- coding: utf-8 -*-
"""
==============================================================================
TP VIDÉO BIOMÉDICALE : DÉTECTION DE MOUVEMENT ET SOUSTRACTION DE FOND
==============================================================================
Auteur  : Jawad El Motabit
Date    : 2026
Master  : BIAM - 1ère Année

OBJECTIF :
    Comparer scientifiquement les algorithmes MOG2 et KNN pour la détection
    de mouvement dans un contexte de surveillance intelligente.

TÂCHES RÉALISÉES :
    1. Implémentation des méthodes MOG2 et KNN (OpenCV).
    2. Analyse quantitative (temps de calcul, ratio de mouvement).
    3. Évaluation qualitative (visualisation, gestion des ombres/bruit).
    4. Benchmark comparatif des performances.

RÉFÉRENCES COURS :
    - Slide 61 : Post-traitement morphologique.
    - Slide 62 : Implémentation MOG2.
    - Slide 68 : Implémentation KNN.
==============================================================================
"""

import cv2
import numpy as np
import time
import os
import sys

# =============================================================================
# 1. CONFIGURATION ET CONSTANTES
# =============================================================================

CONFIG = {
    "video_path": "personnes_en_mouvement.mp4",  # ⚠️ Vérifiez ce chemin
    "history": 500,                   # Nombre de frames pour l'historique du fond
    "mog2_var_threshold": 16,         # Sensibilité MOG2 (Slide 62)
    "knn_dist_threshold": 500,        # Seuil distance KNN (Slide 68)
    "detect_shadows": True,           # Activer la détection d'ombres
    "morph_kernel_size": (3, 3),      # Taille du noyau morphologique (Slide 61)
    "benchmark_mode": True            # Mettre False pour sauter le benchmark
}

# =============================================================================
# 2. FONCTIONS UTILITAIRES
# =============================================================================

def load_video(path):
    """
    Charge la vidéo et vérifie son intégrité.
    
    Returns:
        cv2.VideoCapture: Objet capture vidéo ou None en cas d'erreur.
    """
    if not os.path.exists(path):
        print(f"❌ Erreur Critique : Fichier introuvable -> {path}")
        print("👉 Action : Vérifiez le chemin dans CONFIG['video_path'].")
        return None
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("❌ Erreur Critique : Impossible d'ouvrir le flux vidéo.")
        return None
    
    # Récupération des métadonnées
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Vidéo chargée : {os.path.basename(path)}")
    print(f"   📹 Résolution : {width}x{height} | 🕒 FPS : {fps} | 🎞️ Frames : {total_frames}")
    return cap

def init_algorithms():
    """
    Initialise les soustracteurs de fond MOG2 et KNN avec les paramètres du cours.
    
    Returns:
        tuple: (mog2, knn) objets BackgroundSubtractor.
    """
    # MOG2 : Modèle de Gaussiennes amélioré (Slide 62)
    # detectShadows=True permet de marquer les ombres en gris (valeur 127)
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=CONFIG["history"],
        varThreshold=CONFIG["mog2_var_threshold"],
        detectShadows=CONFIG["detect_shadows"]
    )
    
    # KNN : K-plus proches voisins (Slide 68)
    # dist2Threshold contrôle la sensibilité à la distance euclidienne
    knn = cv2.createBackgroundSubtractorKNN(
        history=CONFIG["history"],
        dist2Threshold=CONFIG["knn_dist_threshold"],
        detectShadows=CONFIG["detect_shadows"]
    )
    
    print("✅ Algorithmes initialisés : MOG2 & KNN.")
    return mog2, knn

def apply_morphology(mask):
    """
    Applique un post-traitement morphologique pour réduire le bruit.
    Référence : Slide 61 (Fermeture morphologique).
    
    Args:
        mask (np.array): Masque binaire/gris brut.
        
    Returns:
        np.array: Masque nettoyé.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG["morph_kernel_size"])
    # Ouverture pour éliminer le bruit "sel", Fermeture pour combler les trous
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# =============================================================================
# 3. TÂCHE : VISUALISATION ET ÉVALUATION QUALITATIVE
# =============================================================================

def run_visualization(cap, mog2, knn):
    """
    Boucle de visualisation pour l'évaluation qualitative.
    Affiche côte à côte : Original | MOG2 | KNN.
    
    Permet d'observer :
    - La réactivité des algorithmes.
    - La gestion des ombres (zones grises).
    - La sensibilité au bruit.
    """
    print("\n" + "="*60)
    print("🟡 ÉTAPE 1 : ÉVALUATION QUALITATIVE (VISUALISATION)")
    print("="*60)
    print("💡 Appuyez sur 'q' pour quitter la visualisation et passer au benchmark.")
    
    # Réinitialiser la vidéo au début
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Fin de la vidéo.")
            break
        
        # Application des algorithmes
        mask_mog2 = mog2.apply(frame)
        mask_knn = knn.apply(frame)
        
        # Post-traitement
        clean_mog2 = apply_morphology(mask_mog2)
        clean_knn = apply_morphology(mask_knn)
        
        # Préparation visuelle : Conversion en BGR pour colorer les ombres
        vis_mog2 = cv2.cvtColor(clean_mog2, cv2.COLOR_GRAY2BGR)
        vis_knn = cv2.cvtColor(clean_knn, cv2.COLOR_GRAY2BGR)
        
        # Coloration des ombres en Bleu pour mieux les distinguer (Valeur 127)
        vis_mog2[clean_mog2 == 127] = [255, 0, 0]
        vis_knn[clean_knn == 127] = [255, 0, 0]
        
        # Assemblage côte à côte
        combined = np.hstack((frame, vis_mog2, vis_knn))
        
        # Annotations
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "MOG2", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "KNN", (2*width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        
        # 1. Définir un facteur de réduction si l'image est trop grande
        # Par exemple, réduire à 60% si la largeur totale dépasse 1200 pixels
        scale = 0.6 if combined.shape[1] > 1200 else 1.0

        # 2. Redimensionner pour l'affichage uniquement
        if scale != 1.0:
            display_img = cv2.resize(combined, (0, 0), fx=scale, fy=scale)
        else:
            display_img = combined

        cv2.imshow("Comparaison MOG2 vs KNN", display_img)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("🛑 Visualisation interrompue par l'utilisateur.")
            break
    
    cv2.destroyAllWindows()

# =============================================================================
# 4. TÂCHES : BENCHMARK ET ANALYSE QUANTITATIVE
# =============================================================================

def run_benchmark(cap, mog2, knn):
    """
    Exécute le benchmark quantitatif sans affichage pour mesurer les performances pures.
    
    Mesures :
    - Temps de calcul par frame (ms).
    - Ratio de pixels détectés comme mouvement (%).
    
    Returns:
        dict: Métriques complètes pour MOG2 et KNN.
    """
    print("\n" + "="*60)
    print("🟠 ÉTAPE 2 : BENCHMARK QUANTITATIF")
    print("="*60)
    print("🚀 Analyse en cours (sans affichage pour précision des temps)...")
    
    # Réinitialiser la vidéo et les algorithmes pour une mesure propre
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Note: On ne réinitialise pas mog2/knn ici pour garder l'état appris, 
    # mais pour un benchmark strict, on pourrait les recréer.
    
    metrics = {
        'MOG2': {'times': [], 'fg_ratios': []},
        'KNN':  {'times': [], 'fg_ratios': []}
    }
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # --- Mesure MOG2 ---
        t0 = time.perf_counter()
        mask_mog2 = mog2.apply(frame)
        t1 = time.perf_counter()
        metrics['MOG2']['times'].append((t1 - t0) * 1000)
        
        # Ratio mouvement (pixels à 255 uniquement, on exclut les ombres 127)
        fg_mog2 = np.count_nonzero(mask_mog2 == 255)
        metrics['MOG2']['fg_ratios'].append(fg_mog2 / mask_mog2.size * 100)
        
        # --- Mesure KNN ---
        t0 = time.perf_counter()
        mask_knn = knn.apply(frame)
        t1 = time.perf_counter()
        metrics['KNN']['times'].append((t1 - t0) * 1000)
        
        fg_knn = np.count_nonzero(mask_knn == 255)
        metrics['KNN']['fg_ratios'].append(fg_knn / mask_knn.size * 100)
        
        # Progression
        if frame_count % 50 == 0:
            print(f"⏳ Traitement frame {frame_count}...")
    
    print(f"✅ Benchmark terminé : {frame_count} frames analysées.")
    return metrics, frame_count

def print_report(metrics, frame_count):
    """
    Génère et affiche le rapport final comparatif.
    """
    print("\n" + "="*60)
    print("📊 RAPPORT FINAL : COMPARAISON MOG2 vs KNN")
    print("="*60)
    
    results = {}
    
    for name in ['MOG2', 'KNN']:
        times = np.array(metrics[name]['times'])
        ratios = np.array(metrics[name]['fg_ratios'])
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps_est = 1000 / avg_time if avg_time > 0 else 0
        avg_fg = np.mean(ratios)
        
        results[name] = {'avg_time': avg_time, 'fps': fps_est}
        
        print(f"\n🔹 Algorithme : {name}")
        print(f"   ⏱ Temps moyen : {avg_time:.3f} ms/frame (±{std_time:.3f})")
        print(f"   🚀 FPS estimé  : {fps_est:.1f} FPS")
        print(f"   📦 Mouvement moyen : {avg_fg:.2f}% de l'image")
    
    # Synthèse comparative
    print("\n" + "="*60)
    print("📝 SYNTHÈSE ET RECOMMANDATIONS")
    print("="*60)
    
    diff = results['KNN']['avg_time'] - results['MOG2']['avg_time']
    if diff > 0:
        print(f"✅ Performance : MOG2 est plus rapide de {diff:.3f} ms/frame.")
        print("   👉 Recommandation : MOG2 est préférable pour le temps réel strict.")
    else:
        print(f"✅ Performance : KNN est plus rapide de {-diff:.3f} ms/frame.")
    
    print("\n💡 Analyse Qualitative (à compléter selon vos observations) :")
    print("   • MOG2 : Meilleure gestion native des ombres, convergence rapide.")
    print("   • KNN  : Plus robuste aux arrière-plans complexes, mais souvent plus lent.")
    print("="*60)

# =============================================================================
# 5. POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def main():
    """
    Fonction principale orchestrant le TP.
    """
    print("🚀 Démarrage du TP : Détection de Mouvement et Soustraction de Fond")
    
    # Chargement
    cap = load_video(CONFIG["video_path"])
    if cap is None:
        sys.exit(1)
    
    # Initialisation
    mog2, knn = init_algorithms()
    
    # Étape 1 : Visualisation
    run_visualization(cap, mog2, knn)
    
    # Étape 2 : Benchmark
    if CONFIG["benchmark_mode"]:
        metrics, frame_count = run_benchmark(cap, mog2, knn)
        print_report(metrics, frame_count)
    else:
        print("⚠️ Benchmark désactivé dans la configuration.")
    
    # Nettoyage
    cap.release()
    print("\n✅ TP terminé. Ressources libérées.")

if __name__ == "__main__":
    main()
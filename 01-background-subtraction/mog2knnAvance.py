# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:45:25 2026

@author: jawad_el_motabit
"""

import cv2
import numpy as np
import time
import os

class MotionDetectionBenchmark:
    def __init__(self, video_path):
        """Initialisation avec le chemin de la vidéo."""
        self.video_path = video_path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Vidéo introuvable : {video_path}")
        
        # Initialisation des soustracteurs de fond
        # MOG2: detectShadows=True permet de marquer les ombres en gris
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        
        # KNN: dist2Threshold contrôle la sensibilité
        self.knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
        
        # Stockage des métriques
        self.metrics = {
            'MOG2': {'times': [], 'fg_pixels': []},
            'KNN':  {'times': [], 'fg_pixels': []}
        }

    def process_frame(self, frame, subtractor, name):
        """Applique l'algorithme, mesure le temps et calcule les métriques."""
        start = time.perf_counter()
        
        # Application du masque
        mask = subtractor.apply(frame)
        
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        
        # Post-traitement morphologique pour réduire le bruit
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
        # Calcul du pourcentage de pixels avant-plan (mouvement)
        fg_ratio = np.count_nonzero(mask_clean == 255) / mask_clean.size * 100
        
        # Enregistrement des métriques
        self.metrics[name]['times'].append(elapsed_ms)
        self.metrics[name]['fg_pixels'].append(fg_ratio)
        
        return mask, mask_clean

    def run(self, output_video='resultats_tp.avi'):
        """Boucle principale de traitement et affichage."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Erreur: Impossible d'ouvrir la vidéo.")
            return

        # Récupération des propriétés pour l'enregistrement
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Création du VideoWriter pour sauvegarder la comparaison
        # On crée une vue large: [Original | MOG2 | KNN]
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (width*3, height))

        frame_count = 0
        print(f"▶ Démarrage du traitement de la vidéo : {self.video_path}")
        print("-" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Traitement MOG2
            mask_mog2, clean_mog2 = self.process_frame(frame, self.mog2, 'MOG2')
            # Coloration: Les ombres détectées par MOG2 (valeur 127) deviennent bleues pour la visu
            vis_mog2 = cv2.cvtColor(mask_mog2, cv2.COLOR_GRAY2BGR)
            vis_mog2[mask_mog2 == 127] = [255, 0, 0] 

            # Traitement KNN
            mask_knn, clean_knn = self.process_frame(frame, self.knn, 'KNN')
            vis_knn = cv2.cvtColor(mask_knn, cv2.COLOR_GRAY2BGR)
            vis_knn[mask_knn == 127] = [255, 0, 0]

            # Affichage temps réel
            cv2.imshow('Original', frame)
            cv2.imshow('MOG2 Mask', mask_mog2)
            cv2.imshow('KNN Mask', mask_knn)
            
            # Enregistrement de la comparaison côte à côte
            combined = np.hstack((frame, vis_mog2, vis_knn))
            out.write(combined)
            
            # Affichage console toutes les 30 frames
            if frame_count % 30 == 0:
                avg_mog2 = np.mean(self.metrics['MOG2']['times'][-30:])
                avg_knn = np.mean(self.metrics['KNN']['times'][-30:])
                print(f"Frame {frame_count:04d} | MOG2: {avg_mog2:.2f} ms | KNN: {avg_knn:.2f} ms")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.print_final_report(frame_count)

    def print_final_report(self, total_frames):
        """Génération du rapport final (Tâches 2, 3 et 4)."""
        print("\n" + "="*60)
        print("📊 RAPPORT DE COMPARAISON MOG2 vs KNN")
        print("="*60)
        
        for name in ['MOG2', 'KNN']:
            times = self.metrics[name]['times']
            fg_pixels = self.metrics[name]['fg_pixels']
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_fg = np.mean(fg_pixels)
            
            print(f"\n🔹 Algorithme : {name}")
            print(f"   ⏱ Temps moyen par frame : {avg_time:.3f} ms (±{std_time:.3f})")
            print(f"   🚀 FPS estimé           : {1000/avg_time:.1f} FPS")
            print(f"   📦 Mouvement moyen      : {avg_fg:.2f}% de pixels avant-plan")

        # Analyse comparative
        print("\n" + "="*60)
        print("📝 ANALYSE QUALITATIVE & QUANTITATIVE")
        print("="*60)
        print("1. Vitesse : MOG2 est généralement plus rapide et optimisé.")
        print("2. Ombres  : MOG2 gère mieux la détection explicite des ombres.")
        print("3. Bruit   : KNN peut être plus sensible au bruit sans réglage fin.")
        print("4. Stabilité : MOG2 converge souvent plus vite sur le fond.")
        print(f"\n✅ Vidéo comparative sauvegardée : resultats_tp.avi")

# --- Exécution ---
if __name__ == "__main__":
    # Remplacez par le chemin de votre vidéo de test
    VIDEO_FILE = "personnes_en_mouvement.mp4"  
    
    try:
        bench = MotionDetectionBenchmark(VIDEO_FILE)
        bench.run()
    except Exception as e:
        print(f"Erreur : {e}")
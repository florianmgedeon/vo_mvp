Plan: Essential-Matrix & Pose-Rekonstruktion

Kurz: Füge Kameraintrinsics in Config, erweitere Matching zum Persistieren (Keypoints+Matches), und implementiere stage_recover_pose in main.cpp, das für jedes Frame‑Paar K baut, findEssentialMat per RANSAC aufruft, mit recoverPose R,t gewinnt, Inlier per mask filtert und Posen validiert/speichert (für spätere BA). Ziel: robuste relative Posen für die meisten Paare.

Steps


DONE: 

Config erweitern: Ergänze config.hpp um fx, fy, cx, cy und optionale Parameter ransac_thresh, ransac_prob, min_inliers. (Symbole: fx, fy, cx, cy, ransac_thresh, ransac_prob, min_inliers.)

Matches persistieren: Erweiterung von stage_orb_matching in main.cpp: zusätzlich zu PNG-Visuals speichere pro Paar eine kleine JSON/YAML mit Keypoint‑Positionen und Match‑Indizes (z.B. matches/match_00000.json). Vorteil: deterministisch, debugbar, keine Rekombination nötig.




NOT DONE:

Pro Paar Ablauf (in stage_recover_pose):
Lade Graustufenbilder aus cfg.framesDir (wie in Stage 2).
Lade Keypoints + Match‑Indices aus der JSON; falls nicht vorhanden, recompute ORB+matching wie in stage_orb_matching.
Erzeuge std::vector<cv::Point2f> pts1, pts2 aus gematchten Keypoint‑Koordinaten (Inlier-Order nicht angewendet yet).
Baue K = [[fx, 0, cx],[0, fy, cy],[0,0,1]] aus Config.
Berechne E = findEssentialMat(pts1, pts2, K, cv::RANSAC, ransac_prob, ransac_thresh).
Rufe recoverPose(E, pts1, pts2, K, R, t, mask); filtere pts1/pts2 und Matches mit mask (Inlier-only).
Trianguliere/oder prüfe Cheirality (positive Tiefen) und berechne Anteil positiver Tiefen.
Validierung: akzeptiere Pose nur wenn inliers >= min_inliers UND cheirality_ratio >= 0.7 (konfigurierbar).
Speichere akzeptierte R (3x3) und t (3x1) pro Paar in cfg.matchesDir als JSON/CSV (z.B. pose_00000.json) zusammen mit num_inliers und cheirality_ratio.
Logging & Rückfall: Logge Gründe für Verwerfen (zu wenig Inlier, schlechte Cheirality, zu kleine Translation). Wenn Persistenz fehlt, optional Rekombination mit gleichen Matching-Parametern erlauben.
Integration für SLAM‑Schritt: Sammle akzeptierte R,t in einer Pose‑Liste/Graph-Struktur (in‑memory). Markiere Posen zum späteren Fenster‑BA (z.B. nach N akzeptierten Posen).

Integration für SLAM‑Schritt: Sammle akzeptierte R,t in einer Pose‑Liste/Graph-Struktur (in‑memory). Markiere Posen zum späteren Fenster‑BA (z.B. nach N akzeptierten Posen).
Further Considerations
K‑Quelle: Verwende echte Kalibrierung (Option A). Heuristik (f ≈ max(w,h)) nur als Fallback; dokumentiere das in Config Defaults.
Persistenz‑Format: JSON mit Feldern: keypoints1 (array of [x,y]), keypoints2, matches ([idx1, idx2]), R, t, inliers_mask, num_inliers, cheirality_ratio. Leicht lesbar und portable.
Parameterempfehlungen: ransac_prob=0.999, ransac_thresh=1.0 (kalibriert) — falls verrauscht, 1.5–4.0. min_inliers=30 oder min_inliers = max(30, 0.2 * matches). cheirality_ratio_threshold=0.7.
Weiteres: Nach Baseline implementiere ein kleines windowed BA (Ceres/g2o) über die gesammelten Posen und Inlier‑Korrespondenzen, sowie optional Loop‑Closure.
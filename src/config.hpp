#pragma once


#include <string>

struct Config {
    std::string videoPath = "data/input.mp4";
    std::string framesDir = "data/frames";
    std::string matchesDir = "out/matches";


    int everyN = 6;
    int max_dim = 960;
    int max_frames = 200;
    int max_pairs = 30;

    bool force_recompute = true;
    // Camera intrinsics (default: heuristic fallback)
    double fx = 0.0; // if 0, fallback to heuristic
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    // RANSAC + validation parameters
    double ransac_thresh = 1.0; // in pixels (or normalized) depending on usage
    double ransac_prob = 0.999;
    int min_inliers = 30; // minimal number of inliers to accept a pose
    double cheirality_ratio_threshold = 0.7; // fraction of points with positive depth
};

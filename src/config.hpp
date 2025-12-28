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
};

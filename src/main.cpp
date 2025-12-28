#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>

#include "config.hpp"

namespace fs = std::filesystem;

// ---------- Utils ----------
static cv::Mat resize_max_dim(const cv::Mat& img, int max_dim) {
    if (img.empty()) return img;

    int w = img.cols;
    int h = img.rows;
    int current_max = std::max(w, h);

    if (current_max <= max_dim) return img;

    double scale = static_cast<double>(max_dim) / static_cast<double>(current_max);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);

    cv::Mat out;
    cv::resize(img, out, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    return out;
}

static std::vector<std::string> list_png_files_sorted(const std::string& dir) {
    std::vector<std::string> files;
    if (!fs::exists(dir)) return files;

    for (auto& p : fs::directory_iterator(dir)) {
        if (!p.is_regular_file()) continue;
        if (p.path().extension() == ".png") files.push_back(p.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ---------- Stage 1: Extract frames ----------
static bool stage_extract_frames(const Config& cfg) {
    std::cout << "[Stage 1] Extract frames\n";

    fs::create_directories(cfg.framesDir);

    // If not forcing recompute and frames already exist, skip
    if (!cfg.force_recompute) {
        auto existing = list_png_files_sorted(cfg.framesDir);
        if (!existing.empty()) {
            std::cout << "[Stage 1] Frames already exist (" << existing.size()
                      << "), skipping extraction.\n";
            return true;
        }
    } else {
        // wipe old frames
        for (auto& p : fs::directory_iterator(cfg.framesDir)) {
            if (p.is_regular_file() && p.path().extension() == ".png") fs::remove(p.path());
        }
    }

    cv::VideoCapture cap(cfg.videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[Stage 1] ERROR: Could not open video: " << cfg.videoPath << "\n";
        return false;
    }

    int frameIndex = 0;
    int savedCount = 0;
    cv::Mat frame;

    while (true) {
        bool ok = cap.read(frame);
        if (!ok || frame.empty()) break;

        if (frameIndex % cfg.everyN == 0) {
            cv::Mat resized = resize_max_dim(frame, cfg.max_dim);

            std::ostringstream name;
            name << cfg.framesDir << "/frame_"
                 << std::setw(5) << std::setfill('0') << savedCount
                 << ".png";

            if (!cv::imwrite(name.str(), resized)) {
                std::cerr << "[Stage 1] ERROR: Failed to write " << name.str() << "\n";
                return false;
            }
            savedCount++;
            if (savedCount >= cfg.max_frames) break;
        }

        frameIndex++;
    }

    std::cout << "[Stage 1] Done. Read frames: " << frameIndex
              << ", saved frames: " << savedCount
              << " into " << cfg.framesDir << "\n";

    return (savedCount >= 2);
}

// ---------- Stage 2: ORB detection + matching ----------
static bool stage_orb_matching(const Config& cfg) {
    std::cout << "[Stage 2] ORB detection + matching\n";

    fs::create_directories(cfg.matchesDir);

    // If not forcing recompute and matches already exist (json or png), skip
    if (!cfg.force_recompute) {
        bool any = false;
        if (fs::exists(cfg.matchesDir)) {
            for (auto& p : fs::directory_iterator(cfg.matchesDir)) {
                if (!p.is_regular_file()) continue;
                auto ext = p.path().extension().string();
                if (ext == ".png" || ext == ".json" || ext == ".yml" || ext == ".yaml") {
                    any = true;
                    break;
                }
            }
        }
        if (any) {
            std::cout << "[Stage 2] Matches already exist in " << cfg.matchesDir
                      << ", skipping matching.\n";
            return true;
        }
    } else {
        // wipe old match images and jsons
        for (auto& p : fs::directory_iterator(cfg.matchesDir)) {
            if (!p.is_regular_file()) continue;
            auto ext = p.path().extension().string();
            if (ext == ".png" || ext == ".json" || ext == ".yml" || ext == ".yaml") fs::remove(p.path());
        }
    }

    auto frames = list_png_files_sorted(cfg.framesDir);
    if (frames.size() < 2) {
        std::cerr << "[Stage 2] ERROR: Need at least 2 frames in " << cfg.framesDir << "\n";
        return false;
    }

    auto orb = cv::ORB::create(2000);
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    int maxPairs = std::min<int>(cfg.max_pairs, static_cast<int>(frames.size()) - 1);
    int written = 0;

    for (int i = 0; i < maxPairs; i++) {
        cv::Mat img1 = cv::imread(frames[i], cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(frames[i + 1], cv::IMREAD_GRAYSCALE);
        if (img1.empty() || img2.empty()) {
            std::cerr << "[Stage 2] ERROR: Could not read frames.\n";
            return false;
        }

        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat desc1, desc2;
        orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
        orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);

        if (desc1.empty() || desc2.empty()) {
            std::cout << "[Stage 2] Pair " << i << ": no descriptors, skipping.\n";
            continue;
        }

        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(desc1, desc2, knn, 2);

        std::vector<cv::DMatch> good;
        good.reserve(knn.size());
        const float ratio = 0.75f;

        for (const auto& m : knn) {
            if (m.size() < 2) continue;
            if (m[0].distance < ratio * m[1].distance) good.push_back(m[0]);
        }

        std::cout << "[Stage 2] Pair " << i
                  << " | kp1=" << kp1.size()
                  << " kp2=" << kp2.size()
                  << " goodMatches=" << good.size()
                  << "\n";

        std::sort(good.begin(), good.end(),
                  [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
        if (good.size() > 80) good.resize(80);

        // draw and write match image
        cv::Mat vis;
        cv::drawMatches(img1, kp1, img2, kp2, good, vis);

        std::ostringstream outNameImg;
        outNameImg << cfg.matchesDir << "/match_"
                   << std::setw(5) << std::setfill('0') << i
                   << ".png";

        if (cv::imwrite(outNameImg.str(), vis)) written++;

        // Persist keypoints and matches as JSON using OpenCV FileStorage (JSON format)
        std::ostringstream outNameJson;
        outNameJson << cfg.matchesDir << "/match_"
                    << std::setw(5) << std::setfill('0') << i
                    << ".json";

        try {
            cv::FileStorage fsOut(outNameJson.str(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

            // keypoints1
            fsOut << "keypoints1" << "[";
            for (const auto& k : kp1) {
                fsOut << "{:" << "x" << k.pt.x << "y" << k.pt.y << "}";
            }
            fsOut << "]";

            // keypoints2
            fsOut << "keypoints2" << "[";
            for (const auto& k : kp2) {
                fsOut << "{:" << "x" << k.pt.x << "y" << k.pt.y << "}";
            }
            fsOut << "]";

            // matches as pairs [idx1, idx2]
            fsOut << "matches" << "[";
            for (const auto& m : good) {
                fsOut << "[" << m.queryIdx << m.trainIdx << "]";
            }
            fsOut << "]";

            // basic meta
            fsOut << "num_keypoints1" << static_cast<int>(kp1.size());
            fsOut << "num_keypoints2" << static_cast<int>(kp2.size());
            fsOut << "num_matches" << static_cast<int>(good.size());

            fsOut.release();
        } catch (const std::exception& e) {
            std::cerr << "[Stage 2] WARNING: Failed to write JSON for pair " << i << " : " << e.what() << "\n";
        }
    }

    std::cout << "[Stage 2] Wrote " << written << " match images into " << cfg.matchesDir << "\n";
    return (written > 0);
}

// ---------- main orchestrator ----------
int main() {
    Config cfg; // loads defaults from config.hpp

    // Ensure dirs exist
    fs::create_directories(cfg.framesDir);
    fs::create_directories(cfg.matchesDir);

    if (!stage_extract_frames(cfg)) return 1;
    if (!stage_orb_matching(cfg)) return 1;

    std::cout << "[OK] Pipeline finished.\n";
    return 0;
}

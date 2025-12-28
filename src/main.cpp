#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string>

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

// ---------- Stage 3: Recover pose ----------
bool stage_recover_pose(const Config &cfg) {
    namespace fs = std::filesystem;
    if (!fs::exists(cfg.matchesDir)) {
        std::cerr << "matchesDir does not exist: " << cfg.matchesDir << "\n";
        return false;
    }

    cv::Mat K = (cv::Mat_<double>(3,3) << cfg.fx, 0.0, cfg.cx,
                                          0.0, cfg.fy, cfg.cy,
                                          0.0, 0.0, 1.0);

    int accepted_count = 0;
    for (auto &entry : fs::directory_iterator(cfg.matchesDir)) {
        if (!entry.is_regular_file()) continue;
        std::string path = entry.path().string();
        std::string name = entry.path().filename().string();
        // expect files like match_XXXXX.yml/.json
        if (name.find("match_") == std::string::npos) continue;

        cv::FileStorage fsr(path, cv::FileStorage::READ);
        if (!fsr.isOpened()) {
            std::cerr << "Failed to open match file: " << path << "\n";
            continue;
        }

        std::vector<cv::Point2f> kps1, kps2;
        cv::Mat matchesMat;
        fsr["keypoints1"] >> kps1;
        fsr["keypoints2"] >> kps2;
        fsr["matches"] >> matchesMat;
        fsr.release();

        if (kps1.empty() || kps2.empty() || matchesMat.empty()) {
            std::cerr << "Missing data in " << path << " (will skip)\n";
            continue;
        }

        std::vector<cv::Point2f> pts1, pts2;
        // matchesMat expected Nx2 (ints)
        for (int r = 0; r < matchesMat.rows; ++r) {
            int a = 0, b = 0;
            if (matchesMat.type() == CV_32S || matchesMat.type() == CV_32SC1) {
                a = matchesMat.at<int>(r,0);
                b = matchesMat.at<int>(r,1);
            } else {
                // fallback for float/double stored indices
                a = static_cast<int>(matchesMat.at<double>(r,0));
                b = static_cast<int>(matchesMat.at<double>(r,1));
            }
            if (a < 0 || a >= (int)kps1.size() || b < 0 || b >= (int)kps2.size()) continue;
            pts1.push_back(kps1[a]);
            pts2.push_back(kps2[b]);
        }

        if (pts1.size() < 8) {
            std::cerr << name << ": not enough matched points (" << pts1.size() << ")\n";
            continue;
        }

        // Essential matrix with RANSAC
        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC,
                                         cfg.ransac_prob, cfg.ransac_thresh, mask);
        if (E.empty()) {
            std::cerr << name << ": findEssentialMat failed\n";
            continue;
        }

        cv::Mat R, t;
        int inliers = cv::recoverPose(E, pts1, pts2, K, R, t, mask);
        if (inliers <= 0) {
            std::cerr << name << ": recoverPose yielded zero inliers\n";
            continue;
        }

        // collect inlier points for cheirality test
        std::vector<cv::Point2f> in1, in2;
        std::vector<int> inlierIdx;
        for (size_t i = 0; i < mask.rows && i < pts1.size(); ++i) {
            if (mask.at<unsigned char>(i,0)) {
                in1.push_back(pts1[i]);
                in2.push_back(pts2[i]);
                inlierIdx.push_back((int)i);
            }
        }

        if ((int)in1.size() < cfg.min_inliers) {
            std::cerr << name << ": too few inliers after RANSAC: " << in1.size() << "\n";
            continue;
        }

        // Triangulate and check cheirality
        cv::Mat P0 = cv::Mat::zeros(3,4,CV_64F);
        P0.at<double>(0,0) = 1.0; P0.at<double>(1,1) = 1.0; P0.at<double>(2,2) = 1.0;
        cv::Mat P1 = cv::Mat::zeros(3,4,CV_64F);
        cv::Mat R64, t64;
        R.convertTo(R64, CV_64F);
        t.convertTo(t64, CV_64F);
        R64.copyTo(P1.colRange(0,3));
        t64.copyTo(P1.col(3));

        // convert to camera coordinates: K * [R|t]
        cv::Mat KP0 = K * P0;
        cv::Mat KP1 = K * P1;

        cv::Mat points4d;
        cv::triangulatePoints(KP0, KP1, in1, in2, points4d);

        int positive_depth_count = 0;
        for (int c = 0; c < points4d.cols; ++c) {
            cv::Mat col = points4d.col(c);
            col /= col.at<double>(3,0); // make non-homogeneous
            cv::Mat X = col.rowRange(0,3); // 3x1
            double z1 = X.at<double>(2,0);
            cv::Mat Xc2 = R64 * X + t64;
            double z2 = Xc2.at<double>(2,0);
            if (z1 > 0 && z2 > 0) ++positive_depth_count;
        }
        double cheirality_ratio = double(positive_depth_count) / double(points4d.cols);

        if (cheirality_ratio < 0.7) {
            std::cerr << name << ": bad cheirality ratio " << cheirality_ratio << "\n";
            continue;
        }

        // Accept pose -> save
        std::string pose_name = name;
        // replace "match" with "pose"
        size_t pos = pose_name.find("match");
        if (pos != std::string::npos) pose_name.replace(pos, 5, "pose");
        std::string outpath = (fs::path(cfg.matchesDir) / pose_name).string();

        cv::FileStorage wfs(outpath, cv::FileStorage::WRITE);
        wfs << "R" << R;
        wfs << "t" << t;
        wfs << "num_inliers" << (int)in1.size();
        wfs << "cheirality_ratio" << cheirality_ratio;
        // write inlier mask as vector<int>
        std::vector<int> mask_vec(mask.rows);
        for (int i = 0; i < mask.rows; ++i) mask_vec[i] = mask.at<unsigned char>(i,0) ? 1 : 0;
        wfs << "inliers_mask" << mask_vec;
        wfs.release();

        ++accepted_count;
        std::cout << name << ": accepted pose saved to " << outpath
                  << " (inliers=" << in1.size() << ", cheirality=" << cheirality_ratio << ")\n";
    }

    std::cout << "stage_recover_pose: accepted poses = " << accepted_count << "\n";
    return true;
}

// ---------- main orchestrator ----------
int main() {
    Config cfg; // loads defaults from config.hpp

    // Ensure dirs exist
    fs::create_directories(cfg.framesDir);
    fs::create_directories(cfg.matchesDir);

    if (!stage_extract_frames(cfg)) return 1;
    if (!stage_orb_matching(cfg)) return 1;
    if (!stage_recover_pose(cfg)) return 1;

    std::cout << "[OK] Pipeline finished.\n";
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <sstream>

static cv::Mat resize_max_dim(const cv::Mat& img, int max_dim) {
    if (img.empty()) return img;

    int w = img.cols;
    int h = img.rows;
    int current_max = std::max(w, h);

    if (current_max <= max_dim) {
        return img; // already small enough
    }

    double scale = static_cast<double>(max_dim) / static_cast<double>(current_max);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);

    cv::Mat out;
    cv::resize(img, out, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    return out;
}


int main() {
    const std::string videoPath = "data/input.mp4";
    const std::string framesDir = "data/frames";

    const int everyN = 6;        // take every X th frame
    const int max_dim = 960; // max height of video resized

    // Ensure output folder exists
    std::filesystem::create_directories(framesDir);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open video: " << videoPath << "\n";
        return 1;
    }

    int frameIndex = 0;
    int savedCount = 0;

    cv::Mat frame;

    while (true) {
        bool ok = cap.read(frame);
        if (!ok || frame.empty()) break;

        if (frameIndex % everyN == 0) {
            cv::Mat resized = resize_max_dim(frame, max_dim);

            std::ostringstream name;
            name << framesDir << "/frame_"
                 << std::setw(5) << std::setfill('0') << savedCount
                 << ".png";

            if (!cv::imwrite(name.str(), resized)) {
                std::cerr << "ERROR: Failed to write " << name.str() << "\n";
                return 1;
            }
            savedCount++;
        }

        frameIndex++;
        if (savedCount >= 200) break; // cap to keep MVP small
    }

    std::cout << "Done. Read frames: " << frameIndex
              << ", saved frames: " << savedCount
              << " into " << framesDir << "\n";

    return 0;
}

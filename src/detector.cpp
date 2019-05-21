//
// Created by xpc on 19-4-28.
//
#include <cmath>
#include <queue>
#include "detector.h"


namespace SeetaOCR {

    void Detector::ResizeImage(cv::Mat& inp, cv::Mat& out, int longest_side) {
        float ratio  = 1.0;
        auto width  = (float) inp.cols;
        auto height = (float) inp.rows;

        if (fmax(height, width) > longest_side) {
            ratio = (height > width) ? (longest_side / height): (longest_side / width);
        }

        auto resizedH = (int) (height * ratio);
        auto resizedW = (int) (width * ratio);

        if (resizedH % 32 != 0) {
            resizedH = 32 * ((int)floor(resizedH / 32.0) + 1);
        }

        if (resizedW % 32 != 0) {
            resizedW = 32 * ((int)floor(resizedW / 32.0) + 1);
        }
        cv::resize(inp, out, cv::Size(resizedW, resizedH));
    }

    void Detector::FeedImageToTensor(cv::Mat& inp){
        ResizeImage(inp, resized, longestSide);

        tf::Tensor input_tensor(tf::DT_FLOAT, tf::TensorShape({1, resized.rows, resized.cols, 3}));
        auto input_tensor_ptr = input_tensor.tensor<float, 4>();

        for (int n=0;n<1;++n)
            for(int h = 0 ; h < resized.rows; ++h)
                for(int w = 0; w < resized.cols; ++w)
                    for(int c = 0; c < 3; ++c){
                        input_tensor_ptr(n, h, w, c) = resized.at<cv::Vec3b>(h, w)[2 - c];
                    }

        inputs = {
                {"input_images:0", input_tensor}
        };
    }

    void Detector::Predict(cv::Mat& inp) {

        FeedImageToTensor(inp);
        FetchTensor(inputs, outputs);

        std::map<int, std::vector<cv::Point>> contoursMap;
        PseAdaptor(outputs[0], contoursMap, 0.9, 10, 1);

        float scaleX = (float) inp.cols / resized.cols;
        float scaleY = (float) inp.rows / resized.rows;

        std::vector<Polygon>().swap(polygons);

        for (auto &cnt: contoursMap) {
            cv::Mat boxPts;
            cv::RotatedRect minRect = cv::minAreaRect(cnt.second);
            cv::boxPoints(minRect, boxPts);

            Polygon polygon(boxPts, cv::Size(inp.rows, inp.cols), scaleX, scaleY);
            polygons.emplace_back(polygon);
        }

        if (DEBUG) {
            cv::Mat tmp(inp);
            for (int i=0; i < polygons.size(); ++i) {
                cv::Mat quad;
                std::vector<cv::Point2f> quad_pts = polygons[i].ToQuadROI();
                cv::Mat transmtx = cv::getPerspectiveTransform(polygons[i].ToVec2f(), quad_pts);
                cv::warpPerspective(tmp, quad, transmtx, cv::Size((int)quad_pts[2].x, (int)quad_pts[2].y));
                cv::resize(quad, quad, cv::Size(0, 0), 0.3, 0.3);
                cv::imshow(std::to_string(i), quad);
                cv::polylines(tmp, polygons[i].ToVec2i(), true, cv::Scalar(0, 0, 255), 2);
            }
            cv::resize(tmp, tmp, cv::Size(0, 0), 0.3, 0.3);
            cv::imshow("debug", tmp);
            cv::waitKey(0);
        }
    }

    void Detector::PseAdaptor(tf::Tensor& features,
                              std::map<int, std::vector<cv::Point>>& contours_map,
                              const float thresh,
                              const float min_area,
                              const float ratio) {

        /// get kernels
        auto features_ptr = features.tensor<float, 4>();

        auto N = (int) features.dim_size(0);
        auto H = (int) features.dim_size(1);
        auto W = (int) features.dim_size(2);
        auto C = (int) features.dim_size(3);

        std::vector<cv::Mat> kernels;

        float _thresh = thresh;
        for (int n = 0; n < N; ++n) {
            for (int c = C - 1; c >= 0; --c) {
                cv::Mat kernel(H, W, CV_8UC1);
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        if (features_ptr(n, h, w, c) > _thresh) {
                            kernel.at<uint8_t>(h, w) = 1;
                        } else {
                            kernel.at<uint8_t>(h, w) = 0;
                        }
                    }
                }
                kernels.push_back(kernel);
                _thresh = thresh * ratio;
            }
        }

        /// make label
        cv::Mat label;
        std::map<int, int> areas;
        cv::Mat mask(H, W, CV_32S, cv::Scalar(0));
        cv::connectedComponents(kernels[C - 1], label, 4);

        for (int y = 0; y < label.rows; ++y) {
            for (int x = 0; x < label.cols; ++x) {
                int value = label.at<int32_t>(y, x);
                if (value == 0) continue;
                areas[value] += 1;
            }
        }

        std::queue<cv::Point> queue, next_queue;

        for (int y = 0; y < label.rows; ++y) {
            for (int x = 0; x < label.cols; ++x) {
                int value = label.at<int>(y, x);
                if (value == 0) continue;
                if (areas[value] < min_area) {
                    areas.erase(value);
                    continue;
                }
                cv::Point point(x, y);
                queue.push(point);
                mask.at<int32_t>(y, x) = value;
            }
        }

        /// growing text line
        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};

        for (int idx = C - 2; idx >= 0; --idx) {
            while (!queue.empty()) {
                cv::Point point = queue.front(); queue.pop();
                int x = point.x;
                int y = point.y;
                int value = mask.at<int32_t>(y, x);

                bool is_edge = true;
                for (int d = 0; d < 4; ++d) {
                    int _x = x + dx[d];
                    int _y = y + dy[d];

                    if (_y < 0 || _y >= mask.rows) continue;
                    if (_x < 0 || _x >= mask.cols) continue;
                    if (kernels[idx].at<uint8_t>(_y, _x) == 0) continue;
                    if (mask.at<int32_t>(_y, _x) > 0) continue;

                    cv::Point point_dxy(_x, _y);
                    queue.push(point_dxy);

                    mask.at<int32_t>(_y, _x) = value;
                    is_edge = false;
                }

                if (is_edge) next_queue.push(point);
            }
            std::swap(queue, next_queue);
        }

        /// make contoursMap
        for (int y=0; y < mask.rows; ++y)
            for (int x=0; x < mask.cols; ++x) {
                int idx = mask.at<int32_t>(y, x);
                if (idx == 0) continue;
                contours_map[idx].emplace_back(cv::Point(x, y));
            }
    }
}

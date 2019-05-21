//
// Created by xpc on 19-4-28.
//

#pragma once

#include <opencv2/opencv.hpp>
#include "tensorflow_graph.h"
#include "polygon.h"

namespace tf = tensorflow;

namespace SeetaOCR {

    class Detector: public TFGraph {
    public:
        Detector(const std::string& graph_file)
                : TFGraph(graph_file){
            longestSide = 1024;
            outputTensorNames = {"seg_maps"};
            Init();
        };

        void Predict(cv::Mat& inp);

        std::vector<Polygon> Polygons() { return polygons; }

        void Debug() {DEBUG=true;}

        void Predict(cv::Mat& inp, std::vector<Polygon>& _polygons) {
            Predict(inp);
            _polygons.assign(polygons.begin(), polygons.end());
        }

    protected:
        bool DEBUG=false;
        int longestSide;
        cv::Mat resized;
        std::vector<tf::Tensor> outputs;
        std::vector<Polygon> polygons;
        std::vector<std::pair<std::string, tf::Tensor>> inputs;

        void FeedImageToTensor(cv::Mat& inp);

        void PseAdaptor(tf::Tensor& features,
                        std::map<int, std::vector<cv::Point>>& contours_map,
                        const float thresh,
                        const float min_area,
                        const float ratio);

        void ResizeImage(cv::Mat& inp, cv::Mat& out, int longest_side);
    };
}

/*
+--------+----+----+----+----+------+------+------+------+
|        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
+--------+----+----+----+----+------+------+------+------+
| CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
| CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
| CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
| CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
| CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
| CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
| CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
+--------+----+----+----+----+------+------+------+------+
*/
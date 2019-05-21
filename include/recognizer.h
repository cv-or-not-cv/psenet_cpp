//
// Created by xpc on 19-4-29.
//

#pragma once

#include <opencv2/opencv.hpp>
#include "tensorflow_graph.h"

namespace tf = tensorflow;

namespace SeetaOCR {
    class Recognizer : public TFGraph {
    public:
        Recognizer(const std::string &graph_file,
                   const std::string &label_file)
                : TFGraph(graph_file) {
            imageHeight = 32;
            outputTensorNames = {"indices", "values", "prob"};
            Init();
            LoadLabelFile(label_file);
        };

        void LoadLabelFile(const std::string &label_file);

        void FeedImageToTensor(cv::Mat &inp);

        void FeedImagesToTensor(std::vector<cv::Mat> &inp);

        void Predict();

        void Predict(std::vector<cv::Mat> &inp, std::map<int, std::pair<std::string, float>>& result) {
            FeedImagesToTensor(inp);
            Predict();
            for (auto &d: decoded) { result[d.first] = d.second; }
        }

        void Debug() {DEBUG=true;}

    protected:
        bool DEBUG=false;
        int imageHeight;

        std::map<int, std::string> label;
        std::vector<std::pair<std::string, tf::Tensor>> inputs;
        std::vector<tf::Tensor> outputs;
        std::map<int, std::pair<std::string, float>> decoded;

        void ResizeImage(cv::Mat &inp, cv::Mat &out);

        void ResizeImages(std::vector<cv::Mat> &inp, std::vector<cv::Mat> &out);

        void Decode(tf::Tensor& indices, tf::Tensor& values, tf::Tensor& probs);
    };
}
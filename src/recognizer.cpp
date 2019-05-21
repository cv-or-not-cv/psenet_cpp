//
// Created by xpc on 19-4-29.
//

#include <cmath>
#include "recognizer.h"

namespace SeetaOCR {

    void Recognizer::LoadLabelFile(const std::string &label_file) {
        std::ifstream infile(label_file, std::ios::in);
        std::string line;
        int i = 0;

        std::map<int, std::string>().swap(label);
        while (std::getline(infile, line)) {
            label[i] = line;
            i++;
        }
        infile.close();
    }

    void Recognizer::FeedImageToTensor(cv::Mat &inp) {

        cv::Mat image;
        ResizeImage(inp, image);

        tf::Tensor input_image_tensor(tf::DT_FLOAT, tf::TensorShape({1, image.rows, image.cols, 3}));
        auto input_image_tensor_ptr = input_image_tensor.tensor<float, 4>();

        for (int n = 0; n < 1; ++n)
            for (int h = 0; h < image.rows; ++h)
                for (int w = 0; w < image.cols; ++w)
                    for (int c = 0; c < 3; ++c) {
                        input_image_tensor_ptr(n, h, w, c) = image.at<cv::Vec3b>(h, w)[2 - c];
                    }

        tf::Tensor input_width_tensor(tf::DT_INT32, tf::TensorShape({1}));
        auto input_input_tensor_ptr = input_width_tensor.tensor<int32_t, 1>();
        input_input_tensor_ptr(0) = image.cols;

        inputs = {
                {"input_images:0", input_image_tensor},
                {"input_widths:0", input_width_tensor}
        };
    }

    void Recognizer::FeedImagesToTensor(std::vector<cv::Mat> &inp) {
        std::vector<cv::Mat> images;
        ResizeImages(inp, images);

        int N = (int)images.size();
        int H = images[0].rows;
        int W = images[0].cols;
        int C = 3;

        tf::Tensor input_images_tensor(tf::DT_FLOAT, tf::TensorShape({N, H, W, C}));
        auto input_images_tensor_ptr = input_images_tensor.tensor<float, 4>();

        tf::Tensor input_widths_tensor(tf::DT_INT32, tf::TensorShape({N}));
        auto input_widths_tensor_ptr = input_widths_tensor.tensor<int32_t, 1>();

        for (int n = 0; n < N; ++n) {
            input_widths_tensor_ptr(n) = W;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < 3; ++c) {
                        input_images_tensor_ptr(n, h, w, c) = images[n].at<cv::Vec3b>(h, w)[2 - c];  // BGR -> RGB
                    }
                }
            }
        }

        inputs = {
                {"input_images:0", input_images_tensor},
                {"input_widths:0", input_widths_tensor}
        };

    }

    void Recognizer::ResizeImage(cv::Mat &inp, cv::Mat &out) {
        int widthNew = (int)ceil(32.0 * inp.cols / inp.rows);

        cv::Mat resized;
        cv::resize(inp, resized, cv::Size(widthNew, 32));

        int widthAlign = widthNew;

        if (widthAlign % 32 != 0) {
            widthAlign = 32 * ((int)floor(widthAlign / 32.0) + 1);
        }

        cv::Mat data(32, widthAlign, CV_8UC3, cv::Scalar(0, 0, 0));

        for(int h = 0 ; h < resized.rows; ++h)
            for(int w = 0; w < resized.cols; ++w)
                for(int c = 0; c < 3; ++c){
                    data.at<cv::Vec3b>(h, w)[c] = resized.at<cv::Vec3b>(h, w)[c];
                }
        out = data;
    }

    void Recognizer::ResizeImages(std::vector<cv::Mat> &inp, std::vector<cv::Mat> &out) {

        int widthMax = 0;
        std::vector<int> widths;
        for (auto &img: inp) {
            int widthNew = (int)ceil(32.0 * img.cols / img.rows);
            widths.push_back(widthNew);
            widthMax = widthNew > widthMax ? widthNew : widthMax;
        }

        if (widthMax % 32 != 0) {
            widthMax = 32 * ((int)floor(widthMax / 32.0) + 1);
        }

        cv::Mat resized;
        for (int i = 0; i < widths.size(); ++i) {
            cv::resize(inp[i], resized, cv::Size(widths[i], 32));
            cv::Mat tmp(32, widthMax, CV_8UC3, cv::Scalar(0, 0, 0));
            for(int h = 0 ; h < resized.rows; ++h)
                for(int w = 0; w < resized.cols; ++w)
                    for(int c = 0; c < 3; ++c){
                        tmp.at<cv::Vec3b>(h, w)[c] = resized.at<cv::Vec3b>(h, w)[c];
                    }
            out.push_back(tmp);
        }
    }

    void Recognizer::Predict() {
        FetchTensor(inputs, outputs);
        Decode(outputs[0], outputs[1], outputs[2]);
    }

    void Recognizer::Decode(tf::Tensor& indices, tf::Tensor& values, tf::Tensor& probs) {

        std::map<int, std::pair<std::string, float>>().swap(decoded);

        auto indices_ptr = indices.tensor<int32_t, 2>();
        auto values_ptr = values.tensor<int32_t, 1>();
        auto probs_ptr = probs.tensor<float, 1>();

        int idx = 0;
        std::string str;

        for (int i = 0; i < indices.dim_size(0) ; ++i){

            if (i == indices.dim_size(0) - 1) {
                str += label[values_ptr(i)];
                decoded[idx] = std::make_pair(str, probs_ptr(idx));
                break;
            }

            if (idx != indices_ptr(i, 0)){
                decoded[idx] = std::make_pair(str, probs_ptr(idx));
                str = "";
                idx = indices_ptr(i, 0);
            }

            str += label[values_ptr(i)];
        }

        if (DEBUG) {
            for (auto &d: decoded) {
                std::cout << d.first << " " << d.second.first << " " << d.second.second << std::endl;
            }
        }
    }
}
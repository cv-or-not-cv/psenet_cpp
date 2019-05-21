#include <iostream>
#include <opencv2/opencv.hpp>

#include "detector.h"
#include "recognizer.h"



void DemoAll() {
    std::vector<std::string> demo_images({"/media/seeta/新加卷2/zzsfp/发票驾驶证/JPEGImages/101.jpg",
                                          "/media/seeta/新加卷2/zzsfp/3.png",
                                          "/media/seeta/新加卷2/zzsfp/4.jpg",
                                          "../data/demo/1234.jpg",
                                          "../data/demo/321.jpg",
                                          "../data/demo/ktp1.jpg",

                                          "../data/demo/npwp1.jpg",
                                          "../data/demo/sim1.jpg",
                                          "../data/demo/img_911.jpg",
                                          "../data/demo/12345.jpg"});
    clock_t  start1, end1, start2, end2;

    SeetaOCR::Detector detector = SeetaOCR::Detector("../data/models/psenet.pb");
    SeetaOCR::Recognizer recognizer = SeetaOCR::Recognizer("../data/models/crnn_mobilnet.pb",
                                                           "../data/models/crnn.txt");
//    detector.Debug();
//    recognizer.Debug();

    for (auto &image_path: demo_images) {

        cv::Mat image = cv::imread(image_path);

        start1 = clock();
        std::vector<Polygon> polygons;
        detector.Predict(image, polygons);
        end1 = clock();

        auto dur1 = (double)(end1 - start1);
        printf("Detector with Batchsize: %d,  Use Time: %f s\n", 1, (dur1 / CLOCKS_PER_SEC));

        std::vector<cv::Mat> ROImages;

        for (auto &p: polygons) {
            cv::Mat roi;
            std::vector<cv::Point2f> quad_pts = p.ToQuadROI();
            cv::Mat transmtx = cv::getPerspectiveTransform(p.ToVec2f(), quad_pts);
            cv::warpPerspective(image, roi, transmtx, cv::Size((int)quad_pts[2].x, (int)quad_pts[2].y));
            ROImages.emplace_back(roi);
            cv::polylines(image, p.ToVec2i(), true, cv::Scalar(0, 0, 255), 2);
        }

        start2 = clock();
        std::map<int, std::pair<std::string, float>> decoded;
        recognizer.Predict(ROImages, decoded);
        end2 = clock();

        auto dur2 = (double)(end2 - start2);
        printf("Recognizer with Batchsize: %d,  Use Time: %f s\n", (int) ROImages.size(), (dur2 / CLOCKS_PER_SEC));

        for (auto &d: decoded) {
            if (d.second.second < 0.92) continue;
            int fontFace = CV_FONT_HERSHEY_SIMPLEX;
            double fontScale = 1;
            int thickness = 1;
            int baseline = 0;
            int lineType = 12;
            cv::Point2i textOrg = polygons[d.first].ToVec2i()[0];
            std::string text = d.second.first; //+ " " + std::to_string(d.second.second);
            cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
            //cv::rectangle(image, textOrg + cv::Point(0, baseline), textOrg + cv::Point(textSize.width, -textSize.height),
            //          cv::Scalar(255, 255, 255), -1, lineType);
            //cv::putText(image, text, polygons[d.first].ToVec2i()[0], fontFace, fontScale, cv::Scalar(0, 0, 0), thickness, lineType);
            std::cout << d.first << " " << d.second.first << " " << d.second.second << std::endl;
        }
        cv::namedWindow("demo", CV_WINDOW_NORMAL);
        cv::imshow("demo", image);
        cv::waitKey(0);
    }
}


int main(){
    DemoAll();
}
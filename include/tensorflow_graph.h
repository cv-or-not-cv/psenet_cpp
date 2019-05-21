//
// Created by seeta on 19-4-28.
//

#pragma once

#include <string>
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;

class TFGraph {
public:
    TFGraph(const std::string& graph_file){
        graphFile = graph_file;
    }
    TFGraph(const std::string& graph_file,
            const std::vector<std::string>& output_tensor_names) {
        graphFile = graph_file;
        outputTensorNames = output_tensor_names;
    }

protected:
    void Init(){
        tf::NewSession(tf::SessionOptions(), &session);

        statusLoad = tf::ReadBinaryProto(tf::Env::Default(), graphFile, &graphdef); //从pb文件中读取图模型;
        if (!statusLoad.ok()) {
            throw std::runtime_error("Loading model failed...\n" + statusLoad.ToString());
        }

        statusCreate = session->Create(graphdef);
        if (!statusCreate.ok()) {
            throw std::runtime_error("Creating graph in session failed...\n" + statusCreate.ToString());
        }
    }

    void FetchTensor(std::vector<std::pair<std::string, tf::Tensor>>& inputs,
                     std::vector<tf::Tensor>& outputs) {

        statusRun = session->Run(inputs, outputTensorNames, targetNodeNames, &outputs);

        if(!statusRun.ok()){
            throw std::runtime_error("Session Run failed...\n" + statusRun.ToString());
        }
    }

    tf::Session* session;
    tf::GraphDef graphdef;
    tf::Status statusRun;
    tf::Status statusLoad;
    tf::Status statusCreate;
    std::string graphFile;
    std::vector<std::string> outputTensorNames;
    std::vector<std::string> targetNodeNames = {};
};


// headers in STL
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
// headers in 3rd-part
#include "../pointpillars/include/pointpillars.h"
#include "gtest/gtest.h"
using namespace std;

const int input_num_feature = 4;

int Bin2Arrary(float* &points_array , string file_name , int num_feature)
{
  ifstream infile(file_name, ios::binary);
  assert(infile.is_open());

  // 获取文件大小
  infile.seekg(0, ios::end);
  size_t file_size = infile.tellg();
  infile.seekg(0, ios::beg);
  if (file_size % (sizeof(float) * num_feature) != 0)
    std::cerr << "Warning: file size does not match expected format!" << std::endl;

  // 计算点的数量：文件大小 / (每个特征的字节数 * 特征数量)
  size_t total_floats = file_size / sizeof(float);
  size_t num_points = total_floats / num_feature;


  // 分配内存并读取二进制数据
  points_array = new float[total_floats];

  // reinterpret_cast不改变内存内容，只改变解释内存的方式。
  // float 存的还是 float 的数据，只是把起始地址作为 char*（一个字节一个字节读取）。
  infile.read(reinterpret_cast<char*>(points_array), file_size);
  infile.close();

  return static_cast<int>(num_points);
};

void Boxes2Txt( std::vector<float> boxes , string file_name , int num_feature = 7)
{
    ofstream ofFile;
    ofFile.open(file_name , std::ios::out );  
    if (ofFile.is_open()) {
        for (int i = 0 ; i < boxes.size() / num_feature ; ++i) {
            for (int j = 0 ; j < num_feature ; ++j) {
                ofFile << boxes.at(i * num_feature + j) << " ";
            }
            ofFile << "\n";
        }
    }
    ofFile.close();
    return ;
};

void SaveBoxPred(std::vector<BoundingBox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << box.w << " ";
          ofs << box.l << " ";
          ofs << box.h << " ";
          ofs << box.rt << " ";
          ofs << box.id << " ";
          ofs << box.score << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};

TEST(PointPillars, __build_model__) {
  const std::string DB_CONF = "/home/gef/PointPillars_MultiHead_40FPS/bootstrap.yaml";
  YAML::Node config = YAML::LoadFile(DB_CONF);

  std::string pfe_file,backbone_file; 
  if(config["UseOnnx"].as<bool>()) {
    pfe_file = config["PfeOnnx"].as<std::string>();
    backbone_file = config["BackboneOnnx"].as<std::string>();
  }else {
    pfe_file = config["PfeTrt"].as<std::string>();
    backbone_file = config["BackboneTrt"].as<std::string>();
  }
  std::cout << backbone_file << std::endl;
  const std::string pp_config = config["ModelConfig"].as<std::string>();
  PointPillars pp(
    config["ScoreThreshold"].as<float>(),
    config["NmsOverlapThreshold"].as<float>(),
    config["UseOnnx"].as<bool>(),
    pfe_file,
    backbone_file,
    pp_config
  ); 
  std::string file_name = config["InputFile"].as<std::string>();
  float* points_array;
  int in_num_points;
  in_num_points = Bin2Arrary(points_array, file_name, input_num_feature);

  
  for (int _ = 0 ; _ < 10 ; _++)
  {
    std::vector<float> out_detections;
    std::vector<int> out_labels;
    std::vector<float> out_scores;

    cudaDeviceSynchronize();
    auto bboxes = pp.DoInference(points_array, in_num_points, &out_detections, &out_labels , &out_scores);
    cudaDeviceSynchronize();

    std::string boxes_file_name = config["OutputFile"].as<std::string>();
    SaveBoxPred(bboxes, boxes_file_name);
    std::cout << ">>>>>>>>>>>" << std::endl;
  }


};

#pragma once

/**
 * @brief 3D 检测框基础结构体
 */
struct BoundingBox {
  float x;
  float y;
  float z;
  float w;
  float l;
  float h;
  float rt;
  int id;
  float score;

  BoundingBox() = default;
  BoundingBox(float x_,
              float y_,
              float z_,
              float w_,
              float l_,
              float h_,
              float rt_,
              int id_,
              float score_)
      : x(x_),
        y(y_),
        z(z_),
        w(w_),
        l(l_),
        h(h_),
        rt(rt_),
        id(id_),
        score(score_) {}
};

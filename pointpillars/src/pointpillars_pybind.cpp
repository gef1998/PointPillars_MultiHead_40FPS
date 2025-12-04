#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "pointpillars.h"
#include "bbox.h"

namespace py = pybind11;

// Python端的BoundingBox封装
void declare_boundingbox(py::module &m) {
    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        .def_readwrite("x", &BoundingBox::x)
        .def_readwrite("y", &BoundingBox::y)
        .def_readwrite("z", &BoundingBox::z)
        .def_readwrite("w", &BoundingBox::w)
        .def_readwrite("l", &BoundingBox::l)
        .def_readwrite("h", &BoundingBox::h)
        .def_readwrite("rt", &BoundingBox::rt)
        .def_readwrite("id", &BoundingBox::id)
        .def_readwrite("score", &BoundingBox::score);
}

// pybind11 主模块
PYBIND11_MODULE(pointpillars_py, m) {
    declare_boundingbox(m);

    py::class_<PointPillars>(m, "PointPillars")
        .def(py::init<float, float, bool, std::string, std::string, std::string>())

        // 接收 numpy array: shape = (N, 4 or 5), dtype float32
        .def("DoInference",
             [](PointPillars &self, py::array_t<float, py::array::c_style | py::array::forcecast> points) {

                 // --- check array format ---
                 auto buf = points.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("points numpy array must be 2D (N x 4 or N x 5)");

                 int in_num_points = buf.shape[0];
                 int feat_dim = buf.shape[1];

                 if (!(feat_dim == 4 || feat_dim == 5))
                     throw std::runtime_error("Each point must have 4 or 5 features");

                 const float *ptr = static_cast<const float *>(buf.ptr);

                 // Output vectors
                 std::vector<float> out_detections;
                 std::vector<int> out_labels;
                 std::vector<float> out_scores;

                 // --- call C++ inference (zero-copy) ---
                 auto boxes = self.DoInference(
                     ptr, in_num_points,
                     &out_detections, &out_labels, &out_scores
                 );

                 // Python 返回: Tuple( List[BoundingBox], List[float], List[int], List[float] )
                 return py::make_tuple(boxes, out_detections, out_labels, out_scores);
             },
             py::arg("points"));
}

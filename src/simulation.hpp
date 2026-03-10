#pragma once

#include <GL/glew.h>

#include <cstddef>

enum class FieldMode {
    Hopfion = 0,
    Torus = 1,
};

class Simulation {
public:
    Simulation() = default;
    ~Simulation();

    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    bool initialize(std::size_t linesPerFamily, std::size_t pointsPerLine, FieldMode fieldMode, int torusP, int torusQ);
    void setFieldConfig(FieldMode fieldMode, int torusP, int torusQ);
    void update(float dt, float timeSeconds);
    void shutdown();

    GLuint vertexBuffer() const { return vertexBuffer_; }
    std::size_t particleCount() const { return particleCount_; }

private:
    std::size_t linesPerFamily_ = 0;
    std::size_t pointsPerLine_ = 0;
    std::size_t particleCount_ = 0;
    FieldMode fieldMode_ = FieldMode::Hopfion;
    int torusP_ = 2;
    int torusQ_ = 3;
    GLuint vertexBuffer_ = 0;

    struct cudaGraphicsResource* interopResource_ = nullptr;
};

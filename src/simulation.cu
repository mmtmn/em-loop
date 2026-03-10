#include "simulation.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kGoldenAngle = 2.39996322972865332f;
constexpr float kSpatialScale = 1.75f;
constexpr float kTimeScale = 0.42f;
constexpr float kTraceStep = 0.075f;

struct ParticleVertex {
    float4 positionSize;
    float4 color;
};

struct Complex {
    float re;
    float im;
};

struct Complex3 {
    Complex x;
    Complex y;
    Complex z;
};

struct FieldSample {
    float3 electric;
    float3 magnetic;
    float3 poynting;
    float energy;
};

__host__ __device__ float3 add3(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 mul3(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 cross3(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ float lengthSquared3(const float3& v) {
    return dot3(v, v);
}

__host__ __device__ float length3(const float3& v) {
    return sqrtf(lengthSquared3(v));
}

__host__ __device__ float3 normalize3(const float3& v) {
    const float len = length3(v);
    if (len <= 1.0e-6f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return mul3(v, 1.0f / len);
}

__host__ __device__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__host__ __device__ float smoothstep(float edge0, float edge1, float x) {
    const float t = clamp01((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

__host__ __device__ float3 mix3(const float3& a, const float3& b, float t) {
    return add3(mul3(a, 1.0f - t), mul3(b, t));
}

__host__ __device__ Complex makeComplex(float re, float im) {
    return {re, im};
}

__host__ __device__ Complex addComplex(const Complex& a, const Complex& b) {
    return {a.re + b.re, a.im + b.im};
}

__host__ __device__ Complex subComplex(const Complex& a, const Complex& b) {
    return {a.re - b.re, a.im - b.im};
}

__host__ __device__ Complex mulComplex(const Complex& a, const Complex& b) {
    return {
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re,
    };
}

__host__ __device__ Complex mulComplexScalar(const Complex& a, float s) {
    return {a.re * s, a.im * s};
}

__host__ __device__ Complex reciprocalComplex(const Complex& z) {
    const float denom = z.re * z.re + z.im * z.im + 1.0e-12f;
    return {z.re / denom, -z.im / denom};
}

__host__ __device__ Complex divComplex(const Complex& a, const Complex& b) {
    return mulComplex(a, reciprocalComplex(b));
}

__host__ __device__ Complex powComplexInt(Complex base, int exponent) {
    if (exponent <= 0) {
        return makeComplex(1.0f, 0.0f);
    }

    Complex result = makeComplex(1.0f, 0.0f);
    for (int i = 0; i < exponent; ++i) {
        result = mulComplex(result, base);
    }
    return result;
}

__host__ __device__ Complex3 makeComplex3(const Complex& x, const Complex& y, const Complex& z) {
    return {x, y, z};
}

__host__ __device__ Complex3 subComplex3(const Complex3& a, const Complex3& b) {
    return {
        subComplex(a.x, b.x),
        subComplex(a.y, b.y),
        subComplex(a.z, b.z),
    };
}

__host__ __device__ Complex3 mulComplex3Scalar(const Complex3& v, const Complex& scalar) {
    return {
        mulComplex(v.x, scalar),
        mulComplex(v.y, scalar),
        mulComplex(v.z, scalar),
    };
}

__host__ __device__ Complex3 divComplex3Scalar(const Complex3& v, const Complex& scalar) {
    return mulComplex3Scalar(v, reciprocalComplex(scalar));
}

__host__ __device__ Complex3 crossComplex3(const Complex3& a, const Complex3& b) {
    return {
        subComplex(mulComplex(a.y, b.z), mulComplex(a.z, b.y)),
        subComplex(mulComplex(a.z, b.x), mulComplex(a.x, b.z)),
        subComplex(mulComplex(a.x, b.y), mulComplex(a.y, b.x)),
    };
}

__device__ FieldSample sampleHopfionField(float3 position, float timeSeconds) {
    // Exact Hopfion field from Kedia et al. (2013), Eq. (8), with F = E + iB.
    const float x = position.x / kSpatialScale;
    const float y = position.y / kSpatialScale;
    const float z = position.z / kSpatialScale;
    const float t = timeSeconds * kTimeScale;

    const float r2 = x * x + y * y + z * z;

    const Complex a = makeComplex(x, -y);
    const Complex b = makeComplex(t - z, -1.0f);
    const Complex d = makeComplex(r2 - t * t + 1.0f, 2.0f * t);

    const Complex a2 = mulComplex(a, a);
    const Complex b2 = mulComplex(b, b);
    const Complex d3 = mulComplex(mulComplex(d, d), d);
    const Complex invD3 = reciprocalComplex(d3);

    const Complex numeratorX = subComplex(b2, a2);
    const Complex sum = addComplex(a2, b2);
    const Complex numeratorY = makeComplex(sum.im, -sum.re);
    const Complex numeratorZ = mulComplexScalar(mulComplex(a, b), 2.0f);

    const Complex fx = mulComplex(numeratorX, invD3);
    const Complex fy = mulComplex(numeratorY, invD3);
    const Complex fz = mulComplex(numeratorZ, invD3);

    const float amplitudeScale = 1.0f / (kSpatialScale * kSpatialScale);
    const float3 electric = mul3(make_float3(fx.re, fy.re, fz.re), amplitudeScale);
    const float3 magnetic = mul3(make_float3(fx.im, fy.im, fz.im), amplitudeScale);
    const float3 poynting = normalize3(cross3(electric, magnetic));
    const float energy = 0.5f * (lengthSquared3(electric) + lengthSquared3(magnetic));

    FieldSample result{};
    result.electric = electric;
    result.magnetic = magnetic;
    result.poynting = poynting;
    result.energy = energy;
    return result;
}

__device__ void sampleBatemanScalars(float3 position, float timeSeconds, Complex& alpha, Complex& beta, Complex3& gradAlpha, Complex3& gradBeta) {
    const float x = position.x / kSpatialScale;
    const float y = position.y / kSpatialScale;
    const float z = position.z / kSpatialScale;
    const float t = timeSeconds * kTimeScale;
    const float r2 = x * x + y * y + z * z;

    const Complex d = makeComplex(r2 - t * t + 1.0f, 2.0f * t);
    const Complex d2 = mulComplex(d, d);
    const Complex numeratorAlpha = makeComplex(r2 - t * t - 1.0f, 2.0f * z);
    const Complex numeratorBeta = makeComplex(2.0f * x, -2.0f * y);

    alpha = divComplex(numeratorAlpha, d);
    beta = divComplex(numeratorBeta, d);

    const Complex3 gradNumeratorAlpha = makeComplex3(
        makeComplex(2.0f * x, 0.0f),
        makeComplex(2.0f * y, 0.0f),
        makeComplex(2.0f * z, 2.0f)
    );
    const Complex3 gradNumeratorBeta = makeComplex3(
        makeComplex(2.0f, 0.0f),
        makeComplex(0.0f, -2.0f),
        makeComplex(0.0f, 0.0f)
    );
    const Complex3 gradD = makeComplex3(
        makeComplex(2.0f * x, 0.0f),
        makeComplex(2.0f * y, 0.0f),
        makeComplex(2.0f * z, 0.0f)
    );

    gradAlpha = divComplex3Scalar(
        subComplex3(
            mulComplex3Scalar(gradNumeratorAlpha, d),
            mulComplex3Scalar(gradD, numeratorAlpha)
        ),
        d2
    );

    gradBeta = divComplex3Scalar(
        subComplex3(
            mulComplex3Scalar(gradNumeratorBeta, d),
            mulComplex3Scalar(gradD, numeratorBeta)
        ),
        d2
    );
}

__device__ FieldSample sampleTorusField(float3 position, float timeSeconds, int p, int q) {
    Complex alpha{};
    Complex beta{};
    Complex3 gradAlpha{};
    Complex3 gradBeta{};
    sampleBatemanScalars(position, timeSeconds, alpha, beta, gradAlpha, gradBeta);

    const Complex alphaPower = powComplexInt(alpha, p - 1);
    const Complex betaPower = powComplexInt(beta, q - 1);
    const Complex prefactor = mulComplexScalar(mulComplex(alphaPower, betaPower), static_cast<float>(p * q));
    const Complex3 baseField = crossComplex3(gradAlpha, gradBeta);
    const Complex3 field = mulComplex3Scalar(baseField, prefactor);

    const float amplitudeScale = 1.0f / (kSpatialScale * kSpatialScale);
    const float3 electric = mul3(make_float3(field.x.re, field.y.re, field.z.re), amplitudeScale);
    const float3 magnetic = mul3(make_float3(field.x.im, field.y.im, field.z.im), amplitudeScale);
    const float3 poynting = normalize3(cross3(electric, magnetic));
    const float energy = 0.5f * (lengthSquared3(electric) + lengthSquared3(magnetic));

    FieldSample result{};
    result.electric = electric;
    result.magnetic = magnetic;
    result.poynting = poynting;
    result.energy = energy;
    return result;
}

__device__ FieldSample sampleField(float3 position, float timeSeconds, int fieldMode, int p, int q) {
    return fieldMode == static_cast<int>(FieldMode::Torus)
        ? sampleTorusField(position, timeSeconds, p, q)
        : sampleHopfionField(position, timeSeconds);
}

__device__ float3 hopfSeed(int lineIndex, int linesPerFamily, int family, int p, int q) {
    const float u = (static_cast<float>(lineIndex) + 0.5f) / static_cast<float>(linesPerFamily);
    const float pWeight = static_cast<float>(p) / static_cast<float>(p + q);
    const float qWeight = static_cast<float>(q) / static_cast<float>(p + q);
    const float etaCenter = atanf(sqrtf(fmaxf(pWeight, 1.0e-4f) / fmaxf(qWeight, 1.0e-4f)));
    const bool emphasizeCore = (lineIndex % 6) == 0;
    const float etaSpread = (0.34f + 0.06f * (family == 1 ? 1.0f : 0.0f)) * (emphasizeCore ? 0.20f : 1.0f);
    const float eta = fminf(fmaxf(etaCenter + (u - 0.5f) * etaSpread, 0.08f), 0.92f * kPi * 0.5f);
    const float phaseA = kGoldenAngle * static_cast<float>(lineIndex) + (family == 1 ? 0.45f * kPi : 0.0f);
    const float phaseB = (-static_cast<float>(p) / static_cast<float>(q)) * phaseA +
        (family == 1 ? -0.25f * kPi / static_cast<float>(q) : 0.20f * kPi / static_cast<float>(q));

    const Complex u0 = makeComplex(cosf(eta) * cosf(phaseA), cosf(eta) * sinf(phaseA));
    const Complex v0 = makeComplex(sinf(eta) * cosf(phaseB), sinf(eta) * sinf(phaseB));

    const float denom = fmaxf(1.0f - u0.re, 0.08f);
    const float x = v0.re / denom;
    const float y = -v0.im / denom;
    const float z = u0.im / denom;

    return mul3(make_float3(x, y, z), 0.95f * kSpatialScale);
}

__device__ float3 streamlineDirection(const FieldSample& field, int family) {
    return normalize3(family == 0 ? field.magnetic : field.electric);
}

__device__ float3 rk4StreamlineStep(float3 position, float ds, float timeSeconds, int family, int fieldMode, int p, int q) {
    const float3 k1 = streamlineDirection(sampleField(position, timeSeconds, fieldMode, p, q), family);
    const float3 k2 = streamlineDirection(sampleField(add3(position, mul3(k1, ds * 0.5f)), timeSeconds, fieldMode, p, q), family);
    const float3 k3 = streamlineDirection(sampleField(add3(position, mul3(k2, ds * 0.5f)), timeSeconds, fieldMode, p, q), family);
    const float3 k4 = streamlineDirection(sampleField(add3(position, mul3(k3, ds)), timeSeconds, fieldMode, p, q), family);

    const float3 weighted = add3(
        add3(k1, mul3(k2, 2.0f)),
        add3(mul3(k3, 2.0f), k4)
    );

    return add3(position, mul3(weighted, ds / 6.0f));
}

__device__ float3 traceSamplePoint(int lineIndex, int sampleIndex, int linesPerFamily, int pointsPerLine, int family, float timeSeconds, int fieldMode, int p, int q) {
    const int seedP = fieldMode == static_cast<int>(FieldMode::Torus) ? p : 1;
    const int seedQ = fieldMode == static_cast<int>(FieldMode::Torus) ? q : 1;
    float3 position = hopfSeed(lineIndex, linesPerFamily, family, seedP, seedQ);

    const float centered = static_cast<float>(sampleIndex) - 0.5f * static_cast<float>(pointsPerLine - 1);
    const int stepCount = abs(static_cast<int>(centered));
    const float directionSign = centered < 0.0f ? -1.0f : 1.0f;

    for (int step = 0; step < stepCount; ++step) {
        position = rk4StreamlineStep(position, directionSign * kTraceStep, timeSeconds, family, fieldMode, p, q);

        const float radius2 = lengthSquared3(position);
        if (radius2 > 38.0f) {
            position = mul3(normalize3(position), 6.0f);
            break;
        }
    }

    return position;
}

__device__ void writeVertex(ParticleVertex& vertex, float3 position, int lineIndex, int sampleIndex, int linesPerFamily, int pointsPerLine, int family, float timeSeconds, int fieldMode, int p, int q) {
    const FieldSample field = sampleField(position, timeSeconds, fieldMode, p, q);
    const float3 direction = streamlineDirection(field, family);
    const float energy = field.energy;

    const float normalizedLine = (static_cast<float>(lineIndex) + 0.5f) / static_cast<float>(linesPerFamily);
    const float normalizedSample = (static_cast<float>(sampleIndex) + 0.5f) / static_cast<float>(pointsPerLine);
    const float pulse = 0.5f + 0.5f * sinf(timeSeconds * 0.9f + normalizedLine * 9.0f + normalizedSample * 5.0f);
    const float edgeFade = sinf(normalizedSample * kPi);
    const float energyBoost = smoothstep(0.002f, 0.060f, energy);

    float3 baseColor = make_float3(0.10f, 0.72f, 1.10f);
    if (family == 1) {
        baseColor = make_float3(1.10f, 0.60f, 0.16f);
    }

    const float3 poyntingTint = mix3(make_float3(0.95f, 0.96f, 1.00f), make_float3(0.65f, 0.88f, 1.00f), 0.5f + 0.5f * field.poynting.y);
    const float3 color = mul3(
        mix3(baseColor, poyntingTint, 0.18f + 0.12f * pulse),
        (0.35f + 1.65f * energyBoost) * (0.45f + 0.55f * edgeFade)
    );

    const float size = 4.2f + 3.6f * energyBoost + 1.4f * pulse + 0.6f * fabsf(direction.y);

    vertex.positionSize = make_float4(position.x, position.y, position.z, size);
    vertex.color = make_float4(color.x, color.y, color.z, 1.0f);
}

__global__ void generateFieldLineVertices(ParticleVertex* vertices, int linesPerFamily, int pointsPerLine, float timeSeconds, int fieldMode, int p, int q) {
    const int totalVertexCount = linesPerFamily * pointsPerLine * 2;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalVertexCount) {
        return;
    }

    const int familyVertexCount = linesPerFamily * pointsPerLine;
    const int family = index / familyVertexCount;
    const int localIndex = index - family * familyVertexCount;
    const int lineIndex = localIndex / pointsPerLine;
    const int sampleIndex = localIndex % pointsPerLine;

    const float3 position = traceSamplePoint(lineIndex, sampleIndex, linesPerFamily, pointsPerLine, family, timeSeconds, fieldMode, p, q);
    writeVertex(vertices[index], position, lineIndex, sampleIndex, linesPerFamily, pointsPerLine, family, timeSeconds, fieldMode, p, q);
}

bool checkCuda(cudaError_t status, const char* what) {
    if (status == cudaSuccess) {
        return true;
    }

    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    return false;
}

}  // namespace

Simulation::~Simulation() {
    shutdown();
}

bool Simulation::initialize(std::size_t linesPerFamily, std::size_t pointsPerLine, FieldMode fieldMode, int torusP, int torusQ) {
    shutdown();

    int deviceCount = 0;
    if (!checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount")) {
        return false;
    }
    if (deviceCount <= 0) {
        std::fprintf(stderr, "No CUDA device found\n");
        return false;
    }
    if (!checkCuda(cudaSetDevice(0), "cudaSetDevice")) {
        return false;
    }

    linesPerFamily_ = linesPerFamily;
    pointsPerLine_ = pointsPerLine;
    particleCount_ = linesPerFamily_ * pointsPerLine_ * 2;
    fieldMode_ = fieldMode;
    torusP_ = torusP;
    torusQ_ = torusQ;

    glGenBuffers(1, &vertexBuffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(particleCount_ * sizeof(ParticleVertex)), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (!checkCuda(cudaGraphicsGLRegisterBuffer(&interopResource_, vertexBuffer_, cudaGraphicsRegisterFlagsWriteDiscard),
                   "cudaGraphicsGLRegisterBuffer")) {
        shutdown();
        return false;
    }

    update(0.0f, 0.0f);
    return true;
}

void Simulation::setFieldConfig(FieldMode fieldMode, int torusP, int torusQ) {
    fieldMode_ = fieldMode;
    torusP_ = torusP;
    torusQ_ = torusQ;
}

void Simulation::update(float, float timeSeconds) {
    if (interopResource_ == nullptr || particleCount_ == 0) {
        return;
    }

    ParticleVertex* mappedVertices = nullptr;
    std::size_t mappedSize = 0;
    if (!checkCuda(cudaGraphicsMapResources(1, &interopResource_), "cudaGraphicsMapResources")) {
        return;
    }
    if (!checkCuda(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedVertices), &mappedSize, interopResource_),
                   "cudaGraphicsResourceGetMappedPointer")) {
        cudaGraphicsUnmapResources(1, &interopResource_);
        return;
    }

    constexpr int threadsPerBlock = 256;
    const int blocks = static_cast<int>((particleCount_ + threadsPerBlock - 1) / threadsPerBlock);
    generateFieldLineVertices<<<blocks, threadsPerBlock>>>(
        mappedVertices,
        static_cast<int>(linesPerFamily_),
        static_cast<int>(pointsPerLine_),
        timeSeconds,
        static_cast<int>(fieldMode_),
        torusP_,
        torusQ_
    );
    checkCuda(cudaGetLastError(), "generateFieldLineVertices launch");
    checkCuda(cudaGraphicsUnmapResources(1, &interopResource_), "cudaGraphicsUnmapResources");
}

void Simulation::shutdown() {
    if (interopResource_ != nullptr) {
        cudaGraphicsUnregisterResource(interopResource_);
        interopResource_ = nullptr;
    }

    if (vertexBuffer_ != 0) {
        glDeleteBuffers(1, &vertexBuffer_);
        vertexBuffer_ = 0;
    }

    linesPerFamily_ = 0;
    pointsPerLine_ = 0;
    particleCount_ = 0;
    fieldMode_ = FieldMode::Hopfion;
    torusP_ = 2;
    torusQ_ = 3;
}

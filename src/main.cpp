#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "math.hpp"
#include "shaders.hpp"
#include "simulation.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct QualityPreset {
    const char* name;
    int windowWidth;
    int windowHeight;
    std::size_t linesPerFamily;
    std::size_t pointsPerLine;
    int blurPasses;
    int bloomDivisor;
    int volumeSteps;
};

struct RunConfig {
    const QualityPreset* quality = nullptr;
    FieldMode fieldMode = FieldMode::Hopfion;
    int torusP = 2;
    int torusQ = 3;
};

struct NamedFieldPreset {
    const char* cliName;
    const char* displayName;
    FieldMode fieldMode;
    int torusP;
    int torusQ;
};

enum class FilterMode {
    Off = 0,
    Filmic = 1,
    Aurora = 2,
    Solar = 3,
    NoGlow = 4,
};

constexpr QualityPreset kQualityLow{
    "low",
    1280,
    720,
    40,
    96,
    4,
    2,
    28,
};

constexpr QualityPreset kQualityMedium{
    "medium",
    1280,
    720,
    56,
    112,
    5,
    2,
    36,
};

constexpr QualityPreset kQualityHigh{
    "high",
    1600,
    900,
    72,
    128,
    6,
    2,
    44,
};

constexpr QualityPreset kQualityUltra{
    "ultra",
    1920,
    1080,
    96,
    144,
    8,
    1,
    56,
};

constexpr NamedFieldPreset kPresetHopfion{"hopfion", "HOPFION", FieldMode::Hopfion, 2, 3};
constexpr NamedFieldPreset kPresetTrefoil{"trefoil", "TREFOIL", FieldMode::Torus, 2, 3};
constexpr NamedFieldPreset kPresetCinquefoil{"cinquefoil", "CINQUEFOIL", FieldMode::Torus, 2, 5};
constexpr NamedFieldPreset kPresetLinkedRings{"linked-rings", "LINKED-RINGS", FieldMode::Torus, 2, 2};

struct Camera {
    Vec3 position{0.0f, 0.0f, 5.2f};
    float yaw = -1.57079632679f;
    float pitch = 0.0f;
    float verticalFov = radians(55.0f);

    Vec3 forward() const {
        const float cosPitch = std::cos(pitch);
        return normalize({
            std::cos(yaw) * cosPitch,
            std::sin(pitch),
            std::sin(yaw) * cosPitch,
        });
    }

    Vec3 right() const {
        return normalize(cross(forward(), {0.0f, 1.0f, 0.0f}));
    }

    Vec3 up() const {
        return normalize(cross(right(), forward()));
    }

    void reset() {
        position = {0.0f, 0.0f, 5.2f};
        yaw = -1.57079632679f;
        pitch = 0.0f;
    }
};

struct InputState {
    bool mouseCaptured = true;
    bool firstMouseSample = true;
    bool tabHeld = false;
    bool resetHeld = false;
    bool filterHeld = false;
    bool preset1Held = false;
    bool preset2Held = false;
    bool preset3Held = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
};

struct RuntimeState {
    RunConfig config;
    FilterMode filterMode = FilterMode::Filmic;
    float simulationStartTime = 0.0f;
};

struct AppState {
    Camera camera;
    InputState input;
    RuntimeState runtime;
    int framebufferWidth = 1280;
    int framebufferHeight = 720;
    bool framebufferResized = false;
};

struct SceneFramebuffer {
    GLuint fbo = 0;
    GLuint colorTexture = 0;
    GLuint brightTexture = 0;
    GLuint depthBuffer = 0;

    void destroy() {
        if (depthBuffer != 0) {
            glDeleteRenderbuffers(1, &depthBuffer);
            depthBuffer = 0;
        }
        if (colorTexture != 0) {
            glDeleteTextures(1, &colorTexture);
            colorTexture = 0;
        }
        if (brightTexture != 0) {
            glDeleteTextures(1, &brightTexture);
            brightTexture = 0;
        }
        if (fbo != 0) {
            glDeleteFramebuffers(1, &fbo);
            fbo = 0;
        }
    }

    void create(int width, int height) {
        destroy();

        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glGenTextures(1, &colorTexture);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);

        glGenTextures(1, &brightTexture);
        glBindTexture(GL_TEXTURE_2D, brightTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, brightTexture, 0);

        glGenRenderbuffers(1, &depthBuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

        const GLenum attachments[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, attachments);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Scene framebuffer is incomplete");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

struct PingPongBuffers {
    std::array<GLuint, 2> fbos{};
    std::array<GLuint, 2> textures{};

    void destroy() {
        glDeleteFramebuffers(static_cast<GLsizei>(fbos.size()), fbos.data());
        glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
        fbos = {};
        textures = {};
    }

    void create(int width, int height) {
        destroy();

        glGenFramebuffers(static_cast<GLsizei>(fbos.size()), fbos.data());
        glGenTextures(static_cast<GLsizei>(textures.size()), textures.data());

        for (int i = 0; i < 2; ++i) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbos[i]);
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[i], 0);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                throw std::runtime_error("Blur framebuffer is incomplete");
            }
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

struct OverlayVertex {
    float x;
    float y;
    float r;
    float g;
    float b;
    float a;
};

struct OverlayRenderer {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint program = 0;
    std::vector<OverlayVertex> vertices;

    void initialize();
    void shutdown();
    void clear();
    void addRect(float x0, float y0, float x1, float y1, const std::array<float, 4>& color);
    void addText(float x, float y, float scale, const std::string& text, const std::array<float, 4>& color);
    void render(int viewportWidth, int viewportHeight);
};

GLuint compileShader(GLenum type, const char* source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_TRUE) {
        return shader;
    }

    GLint logLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
    std::string log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
    glGetShaderInfoLog(shader, logLength, nullptr, log.data());

    glDeleteShader(shader);
    throw std::runtime_error("Shader compilation failed:\n" + log);
}

GLuint createProgram(const char* vertexSource, const char* fragmentSource) {
    const GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    const GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
    const GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint success = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_TRUE) {
        return program;
    }

    GLint logLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    std::string log(static_cast<std::size_t>(std::max(logLength, 1)), '\0');
    glGetProgramInfoLog(program, logLength, nullptr, log.data());

    glDeleteProgram(program);
    throw std::runtime_error("Program link failed:\n" + log);
}

std::string uppercaseCopy(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    for (const char ch : text) {
        result.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
    }
    return result;
}

const std::array<unsigned char, 7>& glyphBitmap(char ch) {
    static constexpr std::array<unsigned char, 7> blank{
        0b00000,
        0b00000,
        0b00000,
        0b00000,
        0b00000,
        0b00000,
        0b00000,
    };
    static constexpr std::array<unsigned char, 7> dash{
        0b00000,
        0b00000,
        0b00000,
        0b01110,
        0b00000,
        0b00000,
        0b00000,
    };
    static constexpr std::array<unsigned char, 7> colon{
        0b00000,
        0b00100,
        0b00100,
        0b00000,
        0b00100,
        0b00100,
        0b00000,
    };

    switch (ch) {
        case '0': {
            static constexpr std::array<unsigned char, 7> rows{
                0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
            };
            return rows;
        }
        case '1': {
            static constexpr std::array<unsigned char, 7> rows{
                0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
            };
            return rows;
        }
        case '2': {
            static constexpr std::array<unsigned char, 7> rows{
                0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
            };
            return rows;
        }
        case '3': {
            static constexpr std::array<unsigned char, 7> rows{
                0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
            };
            return rows;
        }
        case '4': {
            static constexpr std::array<unsigned char, 7> rows{
                0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
            };
            return rows;
        }
        case '5': {
            static constexpr std::array<unsigned char, 7> rows{
                0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
            };
            return rows;
        }
        case '6': {
            static constexpr std::array<unsigned char, 7> rows{
                0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
            };
            return rows;
        }
        case '7': {
            static constexpr std::array<unsigned char, 7> rows{
                0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
            };
            return rows;
        }
        case '8': {
            static constexpr std::array<unsigned char, 7> rows{
                0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
            };
            return rows;
        }
        case '9': {
            static constexpr std::array<unsigned char, 7> rows{
                0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b11100,
            };
            return rows;
        }
        case 'A': {
            static constexpr std::array<unsigned char, 7> rows{
                0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
            };
            return rows;
        }
        case 'B': {
            static constexpr std::array<unsigned char, 7> rows{
                0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
            };
            return rows;
        }
        case 'C': {
            static constexpr std::array<unsigned char, 7> rows{
                0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
            };
            return rows;
        }
        case 'D': {
            static constexpr std::array<unsigned char, 7> rows{
                0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100,
            };
            return rows;
        }
        case 'E': {
            static constexpr std::array<unsigned char, 7> rows{
                0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
            };
            return rows;
        }
        case 'F': {
            static constexpr std::array<unsigned char, 7> rows{
                0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
            };
            return rows;
        }
        case 'G': {
            static constexpr std::array<unsigned char, 7> rows{
                0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
            };
            return rows;
        }
        case 'H': {
            static constexpr std::array<unsigned char, 7> rows{
                0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
            };
            return rows;
        }
        case 'I': {
            static constexpr std::array<unsigned char, 7> rows{
                0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
            };
            return rows;
        }
        case 'J': {
            static constexpr std::array<unsigned char, 7> rows{
                0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110,
            };
            return rows;
        }
        case 'K': {
            static constexpr std::array<unsigned char, 7> rows{
                0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
            };
            return rows;
        }
        case 'L': {
            static constexpr std::array<unsigned char, 7> rows{
                0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
            };
            return rows;
        }
        case 'M': {
            static constexpr std::array<unsigned char, 7> rows{
                0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001,
            };
            return rows;
        }
        case 'N': {
            static constexpr std::array<unsigned char, 7> rows{
                0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
            };
            return rows;
        }
        case 'O': {
            static constexpr std::array<unsigned char, 7> rows{
                0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
            };
            return rows;
        }
        case 'P': {
            static constexpr std::array<unsigned char, 7> rows{
                0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
            };
            return rows;
        }
        case 'Q': {
            static constexpr std::array<unsigned char, 7> rows{
                0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
            };
            return rows;
        }
        case 'R': {
            static constexpr std::array<unsigned char, 7> rows{
                0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
            };
            return rows;
        }
        case 'S': {
            static constexpr std::array<unsigned char, 7> rows{
                0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
            };
            return rows;
        }
        case 'T': {
            static constexpr std::array<unsigned char, 7> rows{
                0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
            };
            return rows;
        }
        case 'U': {
            static constexpr std::array<unsigned char, 7> rows{
                0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
            };
            return rows;
        }
        case 'V': {
            static constexpr std::array<unsigned char, 7> rows{
                0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
            };
            return rows;
        }
        case 'W': {
            static constexpr std::array<unsigned char, 7> rows{
                0b10001, 0b10001, 0b10001, 0b10001, 0b10101, 0b11011, 0b10001,
            };
            return rows;
        }
        case 'X': {
            static constexpr std::array<unsigned char, 7> rows{
                0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
            };
            return rows;
        }
        case 'Y': {
            static constexpr std::array<unsigned char, 7> rows{
                0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
            };
            return rows;
        }
        case 'Z': {
            static constexpr std::array<unsigned char, 7> rows{
                0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
            };
            return rows;
        }
        case '-':
            return dash;
        case ':':
            return colon;
        case ' ':
            return blank;
        default:
            return blank;
    }
}

void OverlayRenderer::initialize() {
    program = createProgram(kOverlayVertexShader, kOverlayFragmentShader);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), reinterpret_cast<void*>(sizeof(float) * 2));
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void OverlayRenderer::shutdown() {
    if (vbo != 0) {
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }
    if (vao != 0) {
        glDeleteVertexArrays(1, &vao);
        vao = 0;
    }
    if (program != 0) {
        glDeleteProgram(program);
        program = 0;
    }
    vertices.clear();
}

void OverlayRenderer::clear() {
    vertices.clear();
}

void OverlayRenderer::addRect(float x0, float y0, float x1, float y1, const std::array<float, 4>& color) {
    const auto pushVertex = [&](float x, float y) {
        vertices.push_back({x, y, color[0], color[1], color[2], color[3]});
    };

    pushVertex(x0, y0);
    pushVertex(x1, y0);
    pushVertex(x1, y1);
    pushVertex(x0, y0);
    pushVertex(x1, y1);
    pushVertex(x0, y1);
}

void OverlayRenderer::addText(float x, float y, float scale, const std::string& text, const std::array<float, 4>& color) {
    const std::string upper = uppercaseCopy(text);
    float cursorX = x;
    const float cell = scale;
    const float pixel = scale * 0.82f;
    const float offset = 0.5f * (cell - pixel);

    for (const char ch : upper) {
        if (ch == ' ') {
            cursorX += cell * 6.0f;
            continue;
        }

        const auto& rows = glyphBitmap(ch);
        for (int row = 0; row < 7; ++row) {
            for (int col = 0; col < 5; ++col) {
                const unsigned char bit = static_cast<unsigned char>(1u << (4 - col));
                if ((rows[row] & bit) == 0u) {
                    continue;
                }

                const float px = cursorX + static_cast<float>(col) * cell + offset;
                const float py = y + static_cast<float>(row) * cell + offset;
                addRect(px, py, px + pixel, py + pixel, color);
            }
        }

        cursorX += cell * 6.0f;
    }
}

void OverlayRenderer::render(int viewportWidth, int viewportHeight) {
    if (program == 0 || vertices.empty()) {
        return;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(program);
    glUniform2f(glGetUniformLocation(program, "uViewportPx"), static_cast<float>(viewportWidth), static_cast<float>(viewportHeight));

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(OverlayVertex)),
        vertices.data(),
        GL_DYNAMIC_DRAW
    );
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDisable(GL_BLEND);
}

std::string injectDefinesAfterVersion(const char* source, const std::string& defines) {
    const std::string shader(source);
    const std::size_t versionPos = shader.find("#version");
    if (versionPos != std::string::npos) {
        const std::size_t lineEnd = shader.find('\n', versionPos);
        if (lineEnd != std::string::npos) {
            return shader.substr(0, lineEnd + 1) + defines + shader.substr(lineEnd + 1);
        }
    }
    return defines + shader;
}

void applyFieldPreset(const NamedFieldPreset& preset, RunConfig& config) {
    config.fieldMode = preset.fieldMode;
    config.torusP = preset.torusP;
    config.torusQ = preset.torusQ;
}

bool applyNamedFieldPreset(std::string_view name, RunConfig& config) {
    if (name == kPresetHopfion.cliName) {
        applyFieldPreset(kPresetHopfion, config);
        return true;
    }
    if (name == kPresetTrefoil.cliName) {
        applyFieldPreset(kPresetTrefoil, config);
        return true;
    }
    if (name == kPresetCinquefoil.cliName) {
        applyFieldPreset(kPresetCinquefoil, config);
        return true;
    }
    if (name == kPresetLinkedRings.cliName || name == "rings") {
        applyFieldPreset(kPresetLinkedRings, config);
        return true;
    }
    return false;
}

RunConfig parseRunConfig(int argc, char** argv) {
    RunConfig config{};

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);

        if (arg == "--quality") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --quality");
            }

            const std::string_view value(argv[++i]);
            if (value == "low") {
                config.quality = &kQualityLow;
            } else if (value == "medium") {
                config.quality = &kQualityMedium;
            } else if (value == "high") {
                config.quality = &kQualityHigh;
            } else if (value == "ultra") {
                config.quality = &kQualityUltra;
            } else {
                throw std::runtime_error("Unknown quality preset. Use: low, medium, high, ultra");
            }
            continue;
        }

        if (arg == "--preset") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --preset");
            }

            const std::string_view value(argv[++i]);
            if (!applyNamedFieldPreset(value, config)) {
                throw std::runtime_error("Unknown field preset. Use: hopfion, trefoil, cinquefoil, linked-rings");
            }
            continue;
        }

        if (arg == "--field") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --field");
            }

            const std::string_view value(argv[++i]);
            if (value == "hopfion") {
                config.fieldMode = FieldMode::Hopfion;
            } else if (value == "torus") {
                config.fieldMode = FieldMode::Torus;
            } else {
                throw std::runtime_error("Unknown field mode. Use: hopfion or torus");
            }
            continue;
        }

        if (arg == "--p") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --p");
            }
            config.torusP = std::stoi(argv[++i]);
            continue;
        }

        if (arg == "--q") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --q");
            }
            config.torusQ = std::stoi(argv[++i]);
            continue;
        }

        throw std::runtime_error("Unknown argument: " + std::string(arg));
    }

    if (config.torusP <= 0 || config.torusQ <= 0) {
        throw std::runtime_error("--p and --q must be positive integers");
    }

    if (config.torusP > 9 || config.torusQ > 9) {
        throw std::runtime_error("--p and --q are currently limited to 9 for performance and numerical stability");
    }

    if (config.fieldMode == FieldMode::Hopfion && (config.torusP != 2 || config.torusQ != 3)) {
        // Allow the user to specify p/q early, but if they did, assume they want the torus family.
        config.fieldMode = FieldMode::Torus;
    }

    return config;
}

const char* fieldModeName(FieldMode fieldMode) {
    switch (fieldMode) {
        case FieldMode::Hopfion:
            return "hopfion";
        case FieldMode::Torus:
            return "torus";
        default:
            return "unknown";
    }
}

std::string namedFieldLabel(const RunConfig& config) {
    if (config.fieldMode == FieldMode::Hopfion) {
        return kPresetHopfion.displayName;
    }
    if (config.torusP == 2 && config.torusQ == 3) {
        return kPresetTrefoil.displayName;
    }
    if (config.torusP == 2 && config.torusQ == 5) {
        return kPresetCinquefoil.displayName;
    }
    if (config.torusP == 2 && config.torusQ == 2) {
        return kPresetLinkedRings.displayName;
    }
    return "CUSTOM";
}

std::string fieldDescriptor(const RunConfig& config) {
    if (config.fieldMode == FieldMode::Hopfion) {
        return "HOPFION";
    }
    return "TORUS P" + std::to_string(config.torusP) + " Q" + std::to_string(config.torusQ);
}

const char* filterModeName(FilterMode filterMode) {
    switch (filterMode) {
        case FilterMode::Off:
            return "OFF";
        case FilterMode::Filmic:
            return "FILMIC";
        case FilterMode::Aurora:
            return "AURORA";
        case FilterMode::Solar:
            return "SOLAR";
        case FilterMode::NoGlow:
            return "NO GLOW";
        default:
            return "UNKNOWN";
    }
}

FilterMode nextFilterMode(FilterMode filterMode) {
    switch (filterMode) {
        case FilterMode::Filmic:
            return FilterMode::NoGlow;
        case FilterMode::NoGlow:
            return FilterMode::Aurora;
        case FilterMode::Aurora:
            return FilterMode::Solar;
        case FilterMode::Solar:
            return FilterMode::Off;
        case FilterMode::Off:
        default:
            return FilterMode::Filmic;
    }
}

void resetSceneState(AppState& state, float currentTime) {
    state.camera.reset();
    state.runtime.simulationStartTime = currentTime;
}

void applyRuntimePreset(AppState& state, Simulation& simulation, const NamedFieldPreset& preset, float currentTime) {
    applyFieldPreset(preset, state.runtime.config);
    simulation.setFieldConfig(state.runtime.config.fieldMode, state.runtime.config.torusP, state.runtime.config.torusQ);
    resetSceneState(state, currentTime);
}

bool consumeKeyPress(GLFWwindow* window, int key, bool& held) {
    const bool down = glfwGetKey(window, key) == GLFW_PRESS;
    const bool pressed = down && !held;
    held = down;
    return pressed;
}

std::string buildWindowTitle(const RunConfig& config, const QualityPreset& quality, FilterMode filterMode, float fps) {
    std::string title = "EM Loop | ";
    title += uppercaseCopy(quality.name);
    title += " | ";
    title += namedFieldLabel(config);
    title += " | ";
    title += fieldDescriptor(config);
    title += " | ";
    title += filterModeName(filterMode);
    title += " | ";
    title += std::to_string(static_cast<int>(fps + 0.5f));
    title += " FPS";
    return title;
}

void setFieldUniforms(GLuint program, FieldMode fieldMode, int torusP, int torusQ) {
    const GLint modeLocation = glGetUniformLocation(program, "uFieldMode");
    if (modeLocation >= 0) {
        glUniform1i(modeLocation, static_cast<int>(fieldMode));
    }

    const GLint pLocation = glGetUniformLocation(program, "uTorusP");
    if (pLocation >= 0) {
        glUniform1i(pLocation, torusP);
    }

    const GLint qLocation = glGetUniformLocation(program, "uTorusQ");
    if (qLocation >= 0) {
        glUniform1i(qLocation, torusQ);
    }
}

const QualityPreset& qualityPreset(const RunConfig& config) {
    return config.quality != nullptr ? *config.quality : kQualityMedium;
}

void printConfiguration(const RunConfig& config) {
    std::cout << "Preset: " << namedFieldLabel(config) << '\n';
    std::cout << "Field: " << fieldModeName(config.fieldMode);
    if (config.fieldMode == FieldMode::Torus) {
        std::cout << " (p=" << config.torusP << ", q=" << config.torusQ << ')';
    }
    std::cout << '\n';
}

void printControls(const QualityPreset& quality) {
    std::cout
        << "Quality: " << quality.name << '\n';
}

void buildOverlayHud(OverlayRenderer& overlay, const RunConfig& config, const QualityPreset& quality, FilterMode filterMode, float fps) {
    overlay.clear();

    const std::array<float, 4> panelColor{0.015f, 0.025f, 0.040f, 0.78f};
    const std::array<float, 4> borderColor{0.18f, 0.46f, 0.70f, 0.55f};
    const std::array<float, 4> textColor{0.92f, 0.97f, 1.00f, 0.92f};
    const std::array<float, 4> accentColor{0.22f, 0.88f, 0.92f, 0.92f};
    const std::array<float, 4> warmColor{1.00f, 0.72f, 0.26f, 0.92f};

    overlay.addRect(18.0f, 18.0f, 430.0f, 152.0f, panelColor);
    overlay.addRect(18.0f, 18.0f, 430.0f, 20.0f, borderColor);
    overlay.addRect(18.0f, 150.0f, 430.0f, 152.0f, borderColor);

    overlay.addText(32.0f, 32.0f, 3.0f, "EM LOOP", accentColor);
    overlay.addText(32.0f, 54.0f, 2.0f, "QUALITY: " + uppercaseCopy(quality.name), textColor);
    overlay.addText(32.0f, 74.0f, 2.0f, "PRESET: " + namedFieldLabel(config), warmColor);
    overlay.addText(32.0f, 94.0f, 2.0f, "FIELD: " + fieldDescriptor(config), textColor);
    overlay.addText(32.0f, 114.0f, 2.0f, "FILTER: " + std::string(filterModeName(filterMode)), textColor);
    overlay.addText(32.0f, 134.0f, 2.0f, "FPS: " + std::to_string(static_cast<int>(fps + 0.5f)), accentColor);
}

}  // namespace

void setMouseCapture(GLFWwindow* window, AppState& state, bool captured);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow* window, AppState& state, Simulation& simulation, float currentTime, float dt);
void setCameraBasisUniforms(GLuint program, const Camera& camera, float aspect, float timeSeconds);
GLuint createParticleVao(GLuint vertexBuffer);
GLuint createFullscreenVao();

int main(int argc, char** argv) {
    GLFWwindow* window = nullptr;
    GLuint fullscreenVao = 0;
    GLuint particleVao = 0;
    GLuint backgroundProgram = 0;
    GLuint volumeProgram = 0;
    GLuint particleProgram = 0;
    GLuint blurProgram = 0;
    GLuint compositeProgram = 0;

    SceneFramebuffer sceneFramebuffer;
    PingPongBuffers pingPongBuffers;
    OverlayRenderer overlayRenderer;
    Simulation simulation;

    try {
        RunConfig runConfig = parseRunConfig(argc, argv);
        const QualityPreset& quality = qualityPreset(runConfig);

        if (glfwInit() != GLFW_TRUE) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        window = glfwCreateWindow(quality.windowWidth, quality.windowHeight, "EM Loop", nullptr, nullptr);
        if (window == nullptr) {
            throw std::runtime_error("Failed to create GLFW window");
        }
        
        AppState state;
        state.runtime.config = runConfig;
        state.framebufferWidth = quality.windowWidth;
        state.framebufferHeight = quality.windowHeight;
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        glfwSetWindowUserPointer(window, &state);
        glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
        glfwSetCursorPosCallback(window, cursorPositionCallback);
        setMouseCapture(window, state, true);
        
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            throw std::runtime_error("Failed to initialize GLEW");
        }
        
        glGetError();
        
        glEnable(GL_PROGRAM_POINT_SIZE);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        
        backgroundProgram = createProgram(kFullscreenVertexShader, kBackgroundFragmentShader);
        const std::string volumeShader = injectDefinesAfterVersion(
            kVolumeFragmentShader,
            "#define EM_LOOP_VOLUME_STEPS " + std::to_string(quality.volumeSteps) + "\n"
        );
        
        volumeProgram = createProgram(kFullscreenVertexShader, volumeShader.c_str());
        particleProgram = createProgram(kParticleVertexShader, kParticleFragmentShader);
        blurProgram = createProgram(kFullscreenVertexShader, kBlurFragmentShader);
        compositeProgram = createProgram(kFullscreenVertexShader, kCompositeFragmentShader);
        overlayRenderer.initialize();
        
        fullscreenVao = createFullscreenVao();
        
        if (!simulation.initialize(
                quality.linesPerFamily,
                quality.pointsPerLine,
                state.runtime.config.fieldMode,
                state.runtime.config.torusP,
                state.runtime.config.torusQ
            )) {
            throw std::runtime_error("Failed to initialize CUDA particle simulation");
        }
        
        particleVao = createParticleVao(simulation.vertexBuffer());
        
        glfwGetFramebufferSize(window, &state.framebufferWidth, &state.framebufferHeight);
        state.framebufferWidth = std::max(state.framebufferWidth, 1);
        state.framebufferHeight = std::max(state.framebufferHeight, 1);
        sceneFramebuffer.create(state.framebufferWidth, state.framebufferHeight);
        pingPongBuffers.create(
            std::max(1, state.framebufferWidth / quality.bloomDivisor),
            std::max(1, state.framebufferHeight / quality.bloomDivisor)
        );
        
        printControls(quality);
        printConfiguration(state.runtime.config);
        std::cout
            << "Controls:\n"
            << "  WASD  move horizontally\n"
            << "  Q / E move down / up\n"
            << "  Mouse look\n"
            << "  Shift speed up\n"
            << "  Tab   toggle mouse capture\n"
            << "  R     reset scene\n"
            << "  1 2 3 switch presets\n"
            << "  F     cycle filters\n"
            << "  Esc   quit\n";
        
        float lastTime = static_cast<float>(glfwGetTime());
        state.runtime.simulationStartTime = lastTime;
        float fpsSmoothed = 60.0f;
        glfwSetWindowTitle(window, buildWindowTitle(state.runtime.config, quality, state.runtime.filterMode, fpsSmoothed).c_str());

        while (glfwWindowShouldClose(window) == GLFW_FALSE) {
            const float currentTime = static_cast<float>(glfwGetTime());
            float dt = currentTime - lastTime;
            lastTime = currentTime;
            dt = std::clamp(dt, 0.0005f, 0.033f);
            fpsSmoothed = fpsSmoothed * 0.92f + (1.0f / dt) * 0.08f;

            glfwPollEvents();
            processInput(window, state, simulation, currentTime, dt);

            glfwSetWindowTitle(window, buildWindowTitle(state.runtime.config, quality, state.runtime.filterMode, fpsSmoothed).c_str());
        
            if (state.framebufferResized) {
                sceneFramebuffer.create(state.framebufferWidth, state.framebufferHeight);
                pingPongBuffers.create(
                    std::max(1, state.framebufferWidth / quality.bloomDivisor),
                    std::max(1, state.framebufferHeight / quality.bloomDivisor)
                );
                state.framebufferResized = false;
            }
        
            const float aspect = static_cast<float>(state.framebufferWidth) / static_cast<float>(state.framebufferHeight);
            const Mat4 projection = perspective(state.camera.verticalFov, aspect, 0.1f, 60.0f);
            const Mat4 view = lookAt(state.camera.position, state.camera.position + state.camera.forward(), state.camera.up());
            const float simulationTime = std::max(0.0f, currentTime - state.runtime.simulationStartTime);
        
            simulation.update(dt, simulationTime);
        
            glBindFramebuffer(GL_FRAMEBUFFER, sceneFramebuffer.fbo);
            glViewport(0, 0, state.framebufferWidth, state.framebufferHeight);
            glDisable(GL_BLEND);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
            glBindVertexArray(fullscreenVao);
        
            glUseProgram(backgroundProgram);
            setCameraBasisUniforms(backgroundProgram, state.camera, aspect, simulationTime);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        
            glEnable(GL_BLEND);
            glBlendEquation(GL_FUNC_ADD);
            glBlendFunc(GL_ONE, GL_ONE);
        
            glUseProgram(volumeProgram);
            setCameraBasisUniforms(volumeProgram, state.camera, aspect, simulationTime);
            setFieldUniforms(volumeProgram, state.runtime.config.fieldMode, state.runtime.config.torusP, state.runtime.config.torusQ);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        
            glUseProgram(particleProgram);
            glUniformMatrix4fv(glGetUniformLocation(particleProgram, "uView"), 1, GL_FALSE, view.data());
            glUniformMatrix4fv(glGetUniformLocation(particleProgram, "uProjection"), 1, GL_FALSE, projection.data());
            glBindVertexArray(particleVao);
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glLineWidth(1.2f);

            const GLsizei pointsPerLine = static_cast<GLsizei>(quality.pointsPerLine);
            const GLsizei linesPerFamily = static_cast<GLsizei>(quality.linesPerFamily);
            const GLsizei familyStride = pointsPerLine * linesPerFamily;
            const GLenum linePrimitive = state.runtime.config.fieldMode == FieldMode::Hopfion ? GL_LINE_LOOP : GL_LINE_STRIP;

            for (GLsizei family = 0; family < 2; ++family) {
                const GLsizei familyBase = family * familyStride;
                for (GLsizei line = 0; line < linesPerFamily; ++line) {
                    glDrawArrays(linePrimitive, familyBase + line * pointsPerLine, pointsPerLine);
                }
            }

            glDisable(GL_DEPTH_TEST);
            glBindVertexArray(0);
        
            glDisable(GL_BLEND);
        
            glUseProgram(blurProgram);
            glBindVertexArray(fullscreenVao);

            GLuint bloomTexture = sceneFramebuffer.brightTexture;
            if (state.runtime.filterMode != FilterMode::NoGlow) {
                bool horizontal = true;
                bool firstIteration = true;
                for (int i = 0; i < quality.blurPasses; ++i) {
                    const int target = horizontal ? 0 : 1;
                    const GLuint sourceTexture = firstIteration ? sceneFramebuffer.brightTexture : pingPongBuffers.textures[1 - target];

                    glBindFramebuffer(GL_FRAMEBUFFER, pingPongBuffers.fbos[target]);
                    glViewport(
                        0,
                        0,
                        std::max(1, state.framebufferWidth / quality.bloomDivisor),
                        std::max(1, state.framebufferHeight / quality.bloomDivisor)
                    );
                    glClear(GL_COLOR_BUFFER_BIT);
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, sourceTexture);
                    glUniform1i(glGetUniformLocation(blurProgram, "uInputTexture"), 0);
                    glUniform2f(glGetUniformLocation(blurProgram, "uDirection"), horizontal ? 1.0f : 0.0f, horizontal ? 0.0f : 1.0f);
                    glDrawArrays(GL_TRIANGLES, 0, 3);

                    horizontal = !horizontal;
                    firstIteration = false;
                }

                bloomTexture = pingPongBuffers.textures[horizontal ? 1 : 0];
            }
        
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, state.framebufferWidth, state.framebufferHeight);
            glClear(GL_COLOR_BUFFER_BIT);
        
            glUseProgram(compositeProgram);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, sceneFramebuffer.colorTexture);
            glUniform1i(glGetUniformLocation(compositeProgram, "uSceneTexture"), 0);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, bloomTexture);
            glUniform1i(glGetUniformLocation(compositeProgram, "uBloomTexture"), 1);
            glUniform1i(glGetUniformLocation(compositeProgram, "uFilterMode"), static_cast<int>(state.runtime.filterMode));
            glBindVertexArray(fullscreenVao);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            buildOverlayHud(overlayRenderer, state.runtime.config, quality, state.runtime.filterMode, fpsSmoothed);
            overlayRenderer.render(state.framebufferWidth, state.framebufferHeight);

            glfwSwapBuffers(window);
        }
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        
        if (particleVao != 0) {
            glDeleteVertexArrays(1, &particleVao);
        }
        if (fullscreenVao != 0) {
            glDeleteVertexArrays(1, &fullscreenVao);
        }
        if (backgroundProgram != 0) {
            glDeleteProgram(backgroundProgram);
        }
        if (volumeProgram != 0) {
            glDeleteProgram(volumeProgram);
        }
        if (particleProgram != 0) {
            glDeleteProgram(particleProgram);
        }
        if (blurProgram != 0) {
            glDeleteProgram(blurProgram);
        }
        if (compositeProgram != 0) {
            glDeleteProgram(compositeProgram);
        }
        sceneFramebuffer.destroy();
        pingPongBuffers.destroy();
        overlayRenderer.shutdown();
        simulation.shutdown();
        
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
        return EXIT_FAILURE;
    }
        
    if (particleVao != 0) {
        glDeleteVertexArrays(1, &particleVao);
    }
    if (fullscreenVao != 0) {
        glDeleteVertexArrays(1, &fullscreenVao);
    }
    if (backgroundProgram != 0) {
        glDeleteProgram(backgroundProgram);
    }
    if (volumeProgram != 0) {
        glDeleteProgram(volumeProgram);
    }
    if (particleProgram != 0) {
        glDeleteProgram(particleProgram);
    }
    if (blurProgram != 0) {
        glDeleteProgram(blurProgram);
    }
    if (compositeProgram != 0) {
        glDeleteProgram(compositeProgram);
    }

    sceneFramebuffer.destroy();
    pingPongBuffers.destroy();
    overlayRenderer.shutdown();
    simulation.shutdown();
        
    if (window != nullptr) {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
    return EXIT_SUCCESS;
}

void setMouseCapture(GLFWwindow* window, AppState& state, bool captured) {
    state.input.mouseCaptured = captured;
    state.input.firstMouseSample = true;
    glfwSetInputMode(window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    state->framebufferWidth = std::max(width, 1);
    state->framebufferHeight = std::max(height, 1);
    state->framebufferResized = true;
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (!state->input.mouseCaptured) {
        return;
    }

    if (state->input.firstMouseSample) {
        state->input.lastMouseX = xpos;
        state->input.lastMouseY = ypos;
        state->input.firstMouseSample = false;
        return;
    }

    const double deltaX = xpos - state->input.lastMouseX;
    const double deltaY = ypos - state->input.lastMouseY;
    state->input.lastMouseX = xpos;
    state->input.lastMouseY = ypos;

    constexpr float sensitivity = 0.0023f;
    state->camera.yaw += static_cast<float>(deltaX) * sensitivity;
    state->camera.pitch -= static_cast<float>(deltaY) * sensitivity;
    state->camera.pitch = std::clamp(state->camera.pitch, -1.45f, 1.45f);
}

void processInput(GLFWwindow* window, AppState& state, Simulation& simulation, float currentTime, float dt) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    if (consumeKeyPress(window, GLFW_KEY_TAB, state.input.tabHeld)) {
        setMouseCapture(window, state, !state.input.mouseCaptured);
    }

    if (consumeKeyPress(window, GLFW_KEY_R, state.input.resetHeld)) {
        resetSceneState(state, currentTime);
    }

    if (consumeKeyPress(window, GLFW_KEY_1, state.input.preset1Held)) {
        applyRuntimePreset(state, simulation, kPresetHopfion, currentTime);
    }
    if (consumeKeyPress(window, GLFW_KEY_2, state.input.preset2Held)) {
        applyRuntimePreset(state, simulation, kPresetTrefoil, currentTime);
    }
    if (consumeKeyPress(window, GLFW_KEY_3, state.input.preset3Held)) {
        applyRuntimePreset(state, simulation, kPresetCinquefoil, currentTime);
    }
    if (consumeKeyPress(window, GLFW_KEY_F, state.input.filterHeld)) {
        state.runtime.filterMode = nextFilterMode(state.runtime.filterMode);
    }

    const Vec3 worldUp{0.0f, 1.0f, 0.0f};
    Vec3 forward = state.camera.forward();
    Vec3 flatForward{forward.x, 0.0f, forward.z};
    if (lengthSquared(flatForward) <= 1.0e-6f) {
        flatForward = {std::cos(state.camera.yaw), 0.0f, std::sin(state.camera.yaw)};
    }
    flatForward = normalize(flatForward);
    const Vec3 right = normalize(cross(flatForward, worldUp));

    Vec3 movement{};
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        movement += flatForward;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        movement -= flatForward;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        movement -= right;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        movement += right;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        movement -= worldUp;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        movement += worldUp;
    }

    if (lengthSquared(movement) > 1.0e-6f) {
        movement = normalize(movement);
        const float speed = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 5.5f : 2.8f) * dt;
        state.camera.position += movement * speed;
    }
}

void setCameraBasisUniforms(GLuint program, const Camera& camera, float aspect, float timeSeconds) {
    glUniform3f(glGetUniformLocation(program, "uCameraPos"), camera.position.x, camera.position.y, camera.position.z);

    const Vec3 forward = camera.forward();
    const Vec3 right = camera.right();
    const Vec3 up = camera.up();

    glUniform3f(glGetUniformLocation(program, "uCameraForward"), forward.x, forward.y, forward.z);
    glUniform3f(glGetUniformLocation(program, "uCameraRight"), right.x, right.y, right.z);
    glUniform3f(glGetUniformLocation(program, "uCameraUp"), up.x, up.y, up.z);
    glUniform1f(glGetUniformLocation(program, "uTanHalfFov"), std::tan(camera.verticalFov * 0.5f));
    glUniform1f(glGetUniformLocation(program, "uAspect"), aspect);
    glUniform1f(glGetUniformLocation(program, "uTime"), timeSeconds);
}

GLuint createParticleVao(GLuint vertexBuffer) {
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 8, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 8, reinterpret_cast<void*>(sizeof(float) * 4));
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vao;
}

GLuint createFullscreenVao() {
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    return vao;
}

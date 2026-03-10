#pragma once

constexpr const char* kFullscreenVertexShader = R"glsl(
#version 330 core

out vec2 vUv;

void main() {
    vec2 position = vec2(
        (gl_VertexID == 1) ? 3.0 : -1.0,
        (gl_VertexID == 2) ? 3.0 : -1.0
    );

    vUv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
)glsl";

constexpr const char* kBackgroundFragmentShader = R"glsl(
#version 330 core

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 brightColor;

in vec2 vUv;

uniform vec3 uCameraForward;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform float uTanHalfFov;
uniform float uAspect;
uniform float uTime;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 rayDirection(vec2 uv) {
    vec2 ndc = uv * 2.0 - 1.0;
    return normalize(
        uCameraForward +
        ndc.x * uAspect * uTanHalfFov * uCameraRight +
        ndc.y * uTanHalfFov * uCameraUp
    );
}

void main() {
    vec3 dir = rayDirection(vUv);
    float horizon = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);

    vec3 skyLow = vec3(0.005, 0.010, 0.020);
    vec3 skyHigh = vec3(0.055, 0.095, 0.140);
    vec3 color = mix(skyLow, skyHigh, pow(horizon, 0.85));

    float aurora = 0.5 + 0.5 * sin(dir.x * 6.0 + dir.z * 4.0 + uTime * 0.08);
    color += vec3(0.010, 0.018, 0.012) * pow(1.0 - abs(dir.y), 2.4) * aurora;

    vec2 sphereUv = vec2(atan(dir.z, dir.x), asin(clamp(dir.y, -1.0, 1.0)));
    vec2 starGrid = vec2(320.0, 170.0);
    vec2 cell = floor((sphereUv + vec2(3.14159265, 1.57079633)) * starGrid);
    float starSeed = hash12(cell);
    vec2 local = fract((sphereUv + vec2(3.14159265, 1.57079633)) * starGrid) - 0.5;
    float sparkle = smoothstep(0.9965, 1.0, starSeed);
    float star = sparkle * smoothstep(0.36, 0.0, length(local));
    color += star * mix(vec3(0.7, 0.8, 1.0), vec3(1.0, 0.93, 0.80), hash12(cell + 7.0));

    fragColor = vec4(color, 1.0);
    brightColor = vec4(0.0);
}
)glsl";

constexpr const char* kVolumeFragmentShader = R"glsl(
#version 330 core

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 brightColor;

in vec2 vUv;

uniform vec3 uCameraPos;
uniform vec3 uCameraForward;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform float uTanHalfFov;
uniform float uAspect;
uniform float uTime;
uniform int uFieldMode;
uniform int uTorusP;
uniform int uTorusQ;

vec2 complexMul(vec2 a, vec2 b) {
    return vec2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

vec2 complexInv(vec2 z) {
    float denom = dot(z, z) + 1e-12;
    return vec2(z.x, -z.y) / denom;
}

vec2 complexDiv(vec2 a, vec2 b) {
    return complexMul(a, complexInv(b));
}

vec2 complexPowInt(vec2 value, int exponent) {
    vec2 result = vec2(1.0, 0.0);
    for (int i = 0; i < 9; ++i) {
        if (i >= exponent) {
            break;
        }
        result = complexMul(result, value);
    }
    return result;
}

vec3 rayDirection(vec2 uv) {
    vec2 ndc = uv * 2.0 - 1.0;
    return normalize(
        uCameraForward +
        ndc.x * uAspect * uTanHalfFov * uCameraRight +
        ndc.y * uTanHalfFov * uCameraUp
    );
}

bool intersectSphere(vec3 ro, vec3 rd, float radius, out float tMin, out float tMax) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) {
        return false;
    }

    h = sqrt(h);
    tMin = -b - h;
    tMax = -b + h;
    return tMax > 0.0;
}

void sampleHopfionField(vec3 p, out vec3 electric, out vec3 magnetic, out vec3 poynting, out float energy) {
    // Exact Hopfion field from Kedia et al. (2013), evaluated in shader for the halo.
    const float spatialScale = 1.75;
    const float timeScale = 0.42;

    vec3 q = p / spatialScale;
    float t = uTime * timeScale;
    float r2 = dot(q, q);

    vec2 a = vec2(q.x, -q.y);
    vec2 b = vec2(t - q.z, -1.0);
    vec2 d = vec2(r2 - t * t + 1.0, 2.0 * t);

    vec2 a2 = complexMul(a, a);
    vec2 b2 = complexMul(b, b);
    vec2 d3 = complexMul(complexMul(d, d), d);
    vec2 invD3 = complexInv(d3);

    vec2 fx = complexMul(b2 - a2, invD3);
    vec2 sum = a2 + b2;
    vec2 fy = complexMul(vec2(sum.y, -sum.x), invD3);
    vec2 fz = complexMul(2.0 * complexMul(a, b), invD3);

    float amplitudeScale = 1.0 / (spatialScale * spatialScale);
    electric = vec3(fx.x, fy.x, fz.x) * amplitudeScale;
    magnetic = vec3(fx.y, fy.y, fz.y) * amplitudeScale;
    poynting = normalize(cross(electric, magnetic) + vec3(1e-7));
    energy = 0.5 * (dot(electric, electric) + dot(magnetic, magnetic));
}

void sampleTorusField(vec3 p, out vec3 electric, out vec3 magnetic, out vec3 poynting, out float energy) {
    const float spatialScale = 1.75;
    const float timeScale = 0.42;

    vec3 q = p / spatialScale;
    float t = uTime * timeScale;
    float r2 = dot(q, q);

    vec2 d = vec2(r2 - t * t + 1.0, 2.0 * t);
    vec2 d2 = complexMul(d, d);
    vec2 numeratorAlpha = vec2(r2 - t * t - 1.0, 2.0 * q.z);
    vec2 numeratorBeta = vec2(2.0 * q.x, -2.0 * q.y);

    vec2 alpha = complexDiv(numeratorAlpha, d);
    vec2 beta = complexDiv(numeratorBeta, d);

    vec2 gradDx = vec2(2.0 * q.x, 0.0);
    vec2 gradDy = vec2(2.0 * q.y, 0.0);
    vec2 gradDz = vec2(2.0 * q.z, 0.0);

    vec2 gradAlphaX = complexDiv(complexMul(gradDx, d) - complexMul(numeratorAlpha, gradDx), d2);
    vec2 gradAlphaY = complexDiv(complexMul(gradDy, d) - complexMul(numeratorAlpha, gradDy), d2);
    vec2 gradAlphaZ = complexDiv(complexMul(vec2(2.0 * q.z, 2.0), d) - complexMul(numeratorAlpha, gradDz), d2);

    vec2 gradBetaX = complexDiv(complexMul(vec2(2.0, 0.0), d) - complexMul(numeratorBeta, gradDx), d2);
    vec2 gradBetaY = complexDiv(complexMul(vec2(0.0, -2.0), d) - complexMul(numeratorBeta, gradDy), d2);
    vec2 gradBetaZ = complexDiv(-complexMul(numeratorBeta, gradDz), d2);

    vec2 alphaPower = complexPowInt(alpha, max(uTorusP - 1, 0));
    vec2 betaPower = complexPowInt(beta, max(uTorusQ - 1, 0));
    vec2 prefactor = float(uTorusP * uTorusQ) * complexMul(alphaPower, betaPower);

    vec2 fieldX = complexMul(prefactor, complexMul(gradAlphaY, gradBetaZ) - complexMul(gradAlphaZ, gradBetaY));
    vec2 fieldY = complexMul(prefactor, complexMul(gradAlphaZ, gradBetaX) - complexMul(gradAlphaX, gradBetaZ));
    vec2 fieldZ = complexMul(prefactor, complexMul(gradAlphaX, gradBetaY) - complexMul(gradAlphaY, gradBetaX));

    float amplitudeScale = 1.0 / (spatialScale * spatialScale);
    electric = vec3(fieldX.x, fieldY.x, fieldZ.x) * amplitudeScale;
    magnetic = vec3(fieldX.y, fieldY.y, fieldZ.y) * amplitudeScale;
    poynting = normalize(cross(electric, magnetic) + vec3(1e-7));
    energy = 0.5 * (dot(electric, electric) + dot(magnetic, magnetic));
}

float sampleDensity(vec3 p, out vec3 color) {
    vec3 electric;
    vec3 magnetic;
    vec3 poynting;
    float energy;

    if (uFieldMode == 1) {
        sampleTorusField(p, electric, magnetic, poynting, energy);
    } else {
        sampleHopfionField(p, electric, magnetic, poynting, energy);
    }

    float energyGlow = smoothstep(0.001, 0.070, energy);
    float density = energyGlow * (0.35 + 1.45 * sqrt(max(energy, 0.0)));

    float phase = 0.5 + 0.5 * sin(atan(electric.y, electric.x) + atan(magnetic.z, magnetic.x));
    vec3 electricColor = vec3(0.08, 0.62, 0.97);
    vec3 magneticColor = vec3(1.00, 0.53, 0.12);
    color = mix(magneticColor, electricColor, phase);
    color = mix(color, vec3(0.92, 0.96, 1.00), 0.22 + 0.18 * abs(poynting.y));
    color *= 0.35 + 0.95 * energyGlow;

    return density;
}

void main() {
    vec3 rayOrigin = uCameraPos;
    vec3 rayDir = rayDirection(vUv);

    float tMin;
    float tMax;
    if (!intersectSphere(rayOrigin, rayDir, 6.0, tMin, tMax)) {
        fragColor = vec4(0.0);
        brightColor = vec4(0.0);
        return;
    }

    tMin = max(tMin, 0.0);
    const int steps = EM_LOOP_VOLUME_STEPS;
    float distanceSpan = tMax - tMin;
    float stepSize = distanceSpan / float(steps);

    vec3 accum = vec3(0.0);
    float transmittance = 1.0;

    for (int i = 0; i < steps; ++i) {
        float t = tMin + (float(i) + 0.5) * stepSize;
        vec3 samplePos = rayOrigin + rayDir * t;
        vec3 sampleColor;
        float density = sampleDensity(samplePos, sampleColor);

        float extinction = density * 0.11;
        float emission = density * 0.26;
        accum += transmittance * sampleColor * emission;
        transmittance *= exp(-extinction);

        if (transmittance < 0.015) {
            break;
        }
    }

    vec3 color = accum * 1.35;
    fragColor = vec4(color, 1.0);
    float highlight = max(max(color.r, color.g), color.b);
    brightColor = vec4(color * smoothstep(0.40, 0.95, highlight), 1.0);
}
)glsl";

constexpr const char* kParticleVertexShader = R"glsl(
#version 330 core

layout(location = 0) in vec4 aPositionSize;
layout(location = 1) in vec4 aColor;

uniform mat4 uView;
uniform mat4 uProjection;

out vec4 vColor;

void main() {
    vec4 viewPos = uView * vec4(aPositionSize.xyz, 1.0);
    gl_Position = uProjection * viewPos;

    float viewDepth = -viewPos.z;
    float fade = smoothstep(0.35, 1.05, viewDepth);
    float energyHint = clamp((aPositionSize.w - 2.0) / 4.5, 0.0, 1.0);
    vColor = vec4(aColor.rgb * (0.72 + 0.28 * energyHint) * fade, aColor.a * fade);
}
)glsl";

constexpr const char* kParticleFragmentShader = R"glsl(
#version 330 core

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 brightColor;

in vec4 vColor;

void main() {
    vec3 color = vColor.rgb * vColor.a;
    fragColor = vec4(color, vColor.a);

    float luma = max(max(color.r, color.g), color.b);
    brightColor = vec4(color * smoothstep(0.78, 1.22, luma) * 0.22, 1.0);
}
)glsl";

constexpr const char* kBlurFragmentShader = R"glsl(
#version 330 core

out vec4 fragColor;

in vec2 vUv;

uniform sampler2D uInputTexture;
uniform vec2 uDirection;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(uInputTexture, 0));

    vec3 result = texture(uInputTexture, vUv).rgb * 0.227027;
    result += texture(uInputTexture, vUv + uDirection * texel * 1.384615).rgb * 0.316216;
    result += texture(uInputTexture, vUv - uDirection * texel * 1.384615).rgb * 0.316216;
    result += texture(uInputTexture, vUv + uDirection * texel * 3.230769).rgb * 0.070270;
    result += texture(uInputTexture, vUv - uDirection * texel * 3.230769).rgb * 0.070270;

    fragColor = vec4(result, 1.0);
}
)glsl";

constexpr const char* kCompositeFragmentShader = R"glsl(
#version 330 core

out vec4 fragColor;

in vec2 vUv;

uniform sampler2D uSceneTexture;
uniform sampler2D uBloomTexture;
uniform int uFilterMode;

vec3 acesApprox(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

void main() {
    vec3 scene = texture(uSceneTexture, vUv).rgb;
    vec3 bloom = vec3(0.0);
    if (uFilterMode != 4) {
        bloom = texture(uBloomTexture, vUv).rgb;
    }
    vec3 hdr = scene + bloom * 0.85;
    vec2 centered = vUv * 2.0 - 1.0;
    float radial = dot(centered, centered);
    vec3 mapped;

    if (uFilterMode == 0) {
        mapped = clamp(hdr * 0.72, 0.0, 1.0);
    } else if (uFilterMode == 4) {
        mapped = acesApprox(scene * 1.02);
        mapped *= mix(1.0, smoothstep(1.85, 0.12, radial), 0.10);
    } else if (uFilterMode == 2) {
        mapped = acesApprox(hdr * 1.04);
        mapped *= vec3(0.90, 1.01, 1.13);
        mapped = mix(mapped, vec3(dot(mapped, vec3(0.22, 0.61, 0.17))), 0.06);
        mapped *= mix(1.0, smoothstep(1.85, 0.08, radial), 0.18);
    } else if (uFilterMode == 3) {
        mapped = acesApprox(hdr * 1.16);
        mapped = pow(mapped, vec3(0.94, 0.96, 1.00));
        mapped *= vec3(1.10, 1.02, 0.92);
        mapped *= mix(1.0, smoothstep(1.68, 0.08, radial), 0.20);
    } else {
        mapped = acesApprox(hdr * 1.10);
        mapped *= mix(1.0, smoothstep(1.75, 0.10, radial), 0.25);
    }

    mapped = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));

    fragColor = vec4(mapped, 1.0);
}
)glsl";

constexpr const char* kOverlayVertexShader = R"glsl(
#version 330 core

layout(location = 0) in vec2 aPositionPx;
layout(location = 1) in vec4 aColor;

uniform vec2 uViewportPx;

out vec4 vColor;

void main() {
    vec2 ndc = vec2(
        aPositionPx.x / uViewportPx.x * 2.0 - 1.0,
        1.0 - aPositionPx.y / uViewportPx.y * 2.0
    );

    gl_Position = vec4(ndc, 0.0, 1.0);
    vColor = aColor;
}
)glsl";

constexpr const char* kOverlayFragmentShader = R"glsl(
#version 330 core

in vec4 vColor;
out vec4 fragColor;

void main() {
    fragColor = vColor;
}
)glsl";

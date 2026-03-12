from Range import *
import math
from collections import OrderedDict
import bgl


commonShader = """
//#define PERFORMANCE_MODE

// ==========================
// Generic Helpers/Constants
// ==========================

#define PI 3.141592653589793
#define TWOPI 6.283185307179586
#define HALFPI 1.570796326794896
#define INV_SQRT_2 0.7071067811865476

#define BOX_MIN vec3(-1.0)
#define BOX_MAX vec3(1.0)

#define MAX_ALPHA_PER_UNIT_DIST 10.0
#define QUIT_ALPHA 0.99
#define QUIT_ALPHA_L 0.95

float len2Inf(vec2 v)
{
    vec2 d = abs(v);
    return max(d.x, d.y);
}

void boxClip(
    in vec3 boxMin,
    in vec3 boxMax,
    in vec3 p,
    in vec3 v,
    out vec2 tRange,
    out float didHit
)
{
    vec3 tb0 = (boxMin - p) / v;
    vec3 tb1 = (boxMax - p) / v;
    vec3 tmin = min(tb0, tb1);
    vec3 tmax = max(tb0, tb1);

    tRange = vec2(
        max(max(tmin.x, tmin.y), tmin.z),
        min(min(tmax.x, tmax.y), tmax.z)
    );

    didHit = step(tRange.x, tRange.y);
}

float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float hash13(vec3 p3)
{
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 hash31(float p)
{
    vec3 p3 = fract(vec3(p) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

vec3 colormap(float t)
{
    return 0.5 + 0.5 * cos(TWOPI * (t + vec3(0.0, 0.1, 0.2)));
}

vec4 blendOnto(vec4 cFront, vec4 cBehind)
{
    return cFront + (1.0 - cFront.a) * cBehind;
}
"""


densityShader = commonShader + """
uniform float time;
uniform float volumeSize;
uniform float atlasGrid;

float noiseValue3(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float n000 = hash13(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash13(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash13(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash13(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash13(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash13(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash13(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash13(i + vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);

    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);

    return mix(nxy0, nxy1, f.z);
}

vec3 decodeLMN(vec2 uv)
{
    float n = volumeSize;
    float g = atlasGrid;

    vec2 atlasSize = vec2(n * g);
    vec2 pixel = floor(uv * atlasSize);
    vec2 tile = floor(pixel / n);
    vec2 local = mod(pixel, n);
    float l = tile.x + tile.y * g;

    return vec3(l, local.x, local.y);
}

float fbm(vec3 p)
{
    float f = 0.0;
    float a = 0.5;

    for (int i = 0; i < 4; i++) {
        f += a * noiseValue3(p);
        p *= 2.03;
        a *= 0.5;
    }

    return f;
}

float fDensity(vec3 lmn, float t)
{
    float n = volumeSize;
    vec3 uvw = (lmn / max(n - 1.0, 1.0)) * 2.0 - 1.0;

    vec3 p = vec3(
        uvw.x * 2.0,
        uvw.y * 1.6 + t * 0.25,
        uvw.z * 2.0
    );

    float warp = fbm(p * 0.85 + vec3(0.0, t * 0.2, 0.0));
    float d = fbm(p + vec3(2.0 * warp, 0.5 * warp, -1.5 * warp));

    float edge = smoothstep(1.15, 0.35, len2Inf(uvw.xz));
    d *= edge;

    return smoothstep(0.45, 0.85, d);
}

void main()
{
    vec2 uv = gl_TexCoord[0].st;
    vec3 lmn = decodeLMN(uv);

    if (lmn.x >= volumeSize) {
        gl_FragColor = vec4(0.0);
        return;
    }

    float density = fDensity(lmn, time);

    gl_FragColor = vec4(density, 0.0, 0.0, 1.0);
}
"""


lightShader = commonShader + """
uniform sampler2D densityTex;
uniform vec3 lightDir;
uniform float volumeSize;
uniform float atlasGrid;
uniform float lightStep;
uniform float maxLightSteps;

vec2 atlasUVFromLMN(vec3 lmn)
{
    float n = volumeSize;
    float g = atlasGrid;

    vec3 q = clamp(lmn, vec3(0.0), vec3(n - 1.0));

    float l = floor(q.x + 0.5);
    vec2 mn = floor(q.yz + 0.5);

    float tx = mod(l, g);
    float ty = floor(l / g);

    vec2 pixel = vec2(tx, ty) * n + mn + 0.5;
    vec2 atlasSize = vec2(n * g);
    return pixel / atlasSize;
}

vec3 decodeLMN(vec2 uv)
{
    float n = volumeSize;
    float g = atlasGrid;

    vec2 atlasSize = vec2(n * g);
    vec2 pixel = floor(uv * atlasSize);
    vec2 tile = floor(pixel / n);
    vec2 local = mod(pixel, n);
    float l = tile.x + tile.y * g;

    return vec3(l, local.x, local.y);
}

vec3 worldFromLMN(vec3 lmn)
{
    float n = max(volumeSize - 1.0, 1.0);
    vec3 uvw = lmn / n;
    return mix(BOX_MIN, BOX_MAX, uvw);
}

vec3 lmnFromWorld(vec3 p)
{
    vec3 uvw = (p - BOX_MIN) / (BOX_MAX - BOX_MIN);
    return clamp(uvw, vec3(0.0), vec3(1.0)) * max(volumeSize - 1.0, 1.0);
}

float sampleDensityLMN(vec3 lmn)
{
    return texture(densityTex, atlasUVFromLMN(lmn)).r;
}

float marchToLight(vec3 p, vec3 nv)
{
    vec2 tRange;
    float didHit;
    boxClip(BOX_MIN, BOX_MAX, p, nv, tRange, didHit);
    if (didHit < 0.5) {
        return 0.0;
    }

    tRange.x = max(tRange.x, 0.0);

    float lightAmount = 1.0;
    const int MAX_STEPS = 150;

    float t = tRange.x + min(tRange.y - tRange.x, lightStep) * hash13(100.0 * p);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (float(i) >= maxLightSteps) {
            break;
        }

        if (t > tRange.y || lightAmount < (1.0 - QUIT_ALPHA_L)) {
            break;
        }

        vec3 pL = p + nv * t;
        vec3 lmn = lmnFromWorld(pL);
        float density = sampleDensityLMN(lmn);
        float calpha = clamp(density * MAX_ALPHA_PER_UNIT_DIST * lightStep, 0.0, 1.0);

        lightAmount *= 1.0 - calpha;

        t += lightStep;
    }

    return lightAmount;
}

void main()
{
    vec2 uv = gl_TexCoord[0].st;
    vec3 lmn = decodeLMN(uv);

    if (lmn.x >= volumeSize) {
        gl_FragColor = vec4(0.0);
        return;
    }

    float density = sampleDensityLMN(lmn);
    vec3 p = worldFromLMN(lmn);
    vec3 toLight = normalize(-lightDir);
    float trans = marchToLight(p, toLight);

    gl_FragColor = vec4(density, trans, 0.0, 1.0);
}
"""


finalShader = commonShader + """
uniform sampler2D bgl_RenderedTexture;
uniform sampler2D bgl_DepthTexture;
uniform sampler2D volumeTex;
uniform mat4 boxMatrixInv;
uniform float volumeSize;
uniform float atlasGrid;
uniform float rayStep;
uniform float intensity;
uniform vec2 screenSize;
uniform float debugNoColor;
uniform float debugNoDensity;
uniform float debugNoLight;
uniform float debugShowSteps;
uniform float vignetteIntensity;

vec3 getViewPos(vec2 coord)
{
    float depth = texture(bgl_DepthTexture, coord).x;
    vec3 ndc = vec3(coord, depth) * 2.0 - 1.0;
    vec4 view = inverse(gl_ProjectionMatrix) * vec4(ndc, 1.0);
    return view.xyz / max(view.w, 1e-6);
}

vec2 atlasUVFromLMN(vec3 lmn)
{
    float n = volumeSize;
    float g = atlasGrid;

    vec3 q = clamp(lmn, vec3(0.0), vec3(n - 1.0));

    float l = floor(q.x + 0.5);
    vec2 mn = floor(q.yz + 0.5);

    float tx = mod(l, g);
    float ty = floor(l / g);

    vec2 pixel = vec2(tx, ty) * n + mn + 0.5;
    vec2 atlasSize = vec2(n * g);
    return pixel / atlasSize;
}

vec3 lmnFromLocal(vec3 pLocal)
{
    vec3 uvw = clamp((pLocal + vec3(1.0)) * 0.5, vec3(0.0), vec3(1.0));
    return uvw * max(volumeSize - 1.0, 1.0);
}

bool intersectVolume(vec3 ro, vec3 rd, out float t0, out float t1)
{
    vec3 roLocal = (boxMatrixInv * vec4(ro, 1.0)).xyz;
    vec3 rdLocal = (boxMatrixInv * vec4(rd, 0.0)).xyz;

    vec2 tRange;
    float didHit;
    boxClip(BOX_MIN, BOX_MAX, roLocal, rdLocal, tRange, didHit);
    t0 = tRange.x;
    t1 = tRange.y;
    return didHit > 0.5 && t1 > max(t0, 0.0);
}

vec2 sampleData(vec3 pWorld)
{
    vec3 pLocal = (boxMatrixInv * vec4(pWorld, 1.0)).xyz;
    vec3 lmn = lmnFromLocal(pLocal);
    return texture(volumeTex, atlasUVFromLMN(lmn)).rg;
}

#define DATA(lmn) texture(volumeTex, atlasUVFromLMN(lmn)).rg

vec2 getDataInterp(vec3 lmn)
{
    vec3 flmn = floor(lmn);

    vec2 d000 = DATA(flmn);
    vec2 d001 = DATA(flmn + vec3(0.0, 0.0, 1.0));
    vec2 d010 = DATA(flmn + vec3(0.0, 1.0, 0.0));
    vec2 d011 = DATA(flmn + vec3(0.0, 1.0, 1.0));
    vec2 d100 = DATA(flmn + vec3(1.0, 0.0, 0.0));
    vec2 d101 = DATA(flmn + vec3(1.0, 0.0, 1.0));
    vec2 d110 = DATA(flmn + vec3(1.0, 1.0, 0.0));
    vec2 d111 = DATA(flmn + vec3(1.0, 1.0, 1.0));

    vec3 t = lmn - flmn;
    return mix(
        mix(mix(d000, d100, t.x), mix(d010, d110, t.x), t.y),
        mix(mix(d001, d101, t.x), mix(d011, d111, t.x), t.y),
        t.z
    );
}

void readLMN(in vec3 lmn, out float density, out float lightAmount)
{
    vec2 data = getDataInterp(lmn);

    lightAmount = (debugNoLight > 0.5) ? 1.0 : data.g;
    lightAmount = mix(lightAmount, 1.0, 0.025);

    density = data.r;
    if (debugNoDensity > 0.5) {
        density = 0.1;
    }
}

vec4 march(vec3 p, vec3 nv, vec2 fragCoord)
{
    vec2 tRange;
    float didHitBox;
    boxClip(BOX_MIN, BOX_MAX, p, nv, tRange, didHitBox);
    tRange.x = max(0.0, tRange.x);

    vec4 color = vec4(0.0);
    if (didHitBox < 0.5) {
        return color;
    }

    float stepSize = max(rayStep, 0.005);
    float t = tRange.x + min(tRange.y - tRange.x, stepSize) * hash12(fragCoord);

    int i = 0;
    for (; i < 150; i++) {
        if (t > tRange.y || color.a > QUIT_ALPHA) {
            break;
        }

        vec3 rayPos = p + t * nv;
        vec3 lmn = lmnFromLocal(rayPos);

        float density;
        float lightAmount;
        readLMN(lmn, density, lightAmount);

        vec3 cfrag = (debugNoColor > 0.5) ? vec3(1.0) : colormap(0.7 * density + 0.8);

        float calpha = density * MAX_ALPHA_PER_UNIT_DIST * stepSize;
        vec4 ci = clamp(vec4(cfrag * lightAmount, 1.0) * calpha, 0.0, 1.0);
        color = blendOnto(color, ci);

        t += stepSize;
    }

    float finalA = clamp(color.a / QUIT_ALPHA, 0.0, 1.0);
    color *= finalA / (color.a + 1e-5);

    if (debugShowSteps > 0.5) {
        return vec4(vec3(float(i) / 150.0), 1.0);
    }

    return color;
}

void main()
{
    vec2 uv = gl_TexCoord[0].st;
    vec3 scene = texture(bgl_RenderedTexture, uv).rgb;

    mat4 invView = inverse(gl_ModelViewMatrix);
    vec3 camPos = invView[3].xyz;
    vec3 view = getViewPos(uv);
    float maxDist = length(view);

    if (maxDist < 1e-6) {
        gl_FragColor = vec4(scene, 1.0);
        return;
    }

    vec3 rayDir = normalize(mat3(invView) * view);

    float tEnter;
    float tExit;
    if (!intersectVolume(camPos, rayDir, tEnter, tExit)) {
        gl_FragColor = vec4(scene, 1.0);
        return;
    }

    tEnter = max(tEnter, 0.0);
    tExit = min(tExit, maxDist);

    if (tEnter >= tExit) {
        gl_FragColor = vec4(scene, 1.0);
        return;
    }

    vec3 roLocal = (boxMatrixInv * vec4(camPos, 1.0)).xyz;
    vec3 rdLocal = (boxMatrixInv * vec4(rayDir, 0.0)).xyz;

    vec2 fragCoord = uv * screenSize;
    vec4 fgColor = march(roLocal, rdLocal, fragCoord);

    vec3 color = blendOnto(fgColor * intensity, vec4(scene, 1.0)).rgb;

    vec2 radv = vec2(0.5, 0.5) - uv;
    float dCorner = length(radv) / INV_SQRT_2;
    float vignetteFactor = 1.0 - mix(0.0, vignetteIntensity, smoothstep(0.4, 0.9, dCorner));
    color *= vignetteFactor;

    gl_FragColor = vec4(color, 1.0);
}
"""


class VoxelVolumetric(types.KX_PythonComponent):

    args = OrderedDict([
        ("Layer", 0),
        ("Sun", "Sun"),
        ("Volume Size", 32),
        ("Ray Step", 0.04),
        ("Light Step", 0.06),
        ("Light Steps", 64),
        ("Intensity", 1.2),
        ("No Color", False),
        ("No Density", False),
        ("No Light", False),
        ("Show Steps", False),
        ("Vignette", 0.25),
    ])


    def start(self, args):

        scene = logic.getCurrentScene()
        self.scene = scene

        layer = int(args["Layer"])
        self.sun_name = str(args.get("Sun", "Sun"))
        self.volume_size = max(8, min(64, int(args.get("Volume Size", 32))))
        self.atlas_grid = int(math.ceil(math.sqrt(self.volume_size)))
        self.atlas_size = self.volume_size * self.atlas_grid
        self.ray_step = max(0.005, float(args.get("Ray Step", 0.04)))
        self.light_step = max(0.005, float(args.get("Light Step", 0.06)))
        self.light_steps = max(8.0, min(150.0, float(args.get("Light Steps", 64))))
        self.intensity = max(0.0, float(args.get("Intensity", 1.2)))
        self.debug_no_color = 1.0 if bool(args.get("No Color", False)) else 0.0
        self.debug_no_density = 1.0 if bool(args.get("No Density", False)) else 0.0
        self.debug_no_light = 1.0 if bool(args.get("No Light", False)) else 0.0
        self.debug_show_steps = 1.0 if bool(args.get("Show Steps", False)) else 0.0
        self.vignette = max(0.0, min(1.0, float(args.get("Vignette", 0.25))))

        getFilter = scene.filterManager.addFilter
        custom = logic.RAS_2DFILTER_CUSTOMFILTER

        # PASS 1 Density
        self.densityFilter = getFilter(layer, custom, densityShader)

        self.densityFilter.addOffScreen(
            1,
            width=self.atlas_size,
            height=self.atlas_size,
            hdr=1,
            mipmap=False
        )

        # PASS 2 Light
        self.lightFilter = getFilter(layer+1, custom, lightShader)

        self.lightFilter.addOffScreen(
            1,
            width=self.atlas_size,
            height=self.atlas_size,
            hdr=1,
            mipmap=False
        )

        # PASS 3 Final
        self.finalFilter = getFilter(layer+2, custom, finalShader)

        # Bind density texture to light pass
        densityTex = self.densityFilter.offScreen.colorBindCodes[0]
        self.lightFilter.setTexture(0, densityTex, "densityTex")

        # Bind volume texture to final pass
        lightTex = self.lightFilter.offScreen.colorBindCodes[0]
        self.finalFilter.setTexture(0, lightTex, "volumeTex")

        self.densityFilter.setUniform1f("volumeSize", float(self.volume_size))
        self.densityFilter.setUniform1f("atlasGrid", float(self.atlas_grid))

        self.lightFilter.setUniform1f("volumeSize", float(self.volume_size))
        self.lightFilter.setUniform1f("atlasGrid", float(self.atlas_grid))
        self.lightFilter.setUniform1f("lightStep", self.light_step)
        self.lightFilter.setUniform1f("maxLightSteps", self.light_steps)

        self.finalFilter.setUniform1f("volumeSize", float(self.volume_size))
        self.finalFilter.setUniform1f("atlasGrid", float(self.atlas_grid))
        self.finalFilter.setUniform1f("rayStep", self.ray_step)
        self.finalFilter.setUniform1f("intensity", self.intensity)
        self.finalFilter.setUniform2f("screenSize", float(render.getWindowWidth()), float(render.getWindowHeight()))
        self.finalFilter.setUniform1f("debugNoColor", self.debug_no_color)
        self.finalFilter.setUniform1f("debugNoDensity", self.debug_no_density)
        self.finalFilter.setUniform1f("debugNoLight", self.debug_no_light)
        self.finalFilter.setUniform1f("debugShowSteps", self.debug_show_steps)
        self.finalFilter.setUniform1f("vignetteIntensity", self.vignette)

        self._warned_sun_missing = False

        print("[VoxelVolumetric] Initialized")
        print(f"[VoxelVolumetric] Volume={self.volume_size}^3 Atlas={self.atlas_size}x{self.atlas_size}")


    def update(self):

        if not hasattr(self, "densityFilter") or not hasattr(self, "finalFilter"):
            return

        time = logic.getRealTime()

        self.densityFilter.setUniform1f("time", time)
        try:
            self.finalFilter.setUniformMatrix4("boxMatrixInv", self.object.worldTransform.inverted())
            self.finalFilter.setUniform2f("screenSize", float(render.getWindowWidth()), float(render.getWindowHeight()))
        except TypeError:
            return

        sun = self.scene.objects.get(self.sun_name)

        if sun:
            dir = sun.worldOrientation.col[2].normalized()

            self._warned_sun_missing = False

            self.lightFilter.setUniform3f(
                "lightDir",
                dir.x,
                dir.y,
                dir.z
            )
        elif not self._warned_sun_missing:
            print(f"[VoxelVolumetric] AVISO: luz '{self.sun_name}' nao encontrada")
            self._warned_sun_missing = True
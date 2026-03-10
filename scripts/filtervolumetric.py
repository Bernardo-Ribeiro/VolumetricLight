from Range import *
from collections import OrderedDict


occlusionShader = """
uniform sampler2D bgl_DepthTexture;

uniform mat4 boxMatrix[boxMax];

uniform vec3 windowPos;
uniform vec3 windowNormal;
uniform vec3 windowRight;
uniform vec3 windowUp;
uniform vec2 windowSize;

uniform vec3 lightDir;
uniform float lightIntensity;
uniform float scattering;
uniform float falloffScale;

vec3 getViewPos(vec2 coord) {
    float depth = texture(bgl_DepthTexture, coord).x;
    vec3 ndc = vec3(coord, depth) * 2.0 - 1.0;
    vec4 view = inverse(gl_ProjectionMatrix) * vec4(ndc, 1.0);
    return view.xyz / max(view.w, 1e-6);
}

vec3 getScale(mat4 model) {
    float sx = length(model[0].xyz);
    float sy = length(model[1].xyz);
    float sz = length(model[2].xyz);
    return vec3(sx, sy, sz);
}

vec2 intersectCube(vec3 ori, vec3 dir, vec3 size) {
    vec3 tMin = (-size - ori) / dir;
    vec3 tMax = ( size - ori) / dir;

    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);

    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(max(0.0, tNear), tFar);
}

vec2 boxVolume(vec3 ori, vec3 dir, mat4 matrix) {
    vec3 size = getScale(matrix);
    vec2 box = intersectCube(ori, dir, size);
    if (box.x > box.y) {
        box.x = 1e9;
    }
    return box;
}

void main() {
    vec2 uv = gl_TexCoord[0].st;

    mat4 invView = inverse(gl_ModelViewMatrix);
    vec3 cameraPos = invView[3].xyz;

    vec3 view = getViewPos(uv);
    float maxDist = length(view);
    if (maxDist < 1e-6) {
        gl_FragColor = vec4(0.0);
        return;
    }

    vec3 rayDir = normalize(mat3(invView) * view);

    float denom = dot(lightDir, windowNormal);
    if (abs(denom) < 1e-4) {
        gl_FragColor = vec4(0.0);
        return;
    }

    float A = dot(windowPos - cameraPos, windowNormal);
    float B = dot(rayDir, windowNormal);

    float lR = dot(lightDir, windowRight) / denom;
    float offsetR = dot(cameraPos - windowPos, windowRight) + A * lR;
    float slopeR = dot(rayDir, windowRight) - B * lR;
    float tR0, tR1;
    if (abs(slopeR) > 1e-6) {
        tR0 = (-windowSize.x - offsetR) / slopeR;
        tR1 = ( windowSize.x - offsetR) / slopeR;
        if (tR0 > tR1) { float tmp = tR0; tR0 = tR1; tR1 = tmp; }
    } else if (abs(offsetR) >= windowSize.x) {
        gl_FragColor = vec4(0.0);
        return;
    } else {
        tR0 = -1e9;
        tR1 = 1e9;
    }

    float lU = dot(lightDir, windowUp) / denom;
    float offsetU = dot(cameraPos - windowPos, windowUp) + A * lU;
    float slopeU = dot(rayDir, windowUp) - B * lU;
    float tU0, tU1;
    if (abs(slopeU) > 1e-6) {
        tU0 = (-windowSize.y - offsetU) / slopeU;
        tU1 = ( windowSize.y - offsetU) / slopeU;
        if (tU0 > tU1) { float tmp = tU0; tU0 = tU1; tU1 = tmp; }
    } else if (abs(offsetU) >= windowSize.y) {
        gl_FragColor = vec4(0.0);
        return;
    } else {
        tU0 = -1e9;
        tU1 = 1e9;
    }

    float tS0, tS1;
    if (abs(B) > 1e-6) {
        float tS = A / B;
        if (denom * B > 0.0) { tS0 = -1e9; tS1 = tS; }
        else                 { tS0 = tS;  tS1 = 1e9; }
    } else if ((denom > 0.0 && A < 0.0) || (denom < 0.0 && A > 0.0)) {
        gl_FragColor = vec4(0.0);
        return;
    } else {
        tS0 = -1e9;
        tS1 = 1e9;
    }

    float d0 = dot(windowPos - cameraPos, lightDir);
    float dr = -dot(rayDir, lightDir);
    float k = falloffScale;

    float accumLight = 0.0;
    for (int i = 0; i < boxMax; i++) {
        vec3 pos = cameraPos - boxMatrix[i][3].xyz;
        vec2 boxDist = boxVolume(pos, rayDir, boxMatrix[i]);

        if (boxDist.x >= maxDist) {
            continue;
        }

        float tNear = boxDist.x;
        float tFar = min(boxDist.y, maxDist);
        float tA = max(max(max(tNear, tR0), tU0), tS0);
        float tB = min(min(min(tFar, tR1), tU1), tS1);

        if (tA >= tB) {
            continue;
        }

        float boxLight;
        if (abs(dr) < 1e-6 || k < 1e-9) {
            float dist = max(0.0, d0 + dr * (tA + tB) * 0.5);
            float falloff = exp(-k * dist);
            boxLight = scattering * falloff * (tB - tA);
        } else {
            float eA = exp(-k * max(0.0, d0 + dr * tA));
            float eB = exp(-k * max(0.0, d0 + dr * tB));
            boxLight = scattering * abs((eA - eB) / (k * dr));
        }

        accumLight = max(accumLight, boxLight);
    }

    float lit = clamp(accumLight * lightIntensity, 0.0, 1.0);
    gl_FragColor = vec4(lit, lit, lit, lit);
}
"""


finalShader = """
uniform sampler2D bgl_RenderedTexture;
uniform sampler2D bgl_RenderedOcclusion;

uniform vec3 lightColor;
uniform float blendStrength;

void main() {
    vec2 uv = gl_TexCoord[0].st;
    vec3 image = texture(bgl_RenderedTexture, uv).rgb;
    float occlusion = texture(bgl_RenderedOcclusion, uv).r;

    vec3 volumetric = image + lightColor * (occlusion * blendStrength);
    gl_FragColor = vec4(volumetric, 1.0);
}
"""


class VolumetricFilter(types.KX_PythonComponent):

    args = OrderedDict([
        ("Sun Object", "Sun"),
        ("Window Object", "WindowPortal"),
        ("Light Color", (1.0, 0.88, 0.7)),
        ("Light Intensity", 1.0),
        ("Scattering", 0.5),
        ("Falloff Scale", 0.05),
        ("Blend Strength", 0.8),
        ("Resolution Scale", 0.5),
    ])

    def start(self, args):
        self.scene = logic.getCurrentScene()
        self.cam = self.scene.active_camera

        self._sun_name = str(args.get("Sun Object", "Sun"))
        self._window_name = str(args.get("Window Object", "WindowPortal"))

        lc = args.get("Light Color", (1.0, 0.88, 0.7))
        self._light_color = lc if isinstance(lc, (list, tuple)) else (1.0, 0.88, 0.7)
        self._light_intensity = float(args.get("Light Intensity", 1.0))
        self._scattering = float(args.get("Scattering", 0.5))
        self._falloff_scale = float(args.get("Falloff Scale", 0.05))
        self._blend_strength = float(args.get("Blend Strength", 0.8))
        self._resolution_scale = max(0.1, min(1.0, float(args.get("Resolution Scale", 0.5))))
        self.boxList = [obj for obj in self.object.scene.objects if "box" in obj.name.lower()]

        getFilter = self.scene.filterManager.addFilter
        custom = logic.RAS_2DFILTER_CUSTOMFILTER

        box_count = max(1, len(self.boxList))
        const = f"const int boxMax = {box_count};"
        self._occlusion_filter = getFilter(0, custom, const + occlusionShader)
        self._final_filter = getFilter(1, custom, finalShader)

        width = int(render.getWindowWidth() * self._resolution_scale)
        height = int(render.getWindowHeight() * self._resolution_scale)
        self._occlusion_filter.addOffScreen(
            1,
            width=max(1, width),
            height=max(1, height),
            hdr=0,
            mipmap=False,
        )

        bind_code = self._occlusion_filter.offScreen.colorBindCodes[0]
        self._final_filter.setTexture(0, bind_code, "bgl_RenderedOcclusion")

        self._debug_counter = 0
        print(f"[VolumetricLight] boxes detectadas: {len(self.boxList)}")
        print("[VolumetricLight] Screen filter pronto.")

    def update(self):
        sun = self.scene.objects.get(self._sun_name)
        window = self.scene.objects.get(self._window_name)

        self._debug_counter += 1
        if self._debug_counter % 60 == 1:
            self._log_debug(sun, window)

        if not sun or not window:
            return

        self._update_occlusion_uniforms(sun, window)
        self._update_final_uniforms()

    def _log_debug(self, sun, window):
        if not sun:
            print(f"[VolumetricLight] ERRO: '{self._sun_name}' nao encontrado.")
        if not window:
            print(f"[VolumetricLight] ERRO: '{self._window_name}' nao encontrado.")
        if sun and window:
            print("[VolumetricLight] lightDir  :", list(sun.worldOrientation.col[2]))
            print("[VolumetricLight] windowPos :", list(window.worldPosition))
            print("[VolumetricLight] cameraPos :", list(self.cam.worldPosition))

    def _update_occlusion_uniforms(self, sun, window):
        shader = self._occlusion_filter

        for i, box in enumerate(self.boxList):
            shader.setUniformMatrix4(f"boxMatrix[{i}]", box.worldTransform)

        sun_dir = sun.worldOrientation.col[2]
        shader.setUniform3f("lightDir", sun_dir.x, sun_dir.y, sun_dir.z)

        orient = window.worldOrientation
        best_axis = max([0, 1, 2], key=lambda i: abs(orient.col[i].dot(sun_dir)))
        normal = orient.col[best_axis]
        if normal.dot(sun_dir) < 0.0:
            normal = -normal

        other = [i for i in (0, 1, 2) if i != best_axis]
        win_right = orient.col[other[0]]
        win_up = orient.col[other[1]]
        scale = window.worldScale

        shader.setUniform3f("windowNormal", normal.x, normal.y, normal.z)
        shader.setUniform3f("windowRight", win_right.x, win_right.y, win_right.z)
        shader.setUniform3f("windowUp", win_up.x, win_up.y, win_up.z)
        shader.setUniform2f("windowSize", scale[other[0]], scale[other[1]])
        shader.setUniform3f("windowPos", *window.worldPosition)

        shader.setUniform1f("lightIntensity", self._light_intensity)
        shader.setUniform1f("scattering", self._scattering)
        shader.setUniform1f("falloffScale", self._falloff_scale)

    def _update_final_uniforms(self):
        shader = self._final_filter
        lc = self._light_color
        shader.setUniform3f("lightColor", float(lc[0]), float(lc[1]), float(lc[2]))
        shader.setUniform1f("blendStrength", self._blend_strength)
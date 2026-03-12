from Range import *
from collections import OrderedDict
import bgl
from mathutils import Matrix


occlusionShader = """
uniform sampler2D bgl_DepthTexture;

uniform mat4 boxMatrix[boxMax];
uniform sampler2D shadowMap;
uniform mat4 shadowMatrix;
uniform float shadowBias;
uniform int shadowEnabled;

uniform vec3 windowPos;
uniform vec3 windowNormal;
uniform vec3 windowRight;
uniform vec3 windowUp;
uniform vec2 windowSize;

uniform vec3 lightDir;

const float LIGHT_INTENSITY = 1.0;
const float SCATTERING = 0.4;
const float FALLOFF_SCALE = 0.5;

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

float sampleShadow(vec3 worldPos)
{
    if (shadowEnabled == 0)
        return 1.0;

    vec4 coord = shadowMatrix * vec4(worldPos, 1.0);
    coord.xyz /= coord.w;

    if(coord.x < 0.0 || coord.x > 1.0 ||
       coord.y < 0.0 || coord.y > 1.0)
        return 1.0;

    float texel = 1.0 / 2048.0; // ajuste para resolução do shadow map

    float shadow = 0.0;

    for(int x = -1; x <= 1; x++)
    for(int y = -1; y <= 1; y++)
    {
        vec2 off = vec2(x,y) * texel;
        float d = texture(shadowMap, coord.xy + off).r;

        if(coord.z - shadowBias <= d)
            shadow += 1.0;
    }

    return shadow / 9.0;
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

    vec3 newPos = windowPos + lightDir * 1.5;

    float A = dot(windowPos - cameraPos, windowNormal);
    float B = dot(rayDir, windowNormal);
    float C = dot(newPos - cameraPos, windowNormal);

    float lR = dot(lightDir, windowRight) / denom;
    float offsetR = dot(cameraPos - newPos, windowRight) + C * lR;
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
    float offsetU = dot(cameraPos - newPos, windowUp) + C * lU;
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

    float d0 = dot(newPos - cameraPos, lightDir);
    float dr = -dot(rayDir, lightDir);
    float k = FALLOFF_SCALE;

    float accumLight = 0.0;
    for (int i = 0; i < boxMax; i++) {
        mat4 invBox = inverse(boxMatrix[i]);

        vec3 roLocal = (invBox * vec4(cameraPos,1.0)).xyz;
        vec3 rdLocal = (invBox * vec4(rayDir,0.0)).xyz;

        vec2 boxDist = intersectCube(roLocal, rdLocal, vec3(1.0));

        if (boxDist.x >= maxDist) {
            continue;
        }

        float tNear = boxDist.x;
        float tFar  = boxDist.y;
        float tA = max(max(max(tNear, tR0), tU0), tS0);
        float tB = min(min(min(tFar, tR1), tU1), tS1);
        tB = min(tB, maxDist);

        if (tA >= tB) {
            continue;
        }

        vec3 P = cameraPos + rayDir * ((tA + tB) * 0.5);
        P += lightDir * shadowBias * 10.0;
        float shadowFactor = sampleShadow(P);

        float boxLight;
        
        if (abs(dr) < 1e-6 || k < 1e-9) {
            float dist = max(0.0, d0 + dr * (tA + tB) * 0.5);
            float falloff = exp(-k * dist);
            boxLight = SCATTERING * falloff * (tB - tA);
        } else {
            float a = max(0.0, d0 + dr * tA);
            float b = max(0.0, d0 + dr * tB);

            float eA = exp(-k * a);
            float eB = exp(-k * b);
            boxLight = SCATTERING * abs((eA - eB) / (k * dr));
        }

        boxLight *= shadowFactor;

        accumLight += boxLight;
    }

    float lit = clamp(accumLight * LIGHT_INTENSITY, 0.0, 1.0);
    gl_FragColor = vec4(lit, lit, lit, lit);
}
"""


finalShader = """
uniform sampler2D bgl_RenderedTexture;
uniform sampler2D bgl_RenderedOcclusion;

const vec3 LIGHT_COLOR = vec3(1.0, 0.88, 0.7);
const float BLEND_STRENGTH = 0.8;

void main() {
    vec2 uv = gl_TexCoord[0].st;
    vec3 image = texture(bgl_RenderedTexture, uv).rgb;
    float occlusion = texture(bgl_RenderedOcclusion, uv).r;

    vec3 volumetric = image + LIGHT_COLOR * (occlusion * BLEND_STRENGTH);
    gl_FragColor = vec4(volumetric, 1.0);
}
"""


class VolumetricFilter(types.KX_PythonComponent):

    args = OrderedDict([
        ("layer", 2),
        ("Sun Object", "Sun"),
        ("Window Object", "WindowPortal"),
        ("Resolution Scale", 0.5),
    ])

    def start(self, args):
        self.scene = logic.getCurrentScene()
        self.cam = self.scene.active_camera

        self.layer = int(args["layer"])

        self._sun_name = str(args.get("Sun Object", "Sun"))

        self._window_name = str(args.get("Window Object", "WindowPortal"))
        self._resolution_scale = max(0.1, min(1.0, float(args.get("Resolution Scale", 0.5))))

        self.boxList = [obj for obj in self.object.scene.objects if "box" in obj.name.lower()]

        if len(self.boxList) == 0:
            print("[VolumetricLight] AVISO: nenhum objeto 'box' encontrado na cena.")
            print("[VolumetricLight] O shader rodara sem oclusão volumetrica.")

        getFilter = self.scene.filterManager.addFilter
        custom = logic.RAS_2DFILTER_CUSTOMFILTER

        box_count = max(1, len(self.boxList))
        const = f"const int boxMax = {box_count};"

        # PASS 1 — OCCLUSION
        self._occlusion_filter = getFilter(self.layer, custom, const + occlusionShader)

        # PASS 2 — FINAL
        self._final_filter = getFilter(self.layer + 1, custom, finalShader)

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
        self._warned_shadow_unavailable = False
        print("[VolumetricLight] Screen filter pronto.")

    def update(self):
        sun = self.scene.objects.get(self._sun_name)
        window = self.scene.objects.get(self._window_name)

        #self._debug_counter += 1
        #if self._debug_counter % 60 == 1:
        #    self._log_debug(sun, window)

        if not sun or not window:
            return

        self._update_occlusion_uniforms(sun, window)

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
        sun_dir = sun.worldOrientation.col[2].normalized()
        shader.setUniform3f("lightDir", sun_dir.x, sun_dir.y, sun_dir.z)

        shader.setUniformMatrix4("shadowMatrix", self._shadow_matrix(sun))
        shader.setUniform1f("shadowBias", 0.0001)

        shadow_bind_id = self._shadow_bind_id(sun)
        bgl.glActiveTexture(bgl.GL_TEXTURE3)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, shadow_bind_id)

        shader.setUniform1i("shadowMap", 3)
        shader.setUniform1i("shadowEnabled", 1 if shadow_bind_id > 0 else 0)

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

    def _shadow_matrix(self, light):
        # Keep compatibility with lights that do not expose full shadow settings.
        try:
            size = float(light.shadowFrustumSize)
            near = float(light.shadowClipStart)
            far = float(light.shadowClipEnd)

            bias = Matrix()
            bias[0][0] = 0.5
            bias[0][3] = 0.5
            bias[1][1] = 0.5
            bias[1][3] = 0.5
            bias[2][2] = 0.5
            bias[2][3] = 0.5

            proj = Matrix.OrthoProjection('XY', 4)
            proj[0][0] = 1.0 / size
            proj[1][1] = 1.0 / size
            proj[2][2] = -2.0 / (far - near)
            proj[2][3] = -((far + near) / (far - near))

            view = light.worldOrientation.to_4x4()
            view[0][3] = light.worldPosition[0]
            view[1][3] = light.worldPosition[1]
            view[2][3] = light.worldPosition[2]
            world_to_lamp = view.inverted()

            # Shadow lookup must stay in world->light space and not depend on camera transforms.
            return bias * proj * world_to_lamp
        except Exception:
            return getattr(light, "shadowMatrix", Matrix.Identity(4))

    def _shadow_bind_id(self, light):
        bind_id = int(getattr(light, "shadowBindId", 0) or 0)
        if bind_id <= 0 and not self._warned_shadow_unavailable:
            print("[VolumetricLight] AVISO: shadow map indisponivel no Sun (ative shadows no light).")
            self._warned_shadow_unavailable = True
        return bind_id


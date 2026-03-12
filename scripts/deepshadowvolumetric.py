from Range import *
from collections import OrderedDict
import bgl
from mathutils import Matrix


occlusionShader = """
uniform sampler2D bgl_DepthTexture;

uniform sampler2D shadowMap;
uniform mat4 shadowMatrix;
uniform int shadowEnabled;

uniform mat4 boxMatrix[boxMax];

uniform vec3 lightDir;
uniform vec2 shadowTexelSize;

const float LIGHT_INTENSITY = 1.0;
const float SCATTERING = 0.50;
const float FALLOFF_SCALE = 1.0;
const float ANISOTROPY = 0.8;
const float EXTINCTION_SCALE = 0.5;
const float SHADOW_SOFTNESS = 0.50;
const float SHADOW_BIAS = 0.0001;
const int MAX_VOLUME_DIST = 100;


vec2 intersectCube(vec3 ori, vec3 dir, vec3 size)
{
    vec3 tMin = (-size - ori) / dir;
    vec3 tMax = ( size - ori) / dir;

    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);

    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(max(0.0, tNear), tFar);
}

bool intersectBox(vec3 ro, vec3 rd, mat4 matrix, out float t0, out float t1)
{
    mat4 invBox = inverse(matrix);
    vec3 roLocal = (invBox * vec4(ro,1.0)).xyz;
    vec3 rdLocal = (invBox * vec4(rd,0.0)).xyz;

    vec2 hit = intersectCube(roLocal, rdLocal, vec3(1.0));
    t0 = hit.x;
    t1 = hit.y;

    return t1 > max(t0,0.0);
}

vec3 getViewPos(vec2 coord)
{
    float depth = texture(bgl_DepthTexture, coord).x;
    vec3 ndc = vec3(coord, depth) * 2.0 - 1.0;
    vec4 view = inverse(gl_ProjectionMatrix) * vec4(ndc,1.0);
    return view.xyz / max(view.w,1e-6);
}

float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float sampleShadow(vec3 worldPos)
{
    if(shadowEnabled == 0)
        return 1.0;

    vec4 coord = shadowMatrix * vec4(worldPos,1.0);
    coord.xyz /= coord.w;

    if(coord.x < 0.0 || coord.x > 1.0 ||
       coord.y < 0.0 || coord.y > 1.0)
        return 1.0;

    vec2 texel = max(shadowTexelSize * max(SHADOW_SOFTNESS, 0.0), vec2(1.0 / 4096.0));

    vec2 o0 = vec2(-0.5, -0.5) * texel;
    vec2 o1 = vec2( 0.5, -0.5) * texel;
    vec2 o2 = vec2(-0.5,  0.5) * texel;
    vec2 o3 = vec2( 0.5,  0.5) * texel;

    float d0 = texture(shadowMap, coord.xy + o0).r;
    float d1 = texture(shadowMap, coord.xy + o1).r;
    float d2 = texture(shadowMap, coord.xy + o2).r;
    float d3 = texture(shadowMap, coord.xy + o3).r;

    float z = coord.z - SHADOW_BIAS;
    float v0 = (z <= d0) ? 1.0 : 0.0;
    float v1 = (z <= d1) ? 1.0 : 0.0;
    float v2 = (z <= d2) ? 1.0 : 0.0;
    float v3 = (z <= d3) ? 1.0 : 0.0;

    return 0.25 * (v0 + v1 + v2 + v3);
}

void main()
{
    vec2 uv = gl_TexCoord[0].st;

    mat4 invView = inverse(gl_ModelViewMatrix);
    vec3 cameraPos = invView[3].xyz;

    vec3 view = getViewPos(uv);
    float maxDist = length(view);
    vec3 rayDir = normalize(mat3(invView) * view);

    if(maxDist < 1e-6)
    {
        gl_FragColor = vec4(0.0);
        return;
    }

    const int MAX_STEPS = 48;
    float depthFactor = clamp(maxDist / 60.0, 0.0, 1.0);
    int steps = int(mix(12.0, float(MAX_STEPS), depthFactor));

    float cosTheta = dot(rayDir, lightDir);
    float isotropic = 0.25;
    float forward = pow(max(cosTheta, 0.0), 6.0);
    float phase = isotropic + forward * 0.6;

    float accumTotal = 0.0;

    for(int b=0;b<boxMax;b++)
    {
        float tEnter;
        float tExit;

        if(!intersectBox(cameraPos, rayDir, boxMatrix[b], tEnter, tExit))
            continue;

        tEnter = max(tEnter, 0.0);
        float marchDist = min(tExit, MAX_VOLUME_DIST);
        float stepSize = marchDist / float(max(steps,1));

        if(stepSize <= 0.0)
            continue;

        float accum = 0.0;
        float transmittance = 1.0;
        float sigmaT = max(EXTINCTION_SCALE, 1e-4);

        for(int i=0;i<MAX_STEPS;i++)
        {
            if(i >= steps)
                break;

            float jitter = hash12(uv * 1024.0 + vec2(float(i), float(i)*1.37));

            float t = tEnter + stepSize * (float(i) + jitter);

            vec3 P = cameraPos + rayDir * t;

            float shadow = sampleShadow(P);

            float lightDist = dot(P - cameraPos, -lightDir);
            lightDist = max(lightDist, 0.0);
            float falloff = exp(-FALLOFF_SCALE * lightDist);
            float ds = stepSize;

            accum += transmittance * SCATTERING * falloff * shadow * phase * ds;

            transmittance *= exp(-sigmaT * ds);
            if(transmittance < 0.02)
                break;
        }

        accumTotal += accum;
    }

    float lit = clamp(accumTotal * LIGHT_INTENSITY,0.0,1.0);

    gl_FragColor = vec4(lit,lit,lit,lit);
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

    vec3 volumetric = LIGHT_COLOR * (occlusion * BLEND_STRENGTH);
    vec3 color = image + volumetric;
    gl_FragColor = vec4(color, 1.0);
}
"""

class DeepShadowVolumetricFilter(types.KX_PythonComponent):

    args = OrderedDict([
        ("layer", 2),
        ("Sun Object", "Sun"),
        ("Resolution Scale", 2),
    ])

    def start(self, args):

        self.scene = logic.getCurrentScene()
        self.cam = self.scene.active_camera

        self.layer = int(args["layer"])

        self._sun_name = str(args.get("Sun Object", "Sun"))
        self._resolution_scale = float(args["Resolution Scale"])

        self.boxList = [obj for obj in self.object.scene.objects if "box" in obj.name.lower()]

        if len(self.boxList) == 0:
            print("[VolumetricLight] AVISO: nenhum objeto 'box' encontrado na cena.")
            print("[VolumetricLight] O shader rodara sem oclusão volumetrica.")


        getFilter = self.scene.filterManager.addFilter
        custom = logic.RAS_2DFILTER_CUSTOMFILTER

        box_count = max(1, len(self.boxList))
        const = f"const int boxMax = {box_count};"

        # PASS 1
        self._occlusion_filter = getFilter(self.layer, custom, const + occlusionShader)

        # PASS 2
        self._final_filter = getFilter(self.layer + 1, custom, finalShader)

        width = int(render.getWindowWidth() / self._resolution_scale)
        height = int(render.getWindowHeight() / self._resolution_scale)

        self._occlusion_filter.addOffScreen(
            1,
            width=max(1, width),
            height=max(1, height),
            hdr=0,
            mipmap=False,
        )

        bind_code = self._occlusion_filter.offScreen.colorBindCodes[0]
        self._final_filter.setTexture(0, bind_code, "bgl_RenderedOcclusion")

        self._warned_shadow_unavailable = False

        print("[DeepShadowVolumetric] Filter iniciado")

    def update(self):

        sun = self.scene.objects.get(self._sun_name)

        if not sun:
            return

        self._update_occlusion_uniforms(sun)

    def _update_occlusion_uniforms(self, sun):

        shader = self._occlusion_filter

        for i, box in enumerate(self.boxList):
            shader.setUniformMatrix4(f"boxMatrix[{i}]", box.worldTransform)

        if len(self.boxList) == 0:
            shader.setUniformMatrix4("boxMatrix[0]", Matrix.Identity(4))

        # direção da luz
        sun_dir = sun.worldOrientation.col[2].normalized()
        shader.setUniform3f("lightDir", sun_dir.x, sun_dir.y, sun_dir.z)

        # shadow map
        shader.setUniformMatrix4("shadowMatrix", self._shadow_matrix(sun))
        texel = self._shadow_texel_size(sun)
        shader.setUniform2f("shadowTexelSize", texel, texel)

        shadow_bind_id = self._shadow_bind_id(sun)

        bgl.glActiveTexture(bgl.GL_TEXTURE3)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, shadow_bind_id)

        shader.setUniform1i("shadowMap", 3)
        shader.setUniform1i("shadowEnabled", 1 if shadow_bind_id > 0 else 0)

    def _shadow_matrix(self, light):

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

            return bias * proj * world_to_lamp

        except Exception:
            return getattr(light, "shadowMatrix", Matrix.Identity(4))

    def _shadow_bind_id(self, light):

        bind_id = int(getattr(light, "shadowBindId", 0) or 0)

        if bind_id <= 0 and not self._warned_shadow_unavailable:
            print("[DeepShadowVolumetric] shadow map nao encontrado")
            self._warned_shadow_unavailable = True

        return bind_id

    def _shadow_texel_size(self, light):

        size = int(
            getattr(light, "shadowMapSize", 0)
            or getattr(light, "shadowBufferSize", 0)
            or 1024
        )

        return 1.0 / float(max(size, 1))


from Range import *
import mathutils
from collections import OrderedDict


# =========================
# VERTEX SHADER (WORLD SPACE)
# =========================
vertexShader = """
varying vec3 vWorldPos;

uniform mat4 objectMatrix;

void main()
{
    vec4 worldPos = objectMatrix * gl_Vertex;
    vWorldPos = worldPos.xyz;

    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
"""


# =========================
# FRAGMENT SHADER (WORLD SPACE)
# =========================
fragmentShader = """
#define MAX_SAMPLES 64

uniform vec3 lightDir;     // direção normalizada DO fragmento PARA o sol
uniform vec3 cameraPos;    // posição da câmera no world space

uniform vec3 windowPos;
uniform vec3 windowNormal;
uniform vec3 windowRight;  // eixo local X da janela (world space)
uniform vec3 windowUp;     // eixo local Y da janela (world space)
uniform vec2 windowSize;   // meia-extensão da janela (world scale)

uniform vec3  lightColor;      // cor da luz volumétrica (RGB)
uniform float lightIntensity;  // intensidade geral
uniform int   numSamples;      // passos de ray march (max MAX_SAMPLES)
uniform float scattering;      // coeficiente de espalhamento
uniform float falloffScale;    // velocidade de queda de intensidade com a distância
uniform float marchStep;       // tamanho fixo do passo em world-units

varying vec3 vWorldPos;

bool passesThroughWindow(vec3 point)
{
    float denom = dot(lightDir, windowNormal);
    if (abs(denom) < 0.0001)
        return false;

    float t = dot(windowPos - point, windowNormal) / denom;
    if (t < 0.0)
        return false;

    vec3 hit   = point + lightDir * t;
    vec3 toHit = hit - windowPos;

    float x = dot(toHit, windowRight);
    float y = dot(toHit, windowUp);

    return abs(x) < windowSize.x && abs(y) < windowSize.y;
}

void main()
{
    // Parte da superfície do fragmento em direção à câmera com passo fixo.
    // Isso torna o efeito completamente independente da distância da câmera
    // e garante que todos os samples estejam dentro do volume (cubo).
    vec3  viewDir  = normalize(cameraPos - vWorldPos);
    float stepSize = marchStep;   // world-units fixo
    float accumLight = 0.0;

    for (int i = 0; i < MAX_SAMPLES; i++)
    {
        if (i >= numSamples) break;

        float t         = (float(i) + 0.5) * stepSize;
        vec3  samplePos = vWorldPos + viewDir * t;

        if (passesThroughWindow(samplePos))
        {
            // falloff quadrático pela distância ao plano da janela
            float distFromWindow = max(0.0, dot(samplePos - windowPos, -lightDir));
            float falloff        = 1.0 / (1.0 + falloffScale * distFromWindow * distFromWindow);
            accumLight += scattering * stepSize * falloff;
        }
    }

    accumLight = clamp(accumLight * lightIntensity, 0.0, 1.0);

    vec3 color = lightColor * accumLight;
    gl_FragColor = vec4(color, accumLight);
}
"""


# =========================
# PYTHON COMPONENT
# =========================
class VolumetricLight(types.KX_PythonComponent):

    args = OrderedDict([
        ("Light Color",      (1.0, 0.85, 0.6)),
        ("Light Intensity",  2.5),
        ("Num Samples",      32),
        ("Scattering",       0.04),
        ("Falloff Scale",    0.05),
    ])

    def start(self, args):

        self.scene  = logic.getCurrentScene()
        self.shader = self.object.meshes[0].materials[0].getShader()

        if self.shader and not self.shader.isValid():
            self.shader.setSource(vertexShader, fragmentShader, 1)

        print("[VolumetricLight] Shader valid:", self.shader.isValid())

        # parâmetros volumétricos
        lc = args.get("Light Color",     (1.0, 0.85, 0.6))
        self._light_color     = lc if isinstance(lc, (list, tuple)) else (1.0, 0.85, 0.6)
        self._light_intensity = float(args.get("Light Intensity", 2.5))
        self._num_samples     = int(args.get("Num Samples",        32))
        self._march_step      = float(args.get("March Step",       0.05))
        self._scattering      = float(args.get("Scattering",       0.04))
        self._falloff_scale   = float(args.get("Falloff Scale",    0.05))

        self._debug_counter = 0

    def update(self):

        light = self.scene.objects.get("Sun")
        window = self.scene.objects.get("WindowPortal")

        self._debug_counter = getattr(self, '_debug_counter', 0) + 1
        if self._debug_counter % 60 == 1:
            obj_names = [o.name for o in self.scene.objects]
            if not light:
                print("[VolumetricLight] ERRO: objeto 'Sun' nao encontrado!")
                print("  Objetos na cena:", obj_names)
            if not window:
                print("[VolumetricLight] ERRO: objeto 'WindowPortal' nao encontrado!")
                print("  Objetos na cena:", obj_names)
            if light and window:
                sun_dir_log = light.worldOrientation.col[2]
                print("[VolumetricLight] lightDir (col[2] do Sun):", list(sun_dir_log))
                print("[VolumetricLight] windowPos:", list(window.worldPosition))
                print("[VolumetricLight] windowNormal (Z local):", list(window.worldOrientation.col[2]))
                print("[VolumetricLight] cuboPos:", list(self.object.worldPosition))

        if not light or not window:
            return

        # matriz do cubo
        self.shader.setUniformMatrix4(
            "objectMatrix",
            self.object.worldTransform
        )

        # direção do sol: no Blender o sol emite na direção -Z local,
        # portanto a direção DO fragmento PARA o sol é o +Z local do objeto Sun
        sun_dir = light.worldOrientation.col[2]
        self.shader.setUniform3f("lightDir", sun_dir.x, sun_dir.y, sun_dir.z)

        if self._debug_counter % 60 == 1:
            print("[VolumetricLight] lightDir:", list(sun_dir))

        # janela world — eixo local mais alinhado com a direção do sol
        to_light = sun_dir  # direção paralela do sol, igual para todos os pontos
        orient = window.worldOrientation
        best_axis = max([0, 1, 2], key=lambda i: abs(orient.col[i].dot(to_light)))
        normal = orient.col[best_axis]
        # garante que aponta em direção à luz (e não para o lado oposto)
        if normal.dot(to_light) < 0:
            normal = -normal

        if self._debug_counter % 60 == 1:
            print("[VolumetricLight] eixo normal escolhido:", best_axis, "normal:", list(normal))

        # os outros dois eixos locais definem right e up da janela
        other_axes = [i for i in [0, 1, 2] if i != best_axis]
        win_right = orient.col[other_axes[0]]
        win_up    = orient.col[other_axes[1]]
        win_scale = window.worldScale
        size_x = win_scale[other_axes[0]]
        size_y = win_scale[other_axes[1]]

        self.shader.setUniform3f("windowNormal", normal.x, normal.y, normal.z)
        self.shader.setUniform3f("windowRight", win_right.x, win_right.y, win_right.z)
        self.shader.setUniform3f("windowUp",    win_up.x,    win_up.y,    win_up.z)
        self.shader.setUniform2f("windowSize",  size_x, size_y)
        self.shader.setUniform3f("windowPos", *window.worldPosition)

        # câmera
        cam_pos = self.scene.active_camera.worldPosition
        self.shader.setUniform3f("cameraPos", cam_pos.x, cam_pos.y, cam_pos.z)

        # uniforms volumétricos
        lc = self._light_color
        self.shader.setUniform3f("lightColor",     float(lc[0]), float(lc[1]), float(lc[2]))
        self.shader.setUniform1f("lightIntensity",  self._light_intensity)
        self.shader.setUniform1i("numSamples",      self._num_samples)
        self.shader.setUniform1f("marchStep",       self._march_step)
        self.shader.setUniform1f("scattering",      self._scattering)
        self.shader.setUniform1f("falloffScale",    self._falloff_scale)

        if self._debug_counter % 60 == 1:
            print("[VolumetricLight] windowRight:", list(win_right), "windowUp:", list(win_up))
            print("[VolumetricLight] windowSize (half-extent):", size_x, size_y)
            print("[VolumetricLight] cameraPos:", list(cam_pos))
            print("[VolumetricLight] lightColor:", self._light_color,
                  "intensity:", self._light_intensity,
                  "samples:", self._num_samples,
                  "marchStep:", self._march_step)
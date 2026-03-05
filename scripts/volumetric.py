from Range import *
import mathutils
from collections import OrderedDict

# Limite de amostras — valor injetado no #define do GLSL via f-string.
# Altere aqui para mudar nos dois lugares ao mesmo tempo.
MAX_SHADER_SAMPLES = 64

# =========================
# VERTEX SHADER
# =========================
vertexShader = """
varying vec3 vWorldPos;

uniform mat4 objectMatrix;

void main()
{
    vWorldPos = (objectMatrix * gl_Vertex).xyz;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
"""

# =========================
# FRAGMENT SHADER
# =========================
fragmentShader = f"""
#define MAX_SAMPLES {MAX_SHADER_SAMPLES}

// --- Janela (portal de luz) ---
uniform vec3  windowPos;    // centro da janela no world space
uniform vec3  windowNormal; // normal do plano da janela (aponta para o sol)
uniform vec3  windowRight;  // eixo local X da janela
uniform vec3  windowUp;     // eixo local Y da janela
uniform vec2  windowSize;   // meia-extensão (half-extent) da janela

// --- Volume ---
uniform mat4  objectMatrixInv; // inversa da world transform do volume
uniform vec3  boundsMin;       // AABB mínimo em espaço de objeto
uniform vec3  boundsMax;       // AABB máximo em espaço de objeto

// --- Câmera e luz ---
uniform vec3  cameraPos;       // posição da câmera no world space
uniform vec3  lightDir;        // direção normalizada DO ponto PARA o sol
uniform vec3  lightColor;      // cor da luz volumétrica (RGB)
uniform float lightIntensity;  // multiplicador de intensidade final

// --- Ray march ---
uniform int   numSamples;   // número de passos (max MAX_SAMPLES)
uniform float marchStep;    // tamanho do passo em world-units
uniform float scattering;   // coeficiente de espalhamento por passo
uniform float falloffScale; // velocidade de queda quadrática com distância à janela
uniform float time;         // tempo em segundos (seed do jitter)

varying vec3 vWorldPos;

// Retorna true se o raio partindo de 'point' na direção 'lightDir'
// intersecta o rectângulo da janela.
bool passesThroughWindow(vec3 point)
{{
    float denom = dot(lightDir, windowNormal);
    if (abs(denom) < 0.0001) return false;

    float t = dot(windowPos - point, windowNormal) / denom;
    if (t < 0.0) return false;

    vec3 toHit = (point + lightDir * t) - windowPos;
    return abs(dot(toHit, windowRight)) < windowSize.x &&
           abs(dot(toHit, windowUp))    < windowSize.y;
}}

// Hash pseudo-aleatório por pixel: quebra o banding sem custo extra de amostras.
float rand(vec2 co, float t)
{{
    return fract(sin(dot(co + t, vec2(127.1, 311.7))) * 43758.5453);
}}

void main()
{{
    vec3 rayDir = normalize(vWorldPos - cameraPos);

    // Slab test: transforma o raio para espaço de objeto e intersecta o AABB.
    // Como 'rayDir' é normalizado, o parâmetro t corresponde a distância
    // world-units, portanto cameraPos + rayDir * t é válido diretamente.
    vec3  rayOriginObj = (objectMatrixInv * vec4(cameraPos, 1.0)).xyz;
    vec3  rayDirObj    = (objectMatrixInv * vec4(rayDir,    0.0)).xyz;
    vec3  t0    = (boundsMin - rayOriginObj) / rayDirObj;
    vec3  t1    = (boundsMax - rayOriginObj) / rayDirObj;
    float tNear = max(max(min(t0.x,t1.x), min(t0.y,t1.y)), min(t0.z,t1.z));
    float tFar  = min(min(max(t0.x,t1.x), max(t0.y,t1.y)), max(t0.z,t1.z));

    if (tNear > tFar || tFar < 0.0) {{ gl_FragColor = vec4(0.0); return; }}
    tNear = max(tNear, 0.0); // câmera dentro do volume: começar na câmera

    float jitter     = rand(gl_FragCoord.xy, time);
    float accumLight = 0.0;

    for (int i = 0; i < MAX_SAMPLES; i++)
    {{
        if (i >= numSamples) break;

        float t = tNear + (float(i) + 0.5 + jitter) * marchStep;
        if (t >= tFar) break;

        vec3 samplePos = cameraPos + rayDir * t;

        if (passesThroughWindow(samplePos))
        {{
            float dist    = max(0.0, dot(samplePos - windowPos, -lightDir));
            float falloff = 1.0 / (1.0 + falloffScale * dist * dist);
            accumLight   += scattering * marchStep * falloff;
        }}
    }}

    accumLight = clamp(accumLight * lightIntensity, 0.0, 1.0);
    gl_FragColor = vec4(lightColor * accumLight, accumLight);
}}
"""


# =========================
# PYTHON COMPONENT
# =========================
class VolumetricLight(types.KX_PythonComponent):

    args = OrderedDict([
        ("Light Color",     (1.0, 0.88, 0.7)),
        ("Light Intensity", 1.0),
        ("Num Samples",     32),
        ("March Step",      0.0),  # 0.0 = derivar automaticamente da diagonal do volume
        ("Scattering",      0.5),
        ("Falloff Scale",   0.05),
    ])

    def start(self, args):
        self.scene  = logic.getCurrentScene()
        self.shader = self.object.meshes[0].materials[0].getShader()

        if self.shader and not self.shader.isValid():
            self.shader.setSource(vertexShader, fragmentShader, 1)
        print("[VolumetricLight] Shader valid:", self.shader.isValid())

        # Parâmetros volumétricos
        lc = args.get("Light Color", (1.0, 0.88, 0.7))
        self._light_color     = lc if isinstance(lc, (list, tuple)) else (1.0, 0.88, 0.7)
        self._light_intensity = float(args.get("Light Intensity", 1.0))
        self._scattering      = float(args.get("Scattering",      0.5))
        self._falloff_scale   = float(args.get("Falloff Scale",   0.05))
        self._march_step      = float(args.get("March Step",      0.0))

        self._num_samples = int(args.get("Num Samples", 32))
        if self._num_samples > MAX_SHADER_SAMPLES:
            print(f"[VolumetricLight] AVISO: Num Samples ({self._num_samples}) "
                  f"excede MAX_SHADER_SAMPLES ({MAX_SHADER_SAMPLES}). Valor reduzido.")
            self._num_samples = MAX_SHADER_SAMPLES

        # Calcular AABB da mesh em espaço de objeto (válido para qualquer forma)
        mesh = self.object.meshes[0]
        min_x = min_y = min_z =  1e9
        max_x = max_y = max_z = -1e9
        for mat_idx in range(mesh.numMaterials):
            for v_idx in range(mesh.getVertexArrayLength(mat_idx)):
                x, y, z = mesh.getVertex(mat_idx, v_idx).XYZ
                if x < min_x: min_x = x
                if y < min_y: min_y = y
                if z < min_z: min_z = z
                if x > max_x: max_x = x
                if y > max_y: max_y = y
                if z > max_z: max_z = z
        self._bounds_min = (min_x, min_y, min_z)
        self._bounds_max = (max_x, max_y, max_z)
        print(f"[VolumetricLight] AABB obj space: min={self._bounds_min} max={self._bounds_max}")

        self._debug_counter = 0

    def update(self):
        light  = self.scene.objects.get("Sun")
        window = self.scene.objects.get("WindowPortal")

        self._debug_counter += 1
        if self._debug_counter % 60 == 1:
            self._log_debug(light, window)

        if not light or not window:
            return

        self._update_uniforms(light, window)

    # ------------------------------------------------------------------

    def _log_debug(self, light, window):
        obj_names = [o.name for o in self.scene.objects]
        if not light:
            print("[VolumetricLight] ERRO: 'Sun' nao encontrado! Cena:", obj_names)
        if not window:
            print("[VolumetricLight] ERRO: 'WindowPortal' nao encontrado! Cena:", obj_names)
        if light and window:
            print("[VolumetricLight] lightDir  :", list(light.worldOrientation.col[2]))
            print("[VolumetricLight] windowPos :", list(window.worldPosition))
            print("[VolumetricLight] volumePos :", list(self.object.worldPosition))
            print("[VolumetricLight] cameraPos :", list(self.scene.active_camera.worldPosition))

    def _update_uniforms(self, light, window):
        # --- Volume: transform e AABB ---
        world_mat = self.object.worldTransform
        self.shader.setUniformMatrix4("objectMatrix",    world_mat)
        self.shader.setUniformMatrix4("objectMatrixInv", world_mat.inverted())
        self.shader.setUniform3f("boundsMin", *self._bounds_min)
        self.shader.setUniform3f("boundsMax", *self._bounds_max)

        # --- Luz direcional ---
        # O sol na Range emite em -Z local; +Z local aponta para o sol.
        sun_dir = light.worldOrientation.col[2]
        self.shader.setUniform3f("lightDir", sun_dir.x, sun_dir.y, sun_dir.z)

        # --- Janela ---
        # Usa o eixo local mais alinhado com o sol como normal do plano,
        # garantindo dot(lightDir, windowNormal) != 0 para a interseção.
        orient    = window.worldOrientation
        best_axis = max([0, 1, 2], key=lambda i: abs(orient.col[i].dot(sun_dir)))
        normal    = orient.col[best_axis]
        if normal.dot(sun_dir) < 0:
            normal = -normal
        other     = [i for i in (0, 1, 2) if i != best_axis]
        win_right = orient.col[other[0]]
        win_up    = orient.col[other[1]]
        scale     = window.worldScale

        self.shader.setUniform3f("windowNormal", normal.x,    normal.y,    normal.z)
        self.shader.setUniform3f("windowRight",  win_right.x, win_right.y, win_right.z)
        self.shader.setUniform3f("windowUp",     win_up.x,    win_up.y,    win_up.z)
        self.shader.setUniform2f("windowSize",   scale[other[0]], scale[other[1]])
        self.shader.setUniform3f("windowPos",    *window.worldPosition)

        # --- Câmera ---
        cam_pos = self.scene.active_camera.worldPosition
        self.shader.setUniform3f("cameraPos", cam_pos.x, cam_pos.y, cam_pos.z)

        # --- Uniforms volumétricos ---
        lc = self._light_color
        self.shader.setUniform3f("lightColor",    float(lc[0]), float(lc[1]), float(lc[2]))
        self.shader.setUniform1f("lightIntensity", self._light_intensity)
        self.shader.setUniform1i("numSamples",     self._num_samples)
        self.shader.setUniform1f("scattering",     self._scattering)
        self.shader.setUniform1f("falloffScale",   self._falloff_scale)
        self.shader.setUniform1f("time",           logic.getRealTime())

        # marchStep: manual se > 0, senão deriva da diagonal do AABB
        if self._march_step > 0.0:
            march_step = self._march_step
        else:
            diagonal   = (mathutils.Vector(self._bounds_max) - mathutils.Vector(self._bounds_min)).length
            march_step = (diagonal * max(self.object.worldScale)) / max(self._num_samples, 1)
        self.shader.setUniform1f("marchStep", march_step)
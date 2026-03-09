from Range import *
import mathutils
from collections import OrderedDict

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
# FRAGMENT SHADER  —  abordagem 100% analítica, sem loop
#
# Em vez de amostrar N pontos ao longo do raio (ray marching), descrevemos
# analiticamente o conjunto de pontos do raio que estão dentro do prisma de luz.
#
# Para luz direcional + janela retangular, a condição de iluminação é:
#   |offsetR + t*slopeR| < windowSize.x   (dentro da largura da janela)
#   |offsetU + t*slopeU| < windowSize.y   (dentro da altura da janela)
#   s(t) >= 0                             (do lado correto da janela)
#
# Cada condição é linear em t → dá um intervalo [t0, t1].
# A interseção de todos os intervalos (+ AABB slab test) dá o segmento iluminado exato.
#
# O falloff exponencial é integrado analiticamente:
#   ∫ exp(-k * dist(t)) dt  com dist(t) = d0 + dr*t  (linear em t)
#   = -1/(k*dr) * [exp(-k*(d0+dr*tB)) - exp(-k*(d0+dr*tA))]
# =========================
fragmentShader = """
// --- Janela (portal de luz) ---
uniform vec3  windowPos;
uniform vec3  windowNormal;
uniform vec3  windowRight;
uniform vec3  windowUp;
uniform vec2  windowSize;

// --- Volume ---
uniform mat4  objectMatrixInv;
uniform vec3  boundsMin;
uniform vec3  boundsMax;

// --- Câmera e luz ---
uniform vec3  cameraPos;
uniform vec3  lightDir;
uniform vec3  lightColor;
uniform float lightIntensity;

// --- Parâmetros volumétricos ---
uniform float scattering;   // densidade do meio
uniform float falloffScale; // coeficiente de attenuação exponencial com distância à janela

varying vec3 vWorldPos;

void main()
{
    vec3 rayDir = normalize(vWorldPos - cameraPos);

    // --- Slab test AABB em espaço de objeto ---
    // Como rayDir é normalizado, t é distância real em world-units desde a câmera.
    vec3 ro = (objectMatrixInv * vec4(cameraPos, 1.0)).xyz;
    vec3 rd = (objectMatrixInv * vec4(rayDir,    0.0)).xyz;
    // Evita divisão por zero nos eixos paralelos
    rd = mix(rd, sign(rd) * vec3(1e-5), lessThan(abs(rd), vec3(1e-5)));
    vec3  t0    = (boundsMin - ro) / rd;
    vec3  t1    = (boundsMax - ro) / rd;
    float tNear = max(max(min(t0.x,t1.x), min(t0.y,t1.y)), min(t0.z,t1.z));
    float tFar  = min(min(max(t0.x,t1.x), max(t0.y,t1.y)), max(t0.z,t1.z));
    if (tNear > tFar || tFar < 0.0) { gl_FragColor = vec4(0.0); return; }
    tNear = max(tNear, 0.0);

    // --- Interseção analítica com o prisma de luz ---
    // Para luz direcional, o prisma é o conjunto de pontos P tal que o raio
    // P + s*lightDir intersecta a janela com s >= 0.
    // A projeção P(t) = cameraPos + t*rayDir sobre os eixos da janela é linear em t,
    // portanto cada condição de bound é um intervalo fechado em t.
    float denom = dot(lightDir, windowNormal);
    if (abs(denom) < 1e-4) { gl_FragColor = vec4(0.0); return; }

    float A = dot(windowPos - cameraPos, windowNormal); // dist cacm->janela
    float B = dot(rayDir, windowNormal);

    // Projeção no eixo Right: f(t) = offsetR + t*slopeR
    float lR      = dot(lightDir, windowRight) / denom;
    float offsetR = dot(cameraPos - windowPos, windowRight) + A * lR;
    float slopeR  = dot(rayDir, windowRight) - B * lR;
    float tR0, tR1;
    if (abs(slopeR) > 1e-6) {
        tR0 = (-windowSize.x - offsetR) / slopeR;
        tR1 = ( windowSize.x - offsetR) / slopeR;
        if (tR0 > tR1) { float tmp = tR0; tR0 = tR1; tR1 = tmp; }
    } else if (abs(offsetR) >= windowSize.x) {
        gl_FragColor = vec4(0.0); return;
    } else { tR0 = -1e9; tR1 = 1e9; }

    // Projeção no eixo Up: g(t) = offsetU + t*slopeU
    float lU      = dot(lightDir, windowUp) / denom;
    float offsetU = dot(cameraPos - windowPos, windowUp) + A * lU;
    float slopeU  = dot(rayDir, windowUp) - B * lU;
    float tU0, tU1;
    if (abs(slopeU) > 1e-6) {
        tU0 = (-windowSize.y - offsetU) / slopeU;
        tU1 = ( windowSize.y - offsetU) / slopeU;
        if (tU0 > tU1) { float tmp = tU0; tU0 = tU1; tU1 = tmp; }
    } else if (abs(offsetU) >= windowSize.y) {
        gl_FragColor = vec4(0.0); return;
    } else { tU0 = -1e9; tU1 = 1e9; }

    // Condição s(t) >= 0: amostra do lado certo da janela (luz viaja para frente)
    float tS0, tS1;
    if (abs(B) > 1e-6) {
        float tS = A / B;
        if (denom * B > 0.0) { tS0 = -1e9; tS1 = tS; }
        else                 { tS0 = tS;   tS1 = 1e9; }
    } else if ((denom > 0.0 && A < 0.0) || (denom < 0.0 && A > 0.0)) {
        gl_FragColor = vec4(0.0); return;
    } else { tS0 = -1e9; tS1 = 1e9; }

    // Segmento final iluminado: interseção de todos os intervalos
    float tA = max(max(max(tNear, tR0), tU0), tS0);
    float tB = min(min(min(tFar,  tR1), tU1), tS1);
    if (tA >= tB) { gl_FragColor = vec4(0.0); return; }

    // --- Integração analítica do falloff exponencial sobre [tA, tB] ---
    // falloff(t) = exp(-falloffScale * dist(t))
    // dist(t) = max(0, d0 + dr*t)  com d0 e dr constantes
    // ∫ exp(-k*(d0+dr*t)) dt = -1/(k*dr) * exp(-k*(d0+dr*t))
    float d0 = dot(windowPos - cameraPos, lightDir);
    float dr = -dot(rayDir, lightDir);
    float k  = falloffScale;

    float accumLight;
    if (abs(dr) < 1e-6 || k < 1e-9) {
        // falloff constante ao longo do raio
        float dist    = max(0.0, d0 + dr * (tA + tB) * 0.5);
        float falloff = exp(-k * dist);
        accumLight = scattering * falloff * (tB - tA);
    } else {
        // Integral exata: [-1/(k*dr)] * [exp(-k*(d0+dr*tB)) - exp(-k*(d0+dr*tA))]
        float eA = exp(-k * max(0.0, d0 + dr * tA));
        float eB = exp(-k * max(0.0, d0 + dr * tB));
        accumLight = scattering * abs((eA - eB) / (k * dr));
    }

    float lit = clamp(accumLight * lightIntensity, 0.0, 0.75);
    gl_FragColor = vec4(lightColor * lit, lit);
}
"""


# =========================
# PYTHON COMPONENT
# =========================
class VolumetricLight(types.KX_PythonComponent):

    args = OrderedDict([
        ("Light Color",     (1.0, 0.88, 0.7)),
        ("Light Intensity", 1.0),
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
        self.shader.setUniform1f("scattering",     self._scattering)
        self.shader.setUniform1f("falloffScale",   self._falloff_scale)
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
uniform vec3 lightDir;    // direção normalizada DO fragmento PARA o sol (paralela, sol no infinito)

uniform vec3 windowPos;
uniform vec3 windowNormal;
uniform vec3 windowRight;  // eixo local X da janela (world space)
uniform vec3 windowUp;     // eixo local Y da janela (world space)
uniform vec2 windowSize;   // tamanho total da janela (worldScale)

varying vec3 vWorldPos;

bool passesThroughWindow(vec3 point)
{
    vec3 dir = lightDir;

    float denom = dot(dir, windowNormal);
    if(abs(denom) < 0.0001)
        return false;

    float t = dot(windowPos - point, windowNormal) / denom;

    if(t < 0.0)
        return false;

    vec3 hit = point + dir * t;
    vec3 toHit = hit - windowPos;

    // usa os eixos reais do objeto janela, sem recalcular
    float x = dot(toHit, windowRight);
    float y = dot(toHit, windowUp);

    // windowSize = meia-extensão (worldScale de um plano padrão Blender com vértices em ±1)
    return abs(x) < windowSize.x &&
           abs(y) < windowSize.y;
}

void main()
{
    // DEBUG
    gl_FragColor = vec4(vec3(passesThroughWindow(vWorldPos)), 1.0);
}
"""


# =========================
# PYTHON COMPONENT
# =========================
class VolumetricLight(types.KX_PythonComponent):

    args = OrderedDict([])

    def start(self, args):

        self.scene = logic.getCurrentScene()
        self.shader = self.object.meshes[0].materials[0].getShader()

        if self.shader and not self.shader.isValid():
            self.shader.setSource(vertexShader, fragmentShader, 1)

        print("[VolumetricLight] Shader valid:", self.shader.isValid())

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

        if self._debug_counter % 60 == 1:
            print("[VolumetricLight] windowRight:", list(win_right), "windowUp:", list(win_up))
            print("[VolumetricLight] windowSize (half-extent):", size_x, size_y)
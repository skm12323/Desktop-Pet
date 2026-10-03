// render3d 运行时场景（M1）——View3D + 透明背景 + toon + 分级光照。
// 属性由 scene_host.py 注入（契约 SceneState 经 lighting.compute 的 as_qml payload）。
// 骨骼驱动（bone_bridge 批量通道）待 S1.4 蒙皮模型就绪后接入，届时本场景加 Joint 层。
import QtQuick
import QtQuick3D

Rectangle {
    id: root
    color: "transparent"

    // ---- 注入属性（scene_host 逐帧/按需写） ----
    property url meshUrl: ""
    property bool spin: false
    property int lightLevel: 1
    property real lightDirX: 0
    property real lightDirY: 1
    property real lightDirZ: 0
    property real lightColorR: 1
    property real lightColorG: 1
    property real lightColorB: 1
    property real lightIntensity: 1
    property real ambientR: 0.35
    property real ambientG: 0.37
    property real ambientB: 0.45
    property bool shadowEnabled: false
    property real wetness: 0

    View3D {
        anchors.fill: parent

        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Transparent
            tonemapMode: SceneEnvironment.TonemapModeLinear
        }

        PerspectiveCamera {
            id: cam
            position: Qt.vector3d(0, 90, 260)
            clipNear: 1
            clipFar: 2000
        }

        DirectionalLight {
            // 方向向量 → euler：绕 y 偏航 = atan2(x, z)；绕 x 俯仰 = -asin(y)
            eulerRotation.y: root.lightDirX === 0 && root.lightDirZ === 0
                             ? 0 : Math.atan2(root.lightDirX, root.lightDirZ) * 180 / Math.PI
            eulerRotation.x: -Math.asin(Math.max(-1, Math.min(1, root.lightDirY))) * 180 / Math.PI
            brightness: 1.4 * root.lightIntensity   // Quick3D 亮度基准=1（spike 验证 1.4；100 会过曝成白块）
            ambientColor: Qt.rgba(root.ambientR, root.ambientG, root.ambientB, 1.0)
            visible: root.lightLevel > 0
        }

        Node {
            id: pivot

            Model {
                source: root.meshUrl
                scale: Qt.vector3d(150, 150, 150)
                position: Qt.vector3d(0, -75, 0)
                materials: toonMat
                visible: root.meshUrl !== ""
            }
            // 占位：TRELLIS 网格 y∈[0,1] 需下移贴地（Hunyuan 版 y∈[0,1.5] 时改回 -112.5）

            SequentialAnimation on eulerRotation.y {
                running: root.spin
                loops: Animation.Infinite
                NumberAnimation { from: 0; to: 360; duration: 6000 }
            }
        }

        CustomMaterial {
            id: toonMat
            shadingMode: CustomMaterial.Shaded
            // 基色占位（S1.1 正式网格贴图就绪后走贴图采样，uniform 只做演示档）
            property color uBase: "#7c86b8"
            property real uStep: 0.5
            property vector3d uAmbient: Qt.vector3d(root.ambientR, root.ambientG, root.ambientB)
            property vector3d uRim: Qt.vector3d(0.5, 0.55, 0.7)
            property real uWet: root.wetness
            property real uLightGain: root.lightIntensity
            property real uHasTex: 1.0
            property TextureInput uBaseTex: TextureInput {
                texture: Texture { source: "maps/textureData.png"; generateMipmaps: true; mipFilter: Texture.Linear }
            }
            fragmentShader: "toon_materials/toon.frag"
        }
    }
}

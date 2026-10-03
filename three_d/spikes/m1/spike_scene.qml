// M1 spike 场景：View3D + 透明背景 + 草模 mesh（Balsam 产物）。
// 由 spike_qml3d.py 注入 mesh 路径（绝对 file:// URL）与可选的自动旋转。
import QtQuick
import QtQuick3D

Rectangle {
    id: root
    color: "transparent"

    property string meshUrl: ""
    property bool spin: false

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
            eulerRotation.x: -35
            eulerRotation.y: 25
            brightness: 1.4
            ambientColor: Qt.rgba(0.55, 0.55, 0.6, 1.0)
        }

        Node {
            id: pivot

            Model {
                source: root.meshUrl
                scale: Qt.vector3d(150, 150, 150)   // 米制 1.5m → 场景单位
                position: Qt.vector3d(0, -75, 0)     // 脚底对齐视口下缘
                materials: PrincipledMaterial {
                    // 草模（减面+merge）绕向混乱，Qt 默认背面剔除会把反向三角剔除成
                    // "雪花洞"（Blender 双面渲染故此前未见）；正式建模网格无此问题。
                    // spike 用 NoCulling 保剪影完整；toon 材质是 S2.3 的事。
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: "#9aa4c0"
                    cullMode: PrincipledMaterial.NoCulling
                    alphaMode: PrincipledMaterial.Opaque
                }
            }

            SequentialAnimation on eulerRotation.y {
                running: root.spin
                loops: Animation.Infinite
                NumberAnimation { from: 0; to: 360; duration: 6000 }
            }
        }
    }
}

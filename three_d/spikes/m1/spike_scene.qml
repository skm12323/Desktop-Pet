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
                materials: toonMat

                CustomMaterial {
                    id: toonMat
                    shadingMode: CustomMaterial.Shaded
                    property color uBase: "#7c86b8"
                    property real uStep: 0.5
                    property vector3d uAmbient: Qt.vector3d(0.30, 0.32, 0.40)
                    property vector3d uRim: Qt.vector3d(0.55, 0.6, 0.75)
                    fragmentShader: "toon.frag"
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

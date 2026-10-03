// Toon 阶梯光 —— CustomMaterial 片元钩子（S2.3 首刀）。
// Qt 6.x API：零参数钩子 + 扁平特殊变量。阶梯两段 + rim 边缘光。
// 透明度走 BASE_COLOR 的 alpha 通道（此 API 无独立 ALPHA 变量，写它=编译失败隐身）。
void MAIN()
{
    BASE_COLOR = uHasTex > 0.5 ? texture(uBaseTex, UV0) : uBase;
}

void DIRECTIONAL_LIGHT()
{
    float ndl = clamp(dot(normalize(VAR_WORLD_NORMAL), normalize(TO_LIGHT_DIR)), 0.0, 1.0);
    float band = ndl > uStep ? 1.0 : 0.55;
    DIFFUSE += LIGHT_COLOR * BASE_COLOR.rgb * band * uLightGain;
}

void AMBIENT_LIGHT()
{
    vec3 n = normalize(VAR_WORLD_NORMAL);
    vec3 v = normalize(CAMERA_POSITION - VAR_WORLD_POSITION);
    float rim = pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 3.0);
    DIFFUSE = min(DIFFUSE + uAmbient * BASE_COLOR.rgb + uRim * rim, vec3(1.2));
}

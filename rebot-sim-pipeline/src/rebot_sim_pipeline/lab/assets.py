"""Physics-ready USD wrappers for the stationery objects.

``stationery_models.usd`` (from the single-arm reBot project) holds bare meshes
whose material lives outside each object, so a single object cannot be spawned
from it directly. :func:`prepare_stationery_assets` writes one small USDA per
object that references the source mesh, rebuilds the Poly Haven PBR material
next to it, and authors the same rigid-body, mass, collision and friction
settings as the single-arm scene (``build_scene.py``). The pencil cup gets a
static triangle-mesh collider.

Needs ``pxr`` (any Isaac Sim Python). Files are regenerated when the task config
or the source paths change.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_TEXTURES = {
    "diffuse": "textures/stationery_supplies_diff_4k.jpg",
    "arm": "textures/stationery_supplies_arm_4k.jpg",
    "normal": "textures/stationery_supplies_nor_gl_4k.exr",
}


def default_output_dir() -> Path:
    """``<checkout>/assets/generated`` for a source checkout, else ~/.cache."""
    from ..cli import PACKAGE_ROOT

    if PACKAGE_ROOT is not None:
        return PACKAGE_ROOT / "assets" / "generated"
    return Path.home() / ".cache" / "rebot_sim_pipeline" / "assets"


def _add_material(stage, path: str, stationery_dir: Path):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{path}/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def texture(name: str, key: str, color_space: str):
        node = UsdShade.Shader.Define(stage, f"{path}/{name}")
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(stationery_dir / _TEXTURES[key]))
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        return node

    diffuse = texture("DiffuseTexture", "diffuse", "sRGB")
    arm = texture("ArmTexture", "arm", "raw")
    normal = texture("NormalTexture", "normal", "raw")
    normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
    normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuse.ConnectableAPI(), "rgb")
    # Poly Haven's ARM map stores AO, roughness and metalness in RGB.
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(arm.ConnectableAPI(), "g")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(arm.ConnectableAPI(), "b")
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _apply_physx(prim, schema: str, values: Mapping[str, Any], tokens: Mapping[str, str] = {}) -> None:
    """Apply a PhysX API schema by name (PhysxSchema is only importable inside Kit)."""
    from pxr import Sdf

    prim.AddAppliedSchema(schema)
    for name, value in values.items():
        if name in tokens:
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Token, custom=False).Set(tokens[name])
        elif isinstance(value, bool):
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool, custom=False).Set(value)
        else:
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(float(value))


def _write_object(
    out: Path, name: str, source_usd: Path, source_prim: str, *, dynamic: bool,
    mass: float | None, physics: Mapping[str, Any],
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_path = f"/{name}"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(str(source_usd), source_prim)

    visual = _add_material(stage, f"{root_path}/Looks/StationeryPBR", source_usd.parent)
    mesh = stage.OverridePrim(f"{root_path}/Mesh")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(visual, UsdShade.Tokens.strongerThanDescendants)
    UsdPhysics.CollisionAPI.Apply(mesh)
    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("convexHull" if dynamic else "none")

    if dynamic:
        rigid = UsdPhysics.RigidBodyAPI.Apply(root)
        rigid.CreateKinematicEnabledAttr(False)
        _apply_physx(root, "PhysxRigidBodyAPI", {
            "physxRigidBody:maxDepenetrationVelocity": float(physics["max_depenetration_velocity"]),
            "physxRigidBody:maxContactImpulse": 0.05,
        })
        mass_api = UsdPhysics.MassAPI.Apply(root)
        mass_api.CreateMassAttr(float(mass))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        _apply_physx(mesh, "PhysxCollisionAPI", {
            "physxCollision:contactOffset": float(physics["contact_offset"]),
            "physxCollision:restOffset": 0.0,
            "physxCollision:torsionalPatchRadius": 0.002,
            "physxCollision:minTorsionalPatchRadius": 0.002,
        })

        material = UsdShade.Material.Define(stage, f"{root_path}/Looks/StationeryPhysics")
        physics_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_api.CreateStaticFrictionAttr(float(physics["static_friction"]))
        physics_api.CreateDynamicFrictionAttr(float(physics["dynamic_friction"]))
        physics_api.CreateRestitutionAttr(0.0)
        _apply_physx(material.GetPrim(), "PhysxMaterialAPI", {
            "physxMaterial:frictionCombineMode": Sdf.ValueTypeNames.Token,
            "physxMaterial:restitutionCombineMode": Sdf.ValueTypeNames.Token,
            "physxMaterial:dampingCombineMode": Sdf.ValueTypeNames.Token,
            "physxMaterial:improvePatchFriction": True,
        }, tokens={"physxMaterial:frictionCombineMode": "max",
                   "physxMaterial:restitutionCombineMode": "min",
                   "physxMaterial:dampingCombineMode": "max"})
        UsdShade.MaterialBindingAPI(mesh).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()


def _bbox(usd_path: Path) -> dict[str, list[float]]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        stage.GetDefaultPrim()).ComputeAlignedRange()
    return {"min": list(box.GetMin()), "max": list(box.GetMax())}


def prepare_stationery_assets(
    task_cfg: Mapping[str, Any], rebot_project: Path, out_dir: Path | None = None
) -> dict[str, Any]:
    """Write the wrappers if needed; return ``{name: usd path}`` plus ``cup`` and bboxes.

    Returns ``{"objects": {name: path}, "cup": path, "bbox": {name: {min, max}}}``.
    """
    out_dir = Path(out_dir) if out_dir else default_output_dir()
    source_usd = (Path(rebot_project) / task_cfg["stationery_usd"]).resolve()
    if not source_usd.is_file():
        raise FileNotFoundError(f"stationery asset not found: {source_usd}")
    spec = {
        "source": str(source_usd),
        "objects": task_cfg["objects"],
        "physics": task_cfg["object_physics"],
        "cup": task_cfg["cup"]["source_prim"],
        "version": 1,
    }
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    target = out_dir / digest
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    target.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"objects": {}, "bbox": {}}
    for name, item in task_cfg["objects"].items():
        path = target / f"{name}.usda"
        _write_object(path, name, source_usd, item["source_prim"], dynamic=True,
                      mass=item["mass"], physics=task_cfg["object_physics"])
        manifest["objects"][name] = str(path)
        manifest["bbox"][name] = _bbox(path)
    cup_path = target / "pencil_cup.usda"
    _write_object(cup_path, "pencil_cup", source_usd, task_cfg["cup"]["source_prim"], dynamic=False,
                  mass=None, physics=task_cfg["object_physics"])
    manifest["cup"] = str(cup_path)
    manifest["bbox"]["pencil_cup"] = _bbox(cup_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_robot_asset(robot_cfg: Mapping[str, Any], rebot_project: Path, out_dir: Path | None = None) -> str:
    """Wrapper USD for one reBot arm with the Seeed colours; returns its path.

    The PhysX asset ships one grey material. Like the single-arm scene builder,
    this de-instances the visual groups (instance descendants cannot take a
    material override) and binds ``appearance.materials`` per visual part;
    collision prototypes and the physics topology are untouched.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    out_dir = Path(out_dir) if out_dir else default_output_dir()
    source = (Path(rebot_project) / robot_cfg["usd_path"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"reBot USD not found: {source}")
    appearance = robot_cfg["appearance"]
    camera_model = robot_cfg.get("wrist_camera_model")
    digest = hashlib.sha1(json.dumps({"source": str(source), "appearance": appearance, "camera": camera_model,
                                      "version": 2}, sort_keys=True).encode()).hexdigest()[:12]
    path = out_dir / digest / "rebot_b601.usda"
    if path.is_file():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/rebot").GetPrim()
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(str(source))

    materials = {}
    for name, (color, roughness, metallic) in appearance["materials"].items():
        material = UsdShade.Material.Define(stage, f"/rebot/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/rebot/Looks/{name}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[name] = material

    bindings = appearance["bindings"]
    for group in sorted({relative.split("/visuals/", 1)[0] for relative in bindings}):
        stage.OverridePrim(f"/rebot/{group}/visuals").SetInstanceable(False)
    for relative, name in bindings.items():
        prim = stage.GetPrimAtPath(f"/rebot/{relative}")
        if not prim.IsValid():
            raise RuntimeError(f"reBot visual part missing in {source}: {relative}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[name], UsdShade.Tokens.strongerThanDescendants)
    if camera_model:
        _add_wrist_camera_model(stage, f"/rebot/{robot_cfg['end_effector_link']}", camera_model,
                                Path(rebot_project))
    stage.GetRootLayer().Save()
    return str(path)


def _add_wrist_camera_model(stage, link_path: str, model: Mapping[str, Any], rebot_project: Path) -> None:
    """Seeed UVC32 mount + 32 mm camera board and lens under the wrist link (visual only)."""
    import math

    from pxr import Gf, Sdf, UsdGeom, UsdShade

    def material(name: str, color, roughness: float, metallic: float = 0.0):
        mat = UsdShade.Material.Define(stage, f"/rebot/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/rebot/Looks/{name}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    def place(prim, translate, orient: Gf.Quatd, scale=None) -> None:
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orient)
        if scale is not None:
            xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))

    yaw = math.radians(float(model["mount_yaw_deg"]))
    yaw_q = Gf.Quatd(math.cos(yaw / 2.0), Gf.Vec3d(0.0, 0.0, math.sin(yaw / 2.0)))
    mount = UsdGeom.Xform.Define(stage, f"{link_path}/UVC32Mount").GetPrim()
    mount.GetReferences().AddReference(str((rebot_project / model["mount_usd"]).resolve()))
    place(mount, (0.0, 0.0, float(model["mount_z"])), yaw_q)
    UsdShade.MaterialBindingAPI.Apply(mount).Bind(material("UVC32Mount", (0.035, 0.038, 0.045), 0.62),
                                                  UsdShade.Tokens.strongerThanDescendants)

    pitch = math.radians(float(model["board_pitch_deg"]))
    board_q = yaw_q * Gf.Quatd(math.cos(pitch / 2.0), Gf.Vec3d(math.sin(pitch / 2.0), 0.0, 0.0))
    camera = UsdGeom.Xform.Define(stage, f"{link_path}/UVC32Camera").GetPrim()
    place(camera, model["board_center"], board_q)  # local +Z is the lens axis
    board = UsdGeom.Cube.Define(stage, f"{link_path}/UVC32Camera/Board")
    board.CreateSizeAttr(1.0)
    place(board.GetPrim(), (0.0, 0.0, 0.0), Gf.Quatd(1.0), scale=(0.032, 0.032, 0.003))
    UsdShade.MaterialBindingAPI.Apply(board.GetPrim()).Bind(material("UVC32PCB", (0.025, 0.19, 0.09), 0.52))
    for name, radius, height, offset, look in (
        ("LensHousing", 0.0065, 0.006, 0.0045, ("UVC32Housing", (0.045, 0.048, 0.055), 0.42, 0.0)),
        ("LensGlass", 0.0047, 0.001, 0.0080, ("UVC32Lens", (0.012, 0.018, 0.03), 0.12, 0.35)),
    ):
        cylinder = UsdGeom.Cylinder.Define(stage, f"{link_path}/UVC32Camera/{name}")
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)
        place(cylinder.GetPrim(), (0.0, 0.0, offset), Gf.Quatd(1.0))
        UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim()).Bind(material(*look))


def find_workstation(robot_cfg: Mapping[str, Any]) -> str:
    """YAMLab's workstation USD: $YAMLAB_HOME/yamlab/<path> or the installed ``yamlab`` package."""
    import importlib.util
    import os

    candidates = []
    if os.environ.get("YAMLAB_HOME"):
        candidates.append(Path(os.environ["YAMLAB_HOME"]).expanduser() / "yamlab")
    spec = importlib.util.find_spec("yamlab")
    if spec is not None and spec.submodule_search_locations:
        candidates.extend(Path(location) for location in spec.submodule_search_locations)
    for package in candidates:
        path = package / robot_cfg["workstation_usd"]
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        f"YAMLab workstation not found ({robot_cfg['workstation_usd']}); install yamlab or set YAMLAB_HOME")

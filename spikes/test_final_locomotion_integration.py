"""FINAL app routing, stage changes, resize, deferred setup and asset fallback."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from app import PetApp
from pet.asset_provider import SpriteRef
from pet.pet_state import PetState,PetStateStore,Stage
from render_rig_rest import RigRenderer


class FinalIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer=RigRenderer(stage='final')
        cls.win=cls.renderer.win

    @classmethod
    def tearDownClass(cls):
        cls.renderer.close()

    def setUp(self):
        self.win.stop_frames()
        self.win.disable_side_locomotion()
        self.win.set_stage('final')
        self.win.set_sprite(SpriteRef(str(ROOT/'assets/ai/final_healthy_neutral.png'),320,320))
        self.app=PetApp.__new__(PetApp)
        self.app.window=self.win
        self.app.store=PetStateStore(PetState(stage=Stage.FINAL))
        self.app.cfg={}
        self.app.logger=logging.getLogger('pet.tests')

    def test_stale_locomotion_keys_are_ignored(self):
        # v0.20.0：*_locomotion 配置已移除，侧身行走按阶段资产自动启用；旧配置残留键不生效
        self.app.cfg={'adult_locomotion':'legacy','final_locomotion':'legacy'}
        self.app._setup_side_locomotion()
        self.assertTrue(self.win.locomotion_available())
        self.assertEqual(Path(self.win._loco_pkg).name,'rig_final_walk_v1')
        self.assertAlmostEqual(self.win._loco.scale,320/1824)
        self.assertTrue(self.win._loco.clip_out.frames[0].path.endswith('000.png'))

    def test_turn_clip_set_follows_physical_pixels(self):
        # v0.20.2：选档按物理像素（逻辑高 × DPR），150% 缩放下 320 逻辑 = 480 物理取 512 档
        pkg=str(ROOT/'assets/rig_final_walk_v1')
        orig=self.win._clip_dpr
        try:
            for dpr,expect in ((1.0,'turn_front_to_side_h256'),(1.5,'turn_front_to_side'),(2.0,'turn_front_to_side')):
                self.win._clip_dpr=lambda d=dpr: d
                self.win.disable_side_locomotion()
                self.assertTrue(self.win.enable_side_locomotion(pkg))
                self.assertEqual(Path(self.win._loco.clip_out.frames[0].path).parent.parent.name,expect)
        finally:
            self.win._clip_dpr=orig
            self.win.disable_side_locomotion()

    def test_final_canvas_rejects_adult_bundle(self):
        self.assertFalse(self.win.enable_side_locomotion(str(ROOT/'assets/rig_adult_walk_v1')))
        self.assertFalse(self.win.locomotion_available())
        self.assertEqual(self.win._root.property('activeFigure'),'healthy_neutral')

    def test_stage_change_clears_session_before_loading_next_bundle(self):
        self.app._setup_side_locomotion()
        self.win.set_locomotion_intent(120)
        old=self.win._loco
        old.update(.2,120,self.win.x())
        self.win._apply_loco(old.update(.2,120,self.win.x()),None)
        self.assertEqual(self.win._root.property('locoMode'),1)
        self.win.set_stage('adult')
        self.assertFalse(self.win.locomotion_available())
        self.assertFalse(self.win._loco_carrying)
        self.assertEqual(self.win._root.property('clipFrameSrc').toString() if hasattr(self.win._root.property('clipFrameSrc'),'toString') else self.win._root.property('clipFrameSrc'),'')
        self.win.set_sprite(SpriteRef(str(ROOT/'assets/ai/adult_healthy_neutral.png'),256,256))
        self.app.store=PetStateStore(PetState(stage=Stage.ADULT))
        self.app._setup_side_locomotion()
        self.assertTrue(self.win.locomotion_available())
        self.assertEqual(Path(self.win._loco_pkg).name,'rig_adult_walk_v1')
        self.assertAlmostEqual(self.win._loco.scale,256/1696)
        self.win.set_stage('young')
        self.assertFalse(self.win.locomotion_available())
        self.assertEqual(self.win._root.property('locoMode'),0)

    def test_stage_resize_refreshes_idle_solver_scale(self):
        self.win.resize(256,256)
        self.app._setup_side_locomotion()
        self.assertAlmostEqual(self.win._loco.scale,256/1824)
        self.win.set_sprite(SpriteRef(str(ROOT/'assets/ai/final_healthy_neutral.png'),320,320))
        self.win._motion_tick()
        self.assertAlmostEqual(self.win._loco.scale,320/1824)
        self.assertIn('turn_front_to_side_h256',self.win._loco.clip_out.frames[0].path)

    def test_missing_clip_frame_restores_logical_mood(self):
        self.app._setup_side_locomotion()
        self.win.set_sprite(SpriteRef(str(ROOT/'assets/ai/final_healthy_sad.png'),320,320))
        self.win.set_locomotion_intent(120)
        with tempfile.TemporaryDirectory() as folder:
            pkg=Path(folder)
            source=ROOT/'assets/rig_final_walk_v1'
            (pkg/'spec.json').write_bytes((source/'spec.json').read_bytes())
            (pkg/'mesh').mkdir()
            (pkg/'mesh/mesh_data.json').write_text('{}')
            for name in ('turn_front_to_side','turn_side_to_front'):
                dest=pkg/'clips'/name
                dest.mkdir(parents=True)
                (dest/'clip.json').write_bytes((source/'clips'/name/'clip.json').read_bytes())
            self.assertFalse(self.win.enable_side_locomotion(str(pkg)))
        self.assertFalse(self.win._loco_carrying)
        self.assertEqual(self.win._root.property('activeFigure'),'healthy_sad')

    def test_removed_presentation_keys_not_in_defaults(self):
        from pet.config import load_config
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json'
            path.write_text(json.dumps({}))
            cfg=load_config(str(path))
            for key in ('presentation','adult_locomotion','final_locomotion','live2d','render3d'):
                self.assertNotIn(key,cfg)
            path.write_text(json.dumps({'presentation':'frames','final_locomotion':'invalid'}))
            self.assertEqual(load_config(str(path))['provider'],'ai')   # 残留键不影响加载

    def test_stage_change_cancels_deferred_package_request(self):
        from PySide6.QtCore import Qt
        from pet.rig.presenter import build_rig_window
        from pet.window import WindowBase
        win=build_rig_window(WindowBase,
            SpriteRef(str(ROOT/'assets/ai/final_healthy_neutral.png'),320,320),
            'final',defer_quick=True)
        try:
            win.setAttribute(Qt.WA_DontShowOnScreen,True)
            self.assertTrue(win.enable_side_locomotion(str(ROOT/'assets/rig_final_walk_v1')))
            self.assertTrue(win._loco_pending)
            win.set_stage('young')
            win.set_sprite(SpriteRef(str(ROOT/'assets/ai/young_healthy_neutral.png'),192,192))
            self.assertFalse(win._loco_pending)
            self.renderer._pump(6)
            win._motion_timer.stop()
            self.assertTrue(win.rig_active)
            self.assertFalse(win.locomotion_available())
            self.assertEqual(win._spec.stage,'young')
        finally:
            win.close()
            win.deleteLater()


if __name__=='__main__':
    unittest.main(verbosity=2)

import ast
from contextlib import nullcontext
import importlib.util
import json
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('eval_utils', ROOT / 'world/wan/eval_utils.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


class EvalUtilsTests(unittest.TestCase):
    def test_resume_and_invalid_results(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'log/stats_a_1.json'
            good = {'id': 'a_1', 'success': 1, 'spl': .5, 'distance_to_goal': 1.0}
            u.write_result(p, good)
            self.assertTrue(u.valid_result(p, 'a_1'))
            self.assertFalse(u.valid_result(p, 'b_1'))
            eps = [NS(episode_id='1', scene_id=f'/assets/{s}.glb') for s in ['a', 'b']]
            original = NS(episodes=eps)
            self.assertEqual(u.pending_dataset(original,d,True).episodes,[eps[1]])
            self.assertEqual(len(original.episodes),2)
            for bad in [dict(good,spl=float('nan')),dict(good,distance_to_goal=float('inf')),{'id':'a_1'}]:
                u.write_result(p,bad)
                self.assertFalse(u.valid_result(p,'a_1'))
            p.write_text('{')
            self.assertFalse(u.valid_result(p,'a_1'))
            self.assertFalse(list(p.parent.glob('*.tmp')))

    def test_ovon_collision_reset_and_resume(self):
        source=(ROOT/'infer_ovon/agent/waypoint_agent_ovon.py').read_text()
        func=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_agent_ovon')
        episodes=[NS(episode_id=str(i),scene_id='/assets/scene.glb',object_category='chair') for i in range(2)]
        class Env:
            def __init__(self,config,dataset): self.episodes=dataset.episodes; self.index=-1
            def reset(self):
                self.index+=1; self.current_episode=self.episodes[self.index]; self.episode_over=False
                rotation=NS(w=1,x=0,y=0,z=0)
                self._sim=NS(get_agent_state=lambda:NS(position=NS(tolist=lambda:[0,0,0]),rotation=rotation))
                return {}
            def get_metrics(self):
                return {'distance_to_goal':1.,'success':1.,'spl':.5,
                        'collisions':{'is_collision':self.episode_over and self.index==0}}
            def step(self,action): self.episode_over=True; return {}
        class Agent:
            def __init__(self,*args,**kwargs): pass
            def reset(self): pass
            def act(self,*args): return {'arrive_pred':1}
        import os
        scope=dict(Env=Env,get_model_name_from_path=lambda x:'model',os=os,
                   pending_dataset=u.pending_dataset,episode_key=u.episode_key,
                   random=random,write_result=u.write_result,
                   Waypoint_Agent=Agent,trange=lambda n,**kwargs:range(n),
                   torch=NS(no_grad=nullcontext),MODEL_TYPE='Waypoint',obj_goal_template=['Find {}'])
        exec(compile(ast.Module(body=[func],type_ignores=[]),'<evaluate_agent_ovon>','exec'),scope)
        with tempfile.TemporaryDirectory() as d:
            config=NS(EVAL=NS(EARLY_STOP_ROTATION=25,EARLY_STOP_STEPS=400,IDENTIFICATION='test'))
            scope['evaluate_agent_ovon'](config,0,NS(episodes=episodes),'checkpoint',d)
            logs=Path(d)/'model/log'
            self.assertTrue(json.loads((logs/'stats_scene_0.json').read_text())['is_collision'])
            self.assertFalse(json.loads((logs/'stats_scene_1.json').read_text())['is_collision'])
            self.assertEqual([e.episode_id for e in episodes],['0','1'])
            scope['Env']=lambda *args:self.fail('Completed run must not construct Env')
            scope['evaluate_agent_ovon'](config,0,NS(episodes=episodes),'checkpoint',d)

    def test_future_video_round_trip(self):
        try:
            import numpy as np
            import imageio.v2 as imageio
            import imageio_ffmpeg
        except ImportError:
            self.skipTest('Video runtime dependencies not installed')
        with tempfile.TemporaryDirectory() as d:
            video=np.zeros((1,5,32,48,3),dtype=np.float32)
            video[:,1:]=.8
            p=u.save_future_video(video,d,'scene_1',0)
            frames=imageio.mimread(p)
            self.assertEqual(len(frames),5)
            self.assertEqual(frames[0].shape,(32,48,3))
            with self.assertRaises(ValueError):u.save_future_video(video*float('nan'),d,'scene_1',1)


if __name__=='__main__': unittest.main()

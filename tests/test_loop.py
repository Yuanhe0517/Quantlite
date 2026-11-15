import json
import tempfile
import unittest
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from lite.expressions import calculate, parse, ema, ExpressionError
from lite.loop import run_loop
from lite.workflow import ROOT
from test_workflow import HYPOTHESIS, FACTORS

class ExpressionTests(unittest.TestCase):
    def test_security_and_lookahead_rejected(self):
        for expr in ['__import__("os")','$close.__class__','DELAY($close,-1)','TS_MEAN($close,300)','FOO($close)','$close ** 100','1']:
            with self.subTest(expr=expr), self.assertRaises(ExpressionError):parse(expr,['$close'])

    def test_numerics_rank_causality_and_zero(self):
        idx=pd.MultiIndex.from_product([['A','B'],pd.date_range('2020-01-01',periods=5)],names=['instrument','datetime'])
        frame=pd.DataFrame({'$close':[1,2,3,4,5,5,4,3,2,1]},index=idx,dtype=float)
        member=pd.Series(True,index=idx)
        result=calculate('RANK(-TS_MEAN($close, 2))',frame,member)
        np.testing.assert_allclose(result.xs('A').values,[np.nan,1,1,.5,.5],equal_nan=True)
        prefix=frame[frame.index.get_level_values('datetime')<=pd.Timestamp('2020-01-03')]
        pd.testing.assert_series_equal(calculate('RANK(-TS_MEAN($close, 2))',prefix,member.reindex(prefix.index)),result.reindex(prefix.index))
        self.assertTrue(calculate('$close / 0',frame,member).isna().all())

    def test_ema_recursion_and_reset(self):
        source=pd.Series([1.,2,3,np.nan,4,5,6])
        np.testing.assert_allclose(ema(source,3),[np.nan,np.nan,2.25,np.nan,np.nan,np.nan,5.25],equal_nan=True)

class LoopTests(unittest.TestCase):
    def test_four_rounds_full_history_output_and_no_fifth_call(self):
        calls=[];logs=[]
        class FakeEvaluator:
            def __init__(self,config):pass
            def evaluate(self,proposal,folder,number,emit):
                result={'round':number,'research_direction':'test','hypothesis':proposal['hypothesis'],
                    'protocol':{},'factors':[{'name':n,**f,'status':'completed','annual':{'2020':{'mean_rank_ic':number/100}}} for n,f in proposal['factors'].items()],
                    'mean_pairwise_factor_rank_correlation':.2}
                (folder/'evaluation.json').write_text(json.dumps(result))
                emit('Rank IC synthetic test result')
                return result
        def generate(config,messages):
            calls.append(messages)
            round_number=(len(calls)+1)//2
            if round_number>1:
                text=json.dumps(messages)
                for earlier in range(1,round_number):self.assertIn(f'"round": {earlier}',text.replace('\\"','"'))
            return json.dumps(HYPOTHESIS if len(calls)%2 else FACTORS),{}
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            cfg=yaml.safe_load((ROOT/'configs/research.yaml').read_text())
            cfg['loop']['max_rounds']=4
            config_path=run/'test_config.yaml'
            config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            run_loop(config_path,run,generate=generate,evaluator_factory=FakeEvaluator,emit=logs.append)
            self.assertEqual(len(calls),8)
            self.assertEqual(len(json.loads((run/'history.json').read_text())),4)
            self.assertEqual(json.loads((run/'status.json').read_text())['status'],'completed')
            self.assertEqual(len(list(run.glob('round_*'))),4)
            self.assertTrue(any('Rank IC' in line for line in logs))
            self.assertTrue(any(next(iter(FACTORS.values()))['expression'] in line for line in logs))

    def test_dry_run_neither_model_nor_data(self):
        def forbidden(*args):self.fail('Must not call API or evaluator')
        with tempfile.TemporaryDirectory() as d:
            run_loop(ROOT/'configs/research.yaml',Path(d),generate=forbidden,evaluator_factory=forbidden,emit=lambda _:None,dry_run=True)

    def test_wrong_factor_count_stops_before_evaluation(self):
        calls=[]
        extra={**FACTORS,'extra':next(iter(FACTORS.values()))}
        class FakeEvaluator:
            def __init__(self,c):pass
            def evaluate(self,*args):raise AssertionError('Must not evaluate three factors')
        def generate(*args):
            calls.append(1)
            return json.dumps(HYPOTHESIS if len(calls)%2 else extra),{}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                run_loop(ROOT/'configs/research.yaml',Path(d),generate=generate,evaluator_factory=FakeEvaluator,emit=lambda _:None)
            self.assertEqual(len(calls),2)
            self.assertEqual(json.loads((Path(d)/'status.json').read_text())['status'],'failed')

class HistoryWindowTests(unittest.TestCase):
    def test_both_prompts_get_only_latest_six_and_history_stays_intact(self):
        from lite.workflow import build_graph
        history=[{'round':i,'marker':f'UNIQUE_ROUND_{i:02}'} for i in range(1,10)]
        saved={};calls=[]
        def generate(config,messages):
            calls.append(messages)
            content=json.dumps(messages,ensure_ascii=False)
            for i in range(1,4): self.assertNotIn(f'UNIQUE_ROUND_{i:02}',content)
            for i in range(4,10): self.assertIn(f'UNIQUE_ROUND_{i:02}',content)
            return json.dumps(HYPOTHESIS if len(calls)==1 else FACTORS),{}
        build_graph(generate,saved.__setitem__).invoke({'config_path':str(ROOT/'configs/research.yaml'),'loop_mode':True,'history':history})
        self.assertEqual(len(calls),2)
        self.assertEqual([x['round'] for x in saved['feedback.json']['all_completed_rounds']],list(range(4,10)))
        self.assertEqual(len(history),9)

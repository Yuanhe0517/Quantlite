import json
import tempfile
import unittest
import yaml
from pathlib import Path
from lite.loop import run_loop
from lite.workflow import ROOT
from test_workflow import HYPOTHESIS, FACTORS

class ResumeTests(unittest.TestCase):
    def test_completed_round_and_cached_responses_reused(self):
        calls=[];evaluated=[];fail=[True]
        class FakeEvaluator:
            def __init__(self,c):pass
            def evaluate(self,proposal,folder,number,emit):
                if number==2 and fail[0]:raise RuntimeError('simulated interruption')
                evaluated.append(number)
                value={'status':'completed','round':number,'research_direction':'test','hypothesis':proposal['hypothesis'],
                       'protocol':{},'factors':[{'name':n,**f,'status':'completed'} for n,f in proposal['factors'].items()],
                       'mean_pairwise_factor_rank_correlation':.1}
                (folder/'evaluation.json').write_text(json.dumps(value))
                (folder/'daily_metrics.csv').write_text('mock')
                (folder/'factor_values.parquet').write_text('mock')
                return value
        def generate(*args):
            calls.append(1)
            return json.dumps(HYPOTHESIS if len(calls)%2 else FACTORS),{}
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            cfg=yaml.safe_load((ROOT/'configs/research.yaml').read_text())
            cfg['loop']['max_rounds']=4
            config_path=run/'test_config.yaml'
            config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
            with self.assertRaises(RuntimeError):run_loop(config_path,run,generate=generate,evaluator_factory=FakeEvaluator,emit=lambda _:None)
            self.assertEqual(len(calls),4)
            fail[0]=False
            logs=[]
            run_loop(config_path,run,generate=generate,evaluator_factory=FakeEvaluator,emit=logs.append,resume=True)
            self.assertEqual(len(calls),8)
            self.assertEqual(evaluated,[1,2,3,4])
            self.assertEqual(sum('不调用 API' in x for x in logs),2)
            self.assertEqual(len(json.loads((run/'history.json').read_text())),4)
            run_loop(config_path,run,generate=lambda *a:self.fail('No further API'),evaluator_factory=lambda c:self.fail('No further data'),emit=lambda _:None,resume=True)

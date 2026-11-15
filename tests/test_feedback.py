import json
from pathlib import Path
import tempfile
import unittest
from lite.workflow import ROOT, build_graph
from test_workflow import HYPOTHESIS, FACTORS


def evaluation():
    protocol={k:'test' for k in ['volume','rank','metrics','missing_values','annual_grouping']}
    protocol.update(market='historical_csi300',horizon_trading_days=21,
                    actual_signal_period=['2020-01-02','2023-11-29'], latest_label_exit='2023-12-29',
                    label='close[T+22]/close[T+1]-1',future_backtest_result='DO_NOT_SEND')
    return {'status':'completed','stage':'exploration','protocol':protocol,
            'hypothesis':HYPOTHESIS,'proposal_run_id':'test_parent','research_direction':'test',
            'factors':[{'name':k,**v,'expected_direction':'positive',
                       'overall':{'mean_rank_ic':-0.039},'annual':{'2023':{'mean_rank_ic':-0.095}}}
                       for k,v in FACTORS.items()], 'mean_pairwise_factor_rank_correlation':0.5}


class FeedbackTests(unittest.TestCase):
    def test_feedback_reaches_both_stages_and_saved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evaluation.json';p.write_text(json.dumps(evaluation()))
            calls=[];saved={}
            def generate(config,messages):
                calls.append(messages)
                text=json.dumps(messages,ensure_ascii=False)
                self.assertIn('-0.039',text)
                self.assertNotIn('DO_NOT_SEND',text)
                self.assertIn('revision round',messages[0]['content'])
                return json.dumps(HYPOTHESIS if len(calls)==1 else FACTORS),{}
            state=build_graph(generate,saved.__setitem__).invoke({'config_path':str(ROOT/'configs/research.yaml'),'evaluation_path':str(p)})
            self.assertEqual(len(calls),2)
            self.assertEqual(state['factors'],FACTORS)
            self.assertIn('source_evaluation_sha256',saved['feedback.json'])

    def test_mismatched_target_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            data=evaluation();data['protocol']['horizon_trading_days']=1
            p=Path(d)/'evaluation.json';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'differs'):
                build_graph(lambda *a:self.fail('API called')).invoke({'config_path':str(ROOT/'configs/research.yaml'),'evaluation_path':str(p)})

    def test_non_exploration_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            data=evaluation();data['stage']='holdout'
            p=Path(d)/'evaluation.json';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'exploration'):
                build_graph(lambda *a:self.fail('API called')).invoke({'config_path':str(ROOT/'configs/research.yaml'),'evaluation_path':str(p)})

class HypothesisCompatibilityTests(unittest.TestCase):
    def test_extra_explanations_do_not_discard_valid_hypothesis(self):
        from lite.workflow import parse_hypothesis
        data={**HYPOTHESIS,'reasoning':'extra','distilled_knowledge':'extra'}
        self.assertEqual(parse_hypothesis(json.dumps(data)),HYPOTHESIS)

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx
import yaml
from openai import OpenAI
from pydantic import ValidationError
from lite.models import Config
from lite.workflow import ROOT, build_graph, openai_generate, parse_json

HYPOTHESIS = {'hypothesis':'唯一标记：放量冲击后的价格反转',
              'concise_knowledge':'若冲击短暂，价格可能回归',
              'concise_observation':'未提供实测结果',
              'concise_justification':'暂时性买卖压力的假说',
              'concise_specification':'沪深300，未来21个交易日'}
FACTORS = {f'test_{n}': {'description':'离线测试候选', 'formulation':'1-C_t/C_{t-n}',
           'expression':f'1-$close/DELAY($close,{n})',
           'variables':{'$close':'复权收盘价','DELAY(A,n)':'滞后'}} for n in [5,10]}


class WorkflowTests(unittest.TestCase):
    def test_two_stage_dependency_and_artifacts(self):
        calls, artifacts = [], {}
        def generate(c, messages):
            calls.append(messages)
            if len(calls)==1:
                self.assertIn(c.research.direction, messages[1]['content'])
                self.assertIn('close[T+22]/close[T+1]-1', messages[0]['content'])
                return json.dumps(HYPOTHESIS), {'stage':'hypothesis'}
            self.assertIn(HYPOTHESIS['hypothesis'], messages[1]['content'])
            self.assertIn('EMA(A, n)', messages[1]['content'])
            self.assertIn('250 characters', messages[0]['content'])
            return json.dumps(FACTORS), {'stage':'factors'}
        result=build_graph(generate, artifacts.__setitem__).invoke({'config_path':str(ROOT/'configs/research.yaml')})
        self.assertEqual(len(calls),2)
        self.assertEqual(result['proposal'],{'hypothesis':HYPOTHESIS,'factors':FACTORS})
        self.assertEqual(artifacts['hypothesis.json'],HYPOTHESIS)
        self.assertEqual(artifacts['factors.json'],FACTORS)

    def test_original_prompt_values_unchanged(self):
        original=ROOT.parent/'QuantaAlpha/quantaalpha/factors/prompts/prompts.yaml'
        if not original.exists(): self.skipTest('Original checkout not present')
        source=yaml.safe_load(original.read_text())
        copied=yaml.safe_load((ROOT/'prompts/original.yaml').read_text())
        for key,value in copied.items(): self.assertEqual(value,source[key],key)

    def test_dry_run_never_calls_provider(self):
        artifacts={}
        result=build_graph(lambda *a:self.fail('API called'), artifacts.__setitem__,dry_run=True).invoke(
            {'config_path':str(ROOT/'configs/research.yaml')})
        self.assertNotIn('hypothesis',result)
        self.assertIn('factor_messages.preview.json',artifacts)
        self.assertNotIn('factor_messages.json',artifacts)

    def test_bad_config_fails_before_api(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.yaml'; p.write_text('research:\n  direction: test\n  horizon_days: -1\n')
            with self.assertRaises(ValidationError):
                build_graph(lambda *a:self.fail('API called')).invoke({'config_path':str(p)})

    def test_invalid_hypothesis_blocks_factor_call(self):
        calls=[]; artifacts={}
        def generate(*args): calls.append(1); return '{}',{}
        with self.assertRaises(ValidationError):
            build_graph(generate,artifacts.__setitem__).invoke({'config_path':str(ROOT/'configs/research.yaml')})
        self.assertEqual(len(calls),1)
        self.assertIn('hypothesis_response.json',artifacts)
        self.assertNotIn('factor_messages.json',artifacts)

    def test_invalid_factor_response_preserves_hypothesis(self):
        calls=[]; artifacts={}
        def generate(*args):
            calls.append(1)
            return (json.dumps(HYPOTHESIS) if len(calls)==1 else '{}'),{}
        with self.assertRaises(ValueError):
            build_graph(generate,artifacts.__setitem__).invoke({'config_path':str(ROOT/'configs/research.yaml')})
        self.assertEqual(len(calls),2)
        self.assertEqual(artifacts['hypothesis.json'],HYPOTHESIS)
        self.assertIn('factor_response.json',artifacts)
        self.assertNotIn('factors.json',artifacts)

    def test_json_parser_rejects_duplicate_names(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            parse_json('{"factor":{},"factor":{}}')
        self.assertEqual(parse_json('```json\n{"a":1}\n```'),{'a':1})

    def test_sdk_json_response_without_network(self):
        config=Config.model_validate(yaml.safe_load((ROOT/'configs/research.yaml').read_text()))
        calls=[]
        def handle(request):
            body=json.loads(request.content); calls.append(body)
            self.assertEqual(body['text']['format']['type'],'json_object')
            self.assertEqual(body['reasoning']['effort'],'low')
            self.assertFalse(body['store'])
            return httpx.Response(200,json={'id':'resp_offline','object':'response','created_at':0,
                'model':config.llm.model,'status':'completed',
                'output':[{'type':'message','id':'msg_offline','role':'assistant','status':'completed',
                    'content':[{'type':'output_text','text':json.dumps(HYPOTHESIS),'annotations':[]}]}]})
        client=OpenAI(api_key='offline-only',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        with patch.dict(os.environ,{'OPENAI_API_KEY':'offline-only'}),patch('openai.OpenAI',return_value=client):
            text,metadata=openai_generate(config,[{'role':'user','content':'JSON test'}])
        self.assertEqual(parse_json(text),HYPOTHESIS)
        self.assertEqual(len(calls),1)


if __name__=='__main__': unittest.main()

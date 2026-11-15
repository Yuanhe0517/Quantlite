import json
import hashlib
from pathlib import Path
from typing import TypedDict, Callable
import yaml
from jinja2 import Environment, StrictUndefined
from langgraph.graph import StateGraph, START, END
from .models import Config, Hypothesis, validate_factors

ROOT = Path(__file__).resolve().parents[1]
ENV = Environment(undefined=StrictUndefined)
LOOP_HISTORY_LIMIT = 6


class State(TypedDict, total=False):
    config_path: str
    loop_mode: bool
    history: list[dict]
    evaluation_path: str
    feedback: dict
    config: dict
    prompts: dict
    hypothesis_messages: list[dict]
    hypothesis: dict
    factors: dict
    proposal: dict


def render(template, **context):
    return ENV.from_string(template).render(**context)


def scenario(config: Config, feedback=None):
    r = config.research
    return (f'Daily A-share factor research; universe: {r.market}. Available fields: {", ".join(r.fields)}. '
            'Prices use consistent adjustment; $return, when available, means close[T]/close[T-1]-1. '
            f'The research target is close[T+{r.horizon_days+1}]/close[T+1]-1 in trading days, '
            'with signals available at the close of T. Only contemporaneous and past data may be features. '
            + ('This is a revision round using supplied exploration results. ' if feedback else 'This is the first research round; no empirical results or market data are supplied. ')
            + 'Do not invent observed statistics. The field list here takes precedence over generic examples.')


def hypothesis_messages(config, prompts, feedback=None):
    direction = render(prompts['potential_direction_transformation'], potential_direction=config.research.direction)
    return [
        {'role':'system', 'content':render(prompts['hypothesis_gen']['system_prompt'],
            targets='factors', scenario=scenario(config, feedback),
            hypothesis_output_format=prompts['hypothesis_output_format'],
            hypothesis_specification=prompts['factor_hypothesis_specification'])},
        {'role':'user', 'content':render(prompts['hypothesis_gen']['user_prompt'],
            targets='factors', hypothesis_and_feedback=feedback_text(feedback) if feedback else direction, RAG=None, round=1 if feedback else 0)},
    ]


def factor_messages(config, prompts, hypothesis, feedback=None):
    target = hypothesis if isinstance(hypothesis, str) else '\n'.join(f'{k}: {v}' for k,v in hypothesis.items())
    return [
        {'role':'system', 'content':render(prompts['hypothesis2experiment']['system_prompt'],
            targets='factors', scenario=scenario(config, feedback), experiment_output_format=prompts['factor_experiment_output_format'])},
        {'role':'user', 'content':render(prompts['hypothesis2experiment']['user_prompt'],
            targets='factors', target_hypothesis=target,
            hypothesis_and_feedback=feedback_text(feedback) if feedback else "No previous hypothesis and feedback available since it's the first round.",
            function_lib_description=prompts['function_lib_description'], expression_duplication=None,
            target_list=[], RAG=None)},
    ]



def load_feedback(path, config):
    source = Path(path)
    raw = source.read_bytes()
    data = json.loads(raw)
    if data.get('status') != 'completed' or data.get('stage') != 'exploration':
        raise ValueError('Feedback must be a completed exploration evaluation')
    protocol = data['protocol']
    expected_market = {'csi300':'historical_csi300', 'all_a':'all_a'}[config.research.market]
    if protocol['market'] != expected_market or protocol['horizon_trading_days'] != config.research.horizon_days:
        raise ValueError('Evaluation market or horizon differs from research config')
    Hypothesis.model_validate(data['hypothesis'])
    if not data.get('factors'):
        raise ValueError('No evaluated factors')
    # Whitelist material for the agent; no future-backtest fields, paths or unrelated results.
    return {
        'source_run_id': data['proposal_run_id'],
        'source_evaluation_sha256': hashlib.sha256(raw).hexdigest(),
        'research_direction': data['research_direction'],
        'previous_hypothesis': data['hypothesis'],
        'protocol': {k:protocol[k] for k in [
            'market','actual_signal_period','latest_label_exit','horizon_trading_days','label',
            'volume','rank','metrics','missing_values','annual_grouping']},
        'factors': [{k:f[k] for k in ['name','description','expression','formulation',
                    'expected_direction','overall','annual']} for f in data['factors']],
        'mean_pairwise_factor_rank_correlation': data['mean_pairwise_factor_rank_correlation'],
    }


def feedback_text(feedback):
    return ('Previous hypothesis, exact expressions and exploration evaluation:\n'
            + json.dumps(feedback, ensure_ascii=False, indent=2)
            + '\nRevise the hypothesis using only these results. In concise_observation summarize the evidence; '
            'in concise_justification explain what changed and why. Separate evidence from causal speculation. '
            'In concise_specification identify testable changes and comparisons against the previous factors. '
            'Do not treat a post-hoc sign reversal as independent validation. '
            'Overlapping 21-day returns are not independent observations. '
            'No price-only/volume-only controls or causal tests have been performed. '
            'Generate 2-3 factors for the revised hypothesis; do not invent test results for them.')

def parse_json(text):
    # Accept JSON with optional outer code fence, but reject duplicate keys.
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def parse_hypothesis(text):
    data = parse_json(text)
    if not isinstance(data, dict):
        raise ValueError('Hypothesis must be a JSON object')
    # Original QuantaAlpha extracts named fields; its prompt also asks for reasoning.
    # Keep the entire response on disk, but use only the defined hypothesis fields.
    return Hypothesis.model_validate({k:v for k,v in data.items() if k in Hypothesis.model_fields}).model_dump()


def openai_generate(config: Config, messages: list[dict]):
    import os
    from openai import OpenAI
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('OPENAI_API_KEY is missing; use --env-file or environment variable')
    with OpenAI(api_key=key, base_url='https://api.openai.com/v1',
                max_retries=0, timeout=config.llm.timeout_seconds) as client:
        response = client.responses.create(
            model=config.llm.model, input=messages,
            text={'format': {'type': 'json_object'}},
            reasoning={'effort': config.llm.reasoning_effort},
            max_output_tokens=config.llm.max_output_tokens, store=False)
    metadata = {'response_id': response.id, 'model': response.model,
                'status': response.status,
                'usage': response.usage.model_dump() if response.usage else None}
    if response.status != 'completed' or not response.output_text:
        raise RuntimeError(f'No completed JSON output; status={response.status}, response_id={response.id}')
    return response.output_text, metadata


def build_graph(generate: Callable = openai_generate, persist: Callable | None = None,
                dry_run: bool = False):
    def save(name, data):
        if persist:
            persist(name, data)

    def load(state):
        config = Config.model_validate(yaml.safe_load(Path(state['config_path']).read_text()))
        prompts = yaml.safe_load((ROOT/'prompts/original.yaml').read_text())
        feedback = load_feedback(state['evaluation_path'], config) if state.get('evaluation_path') else None
        if state.get('history'):
            feedback = {'research_direction': config.research.direction, 'all_completed_rounds': state['history'][-LOOP_HISTORY_LIMIT:]}
        if state.get('loop_mode'):
            from .expressions import GUIDE
            prompts = dict(prompts)
            prompts['function_lib_description'] = GUIDE
        if feedback:
            save('feedback.json', feedback)
        messages = hypothesis_messages(config, prompts, feedback)
        if state.get('loop_mode'):
            from .expressions import GUIDE
            messages[0]['content'] += '\n' + GUIDE + '\nPlan exactly two factors. Use the supplied recent rounds; older records remain local and are not supplied.'
        save('config.json', config.model_dump())
        save('prompt_snapshot.json', prompts)
        save('hypothesis_messages.json', messages)
        if dry_run:
            save('factor_messages.preview.json', factor_messages(config, prompts,
                '<PREVIEW ONLY: the actual generated hypothesis will be inserted here>', feedback))
        return {'config': config.model_dump(), 'prompts': prompts, 'hypothesis_messages': messages, 'feedback': feedback}

    def hypothesize(state):
        text, metadata = generate(Config.model_validate(state['config']), state['hypothesis_messages'])
        save('hypothesis_response.json', {'text': text, 'metadata': metadata})
        hypothesis = parse_hypothesis(text)
        save('hypothesis.json', hypothesis)
        return {'hypothesis': hypothesis}

    def factors(state):
        config = Config.model_validate(state['config'])
        messages = factor_messages(config, state['prompts'], state['hypothesis'], state.get('feedback'))
        if state.get('loop_mode'):
            from .expressions import GUIDE
            messages[0]['content'] += '\n' + GUIDE
        save('factor_messages.json', messages)
        text, metadata = generate(config, messages)
        save('factor_response.json', {'text': text, 'metadata': metadata})
        values = validate_factors(parse_json(text))
        if state.get('loop_mode') and len(values) != 2:
            raise ValueError('Loop requires exactly two factors')
        save('factors.json', values)
        return {'factors': values, 'proposal': {'hypothesis': state['hypothesis'], 'factors': values}}

    graph = StateGraph(State)
    graph.add_node('read_research_config', load)
    graph.add_node('generate_hypothesis', hypothesize)
    graph.add_node('generate_factors', factors)
    graph.add_edge(START, 'read_research_config')
    graph.add_edge('read_research_config', END if dry_run else 'generate_hypothesis')
    graph.add_edge('generate_hypothesis', 'generate_factors')
    graph.add_edge('generate_factors', END)
    return graph.compile()

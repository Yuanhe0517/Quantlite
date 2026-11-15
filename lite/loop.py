"""Bounded LangGraph research loop, using local evaluation and complete history."""
import json
import hashlib
from pathlib import Path
from typing import TypedDict
import yaml
from langgraph.graph import StateGraph, START, END
from .models import Config
from .workflow import build_graph, openai_generate, load_feedback

class LoopState(TypedDict, total=False):
    round: int
    history: list[dict]
    proposal: dict


def feedback_record(evaluation):
    """Only exploration evidence goes back to the model, never artifact paths."""
    return {k:evaluation[k] for k in ['round','research_direction','hypothesis','protocol','factors','mean_pairwise_factor_rank_correlation']}


def run_loop(config_path,run,evaluation_path=None,generate=openai_generate,evaluator_factory=None,emit=None,dry_run=False,resume=False):
    if resume:
        if dry_run or evaluation_path:
            raise ValueError('Resume cannot be combined with dry-run or seed evaluation')
        config_path=run/'config.json'
        if not config_path.is_file(): raise ValueError('Missing saved run config')
    config=Config.model_validate(yaml.safe_load(Path(config_path).read_text()))
    if config.research.market!='csi300' or config.research.horizon_days!=21:
        raise ValueError('Loop requires historical csi300 and 21-day horizon')
    emit=emit or (lambda text:print(text,flush=True))
    def save(path,value):
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if not resume: save(run/'config.json',config.model_dump())
    history=[]
    if evaluation_path:
        seed=load_feedback(evaluation_path,config)
        protocol=seed['protocol']
        if protocol['actual_signal_period'][0]<'2020-01-01' or protocol['actual_signal_period'][1]>'2023-12-31' or protocol['latest_label_exit']>'2023-12-31':
            raise ValueError('Seed feedback must stay within 2020-2023 exploration data')
        history.append(seed)
    completed=0
    if resume:
        first=run/'round_01/feedback.json'
        if first.exists():
            seed=json.loads(first.read_text())
            history=list(seed.get('all_completed_rounds',[seed]))
        for number in range(1,config.loop.max_rounds+1):
            folder=run/f'round_{number:02d}'
            path=folder/'evaluation.json'
            # Evaluation is committed only when all accompanying data artifacts exist.
            if not all((folder/n).is_file() for n in ['evaluation.json','daily_metrics.csv','factor_values.parquet']): break
            evaluation=json.loads(path.read_text())
            if evaluation.get('status')!='completed' or evaluation.get('round')!=number:
                raise ValueError('Invalid saved evaluation checkpoint')
            record=feedback_record(evaluation)
            record['source_evaluation_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            history.append(record);completed=number
        save(run/'history.json',history)
        if completed>=config.loop.max_rounds:
            emit('全部轮次已经完成，无需继续。')
            save(run/'status.json',{'status':'completed','completed_rounds':completed,'max_rounds':config.loop.max_rounds})
            return
        emit(f'断点续跑：保留前 {completed} 轮，从第 {completed+1} 轮继续。')
    if dry_run:
        folder=run/'round_01';folder.mkdir()
        build_graph(generate=generate,persist=lambda n,v:save(folder/n,v),dry_run=True).invoke(
            {'config_path':str(config_path),'loop_mode':True,'history':history})
        save(run/'status.json',{'status':'dry_run','max_rounds':config.loop.max_rounds,'api_calls':0,'market_data_loaded':False})
        emit('Loop 预览已保存；未调用 API、未加载行情或评估。')
        return
    if evaluator_factory is None:
        from .evaluation import Evaluator
        evaluator_factory=Evaluator
    emit('准备本地 2020—2023 年探索数据（只读取一次）…')
    evaluator=evaluator_factory(config)  # Fail before any model call if local data/config is invalid.
    if resume and hasattr(evaluator,'seen'):
        from .expressions import canonical, ExpressionError
        for record in history:
            for factor in record.get('factors',[]):
                try: evaluator.seen[canonical(factor['expression'],config.research.fields)]={'round':record.get('round'),'factor':factor['name']}
                except ExpressionError: pass
    save(run/'status.json',{'status':'running','completed_rounds':completed,'max_rounds':config.loop.max_rounds})
    def propose(state):
        number=state['round']+1
        folder=run/f'round_{number:02d}';folder.mkdir(exist_ok=resume)
        emit(f'\n[{number}/{config.loop.max_rounds}] 生成假设与 2 个因子…')
        save(run/'status.json',{'status':'running','phase':'proposal','active_round':number,'completed_rounds':number-1})
        cached={}
        if resume:
            for stage in ['hypothesis','factor']:
                response=folder/f'{stage}_response.json'; messages=folder/f'{stage}_messages.json'
                if response.exists():
                    if not messages.exists(): raise ValueError('Cached response has no prompt')
                    cached[stage]=(json.loads(messages.read_text()),json.loads(response.read_text()))
        stages=iter(['hypothesis','factor'])
        def generate_or_reuse(cfg,messages):
            stage=next(stages)
            if stage in cached:
                previous,response=cached[stage]
                if previous!=messages: raise ValueError('Saved prompt differs; refusing to reuse a mismatched response')
                emit(f'  复用已保存的 {stage} 响应（不调用 API）')
                return response['text'],response['metadata']
            return generate(cfg,messages)
        def persist(name,value):
            path=folder/name
            if resume and path.exists() and name.endswith('_messages.json'):
                if json.loads(path.read_text())!=value: raise ValueError('Saved prompt differs from resume prompt')
            save(path,value)
        result=build_graph(generate=generate_or_reuse,persist=persist).invoke(
            {'config_path':str(config_path),'loop_mode':True,'history':state['history']})
        proposal=result['proposal'];save(folder/'proposal.json',proposal)
        for name,factor in proposal['factors'].items(): emit(f"  {name}\n    {factor['expression']}")
        return {'round':number,'proposal':proposal}
    def evaluate(state):
        number=state['round'];folder=run/f'round_{number:02d}'
        emit(f'[{number}/{config.loop.max_rounds}] 本地计算及年度 Rank IC 评估…')
        save(run/'status.json',{'status':'running','phase':'evaluation','active_round':number,'completed_rounds':number-1})
        evaluation=evaluator.evaluate(state['proposal'],folder,number,emit)
        record=feedback_record(evaluation)
        record['source_evaluation_sha256']=hashlib.sha256((folder/'evaluation.json').read_bytes()).hexdigest()
        updated=[*state['history'],record]
        save(run/'history.json',updated)
        save(folder/'status.json',{'status':'evaluated','successful_factors':sum(f['status']=='completed' for f in evaluation['factors'])})
        return {'history':updated}
    graph=StateGraph(LoopState)
    graph.add_node('propose',propose);graph.add_node('evaluate',evaluate)
    graph.add_edge(START,'propose');graph.add_edge('propose','evaluate')
    graph.add_conditional_edges('evaluate',lambda state:END if state['round']>=config.loop.max_rounds else 'propose')
    try:
        result=graph.compile().invoke({'round':completed,'history':history},config={'recursion_limit':2*config.loop.max_rounds+5})
    except Exception as error:
        status=json.loads((run/'status.json').read_text())
        save(run/'status.json',{**status,'status':'failed','error_type':type(error).__name__})
        raise
    save(run/'status.json',{'status':'completed','completed_rounds':result['round'],'max_rounds':config.loop.max_rounds})
    emit(f"\n完成 {result['round']} 轮，结果保存至 {run}")

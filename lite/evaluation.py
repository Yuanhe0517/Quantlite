"""Local exploration evaluator; load once and reuse across research rounds."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .expressions import calculate, canonical, ExpressionError
from .workflow import ROOT


def aggregate(g):
    def number(x): return float(x) if pd.notna(x) and np.isfinite(x) else None
    valid=g.rank_ic.dropna()
    return {'mean_rank_ic':number(valid.mean()),'mean_ic':number(g.ic.mean()),
            'mean_coverage':number(g.coverage.mean()),'signal_days':len(g),
            'valid_rank_ic_days':len(valid),'paired_stock_days':int(g.n.sum()),
            'rank_ic_positive_day_fraction':number((valid>0).mean())}


class Evaluator:
    def __init__(self, config):
        self.config=config
        if config.research.market!='csi300' or config.research.horizon_days!=21:
            raise ValueError('Loop evaluator currently requires csi300 and 21-day horizon')
        import qlib
        from qlib.data import D
        provider=Path(config.loop.provider_uri).expanduser()
        if not provider.is_absolute(): provider=ROOT/provider
        provider=provider.resolve()
        if not (provider/'calendars/day.txt').is_file(): raise ValueError('Local Qlib provider missing')
        qlib.init(provider_uri=str(provider),region='cn',kernels=1)
        calendar=pd.DatetimeIndex(D.calendar())
        if calendar.min()>pd.Timestamp(config.loop.warmup_start) or calendar.max()<pd.Timestamp('2023-12-29'):
            raise ValueError('Local calendar does not cover configured exploration period')
        last=calendar.searchsorted(pd.Timestamp(config.loop.end),side='right')-1
        end=calendar[last-22];exit_date=calendar[last]
        market=D.instruments('csi300')
        membership=D.features(market,['$close'],start_time=config.loop.warmup_start,end_time=str(end.date())).index
        ids=membership.get_level_values('instrument').unique().tolist()
        dates=calendar[(calendar>=config.loop.warmup_start)&(calendar<=end)]
        grid=pd.MultiIndex.from_product([ids,dates],names=['instrument','datetime'])
        fields=list(dict.fromkeys([f for f in config.research.fields if f!='$return']+['$close']))
        data=D.features(ids,fields,start_time=config.loop.warmup_start,end_time=str(end.date())).astype('float64').reindex(grid).sort_index()
        for f in fields:
            data[f]=data[f].where(data[f]>=0 if f=='$volume' else data[f]>0)
            if not data[f].notna().any(): raise ValueError('Configured market field has no data: '+f)
        if '$return' in config.research.fields:
            data['$return']=data['$close']/data['$close'].groupby(level='instrument').shift(1)-1
        self.member=pd.Series(data.index.isin(membership),index=data.index)
        self.frame=data[config.research.fields].where(self.member,axis=0)
        self.label=D.features(market,['Ref($close,-22)/Ref($close,-1)-1'],start_time=config.loop.start,end_time=str(end.date())).iloc[:,0].replace([np.inf,-np.inf],np.nan).rename('forward_return')
        if self.label.empty: raise ValueError('No exploration labels')
        self.protocol={'market':'historical_csi300','actual_signal_period':[str(self.label.index.get_level_values('datetime').min().date()),str(end.date())],
            'latest_label_exit':str(exit_date.date()),'horizon_trading_days':21,'label':'close[T+22]/close[T+1]-1',
            'volume':'Qlib $volume, float64','rank':'daily historical-member average-tie percentile before label filtering',
            'metrics':'daily cross-sectional IC/Rank IC, equal weight over valid signal dates; minimum 10 paired stocks',
            'missing_values':'No fill; full rolling windows. Fields and outputs masked outside historical membership; EMA reset on gaps and requires span consecutive values.',
            'annual_grouping':'Signal year; late 2020-2022 labels can enter next exploration year; no labels enter 2024',
            'warmup_start':config.loop.warmup_start,'model_training':False,'portfolio_backtest':False}
        self.provenance={'provider':str(provider),'calendar_sha256':hashlib.sha256((provider/'calendars/day.txt').read_bytes()).hexdigest()}
        self.seen={}

    def evaluate(self,proposal,run,round_number,emit=print):
        results=[];daily_rows=[];values={};fingerprints={}
        for name,f in proposal['factors'].items():
            result={'name':name,**f,'expected_direction':'positive','overall':None,'annual':{},'status':'failed'}
            try:
                key=canonical(f['expression'],self.config.research.fields)
                fingerprints[name]=key
                if key in self.seen:
                    result['duplicate_of']=self.seen[key]
                # Exact duplicates are evaluated but explicitly flagged, never counted as novel.
                signal=calculate(f['expression'],self.frame,self.member).reindex(self.label.index)
                if signal.notna().sum()==0: raise ExpressionError('No valid factor values')
                values[name]=signal
                joined=pd.concat([signal.rename('factor'),self.label],axis=1)
                rows=[]
                for date,g in joined.groupby(level='datetime'):
                    pair=g.dropna();ok=len(pair)>=10 and pair.factor.nunique()>1 and pair.forward_return.nunique()>1
                    rows.append({'date':str(date.date()),'factor':name,'n':len(pair),'coverage':float(g.factor.notna().mean()),
                        'ic':pair.factor.corr(pair.forward_return) if ok else np.nan,
                        'rank_ic':pair.factor.corr(pair.forward_return,method='spearman') if ok else np.nan})
                daily_rows.extend(rows);daily=pd.DataFrame(rows)
                result.update(overall=aggregate(daily),annual={str(y):aggregate(g) for y,g in daily.groupby(daily.date.str[:4])},status='completed')
                if not result['overall']['valid_rank_ic_days']:
                    result.update(status='failed',error='No valid daily Rank IC (constant or insufficient observations)')
                self.seen.setdefault(key,{'round':round_number,'factor':name})
            except ExpressionError as error:
                result['error']=str(error)
            results.append(result)
            if result['status']=='completed':
                def fmt(x): return 'N/A' if x is None else f'{x:+.4f}'
                annual=' | '.join(f"{y}: {fmt(v['mean_rank_ic'])}" for y,v in result['annual'].items())
                emit(f"  {name} Rank IC → {annual} | 全期: {fmt(result['overall']['mean_rank_ic'])}")
                if 'duplicate_of' in result: emit('  注意：此表达式与历史候选重复。')
            else: emit(f"  {name} 评估失败：{result['error']}")
        value_frame=pd.DataFrame(values,index=self.label.index)
        correlation=None
        if len(values)==2:
            columns=list(values)
            v=value_frame.groupby(level='datetime').apply(lambda g:g[columns[0]].corr(g[columns[1]],method='spearman')).mean()
            correlation=float(v) if pd.notna(v) else None
        evaluation={'schema_version':'1.1','status':'completed','stage':'exploration','proposal_run_id':run.name,
            'round':round_number,'research_direction':self.config.research.direction,'hypothesis':proposal['hypothesis'],
            'protocol':self.protocol,'factors':results,'mean_pairwise_factor_rank_correlation':correlation,
            'provenance':self.provenance,'limitations':['Adaptive reuse of 2020-2023 exploration results; not independent validation.',
                'Overlapping 21-day labels are not independent; no significance test.',
                'No ST exclusion, industry/size neutralization, price/volume controls, costs or portfolio simulation.',
                'Membership-masked generic evaluator can differ from old fixed-formula scripts around constituent entry.']}
        (run/'evaluation.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        pd.DataFrame(daily_rows,columns=['date','factor','n','coverage','ic','rank_ic']).to_csv(run/'daily_metrics.csv',index=False)
        value_frame.join(self.label).to_parquet(run/'factor_values.parquet')
        return evaluation

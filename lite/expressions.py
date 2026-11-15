"""Restricted expression interpreter. No eval, Python code generation or execution."""
import ast
import re
import numpy as np
import pandas as pd

GUIDE = '''Loop expression contract (takes precedence over generic operator examples):
Exactly 2 factors. Positive expected direction: larger score predicts higher target return.
Use only configured $fields, finite numeric constants, + - * / **, comparisons,
and these functions: ABS(x), SIGN(x), LOG(x), SQRT(x), POW(x,n), MIN(x,y), MAX(x,y),
RANK(x), MEAN(x), STD(x), ZSCORE(x), SCALE(x), DELAY(x,n), TS_SUM(x,n), TS_MEAN(x,n), TS_STD(x,n), TS_MIN(x,n),
TS_MAX(x,n), EMA(x,n). No other functions, Python, conditionals or attributes.
Integer n must be 1..252 (DELAY permits 0); exponent must be a constant integer -4..4.
RANK is daily historical-universe average-tie percentile ranking, before label filtering.
MEAN and STD are daily cross-sectional statistics over valid universe members, broadcast to valid inputs;
STD uses ddof=1. ZSCORE=(x-MEAN(x))/STD(x); SCALE=x/sum(abs(x)) per date.
Zero or undefined cross-sectional denominators produce missing values.
Rolling windows require n valid observations; STD uses ddof=1. EMA alpha=2/(n+1),
adjust=False, initialize at first valid value, emit after n consecutive valid values,
reset at missing values. Membership gaps reset time-series operators.
Division by zero, LOG of nonpositive values and SQRT of negative values yield missing.
Use at most 250 expression characters. Do not repeat an earlier expression.
Keep rank/EMA order explicit. Do not assume smoothing or neutralization is automatic.'''

FUNCS={'ABS':1,'SIGN':1,'LOG':1,'SQRT':1,'POW':2,'MIN':2,'MAX':2,'RANK':1,
       'MEAN':1,'STD':1,'ZSCORE':1,'SCALE':1,
       'DELAY':2,'TS_SUM':2,'TS_MEAN':2,'TS_STD':2,'TS_MIN':2,'TS_MAX':2,'EMA':2}
class ExpressionError(ValueError): pass

def parse(expression, fields):
    if len(expression)>250: raise ExpressionError('Expression exceeds 250 characters')
    text=re.sub(r'\$([a-z]+)',r'field_\1',expression)
    try: tree=ast.parse(text,mode='eval')
    except SyntaxError as e: raise ExpressionError('Invalid expression syntax') from e
    if len(list(ast.walk(tree)))>150: raise ExpressionError('Expression too complex')
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float) and np.isfinite(n.value): return
        if isinstance(n,ast.Name) and n.id in {'field_'+f[1:] for f in fields}: return
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)): visit(n.operand);return
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            if isinstance(n.op,ast.Pow):
                try: power=ast.literal_eval(n.right)
                except (ValueError,TypeError): raise ExpressionError('Power must be a bounded integer')
                if type(power)!=int or not -4<=power<=4: raise ExpressionError('Power must be -4..4')
            visit(n.left);visit(n.right);return
        if isinstance(n,ast.Compare) and len(n.ops)==1 and isinstance(n.ops[0],(ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.Eq,ast.NotEq)):
            visit(n.left);visit(n.comparators[0]);return
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in FUNCS and not n.keywords and len(n.args)==FUNCS[n.func.id]:
            for a in n.args: visit(a)
            if n.func.id=='POW':
                try: power=ast.literal_eval(n.args[1])
                except (ValueError,TypeError): raise ExpressionError('Power must be a bounded integer')
                if type(power)!=int or not -4<=power<=4: raise ExpressionError('Power must be -4..4')
            if n.func.id in {'DELAY','EMA','TS_SUM','TS_MEAN','TS_STD','TS_MIN','TS_MAX'}:
                arg=n.args[1]
                if not isinstance(arg,ast.Constant) or type(arg.value)!=int or not (0 if n.func.id=='DELAY' else 1)<=arg.value<=252:
                    raise ExpressionError('Window must be a bounded literal integer')
            return
        raise ExpressionError('Unsupported syntax, field or operator')
    visit(tree.body)
    if not any(isinstance(n,ast.Name) and n.id.startswith('field_') for n in ast.walk(tree)):
        raise ExpressionError('Expression must use a market field')
    return tree

def canonical(expression,fields): return ast.dump(parse(expression,fields),include_attributes=False)

def ema(s,n):
    valid=s.dropna()
    return valid.groupby(s.isna().cumsum().reindex(valid.index),sort=False).transform(
        lambda x:x.ewm(alpha=2/(n+1),adjust=False,min_periods=n).mean()).reindex(s.index)

def calculate(expression, frame, member):
    tree=parse(expression,list(frame.columns))
    def series(x): return x if isinstance(x,pd.Series) else pd.Series(x,index=frame.index,dtype=float)
    def clean(x): return series(x).replace([np.inf,-np.inf],np.nan).where(member)
    def ts(x,fn): return series(x).groupby(level='instrument',sort=False).transform(fn)
    def run(n):
        if isinstance(n,ast.Constant): return n.value
        if isinstance(n,ast.Name): return frame['$'+n.id[6:]]
        if isinstance(n,ast.UnaryOp): return clean(-run(n.operand) if isinstance(n.op,ast.USub) else run(n.operand))
        if isinstance(n,ast.BinOp):
            a,b=run(n.left),run(n.right)
            with np.errstate(all='ignore'):
                if isinstance(n.op,ast.Add): x=a+b
                elif isinstance(n.op,ast.Sub): x=a-b
                elif isinstance(n.op,ast.Mult): x=a*b
                elif isinstance(n.op,ast.Div): x=series(a)/series(b).replace(0,np.nan)
                else: x=series(a)**b
            return clean(x)
        if isinstance(n,ast.Compare):
            a,b=series(run(n.left)),series(run(n.comparators[0]))
            op={ast.Lt:lambda:a<b,ast.LtE:lambda:a<=b,ast.Gt:lambda:a>b,ast.GtE:lambda:a>=b,ast.Eq:lambda:a==b,ast.NotEq:lambda:a!=b}
            return clean(op[type(n.ops[0])]().astype(float).where(a.notna()&b.notna()))
        name=n.func.id;a=series(run(n.args[0]));b=(ast.literal_eval(n.args[1]) if name=='POW' else run(n.args[1])) if len(n.args)>1 else None
        if name=='RANK': return a.where(member).groupby(level='datetime').rank(method='average',pct=True)
        if name in {'MEAN','STD','ZSCORE','SCALE'}:
            a=a.where(member).replace([np.inf,-np.inf],np.nan)
            grouped=a.groupby(level='datetime')
            if name=='MEAN': x=grouped.transform('mean').where(a.notna())
            elif name=='STD': x=grouped.transform('std').where(a.notna())
            elif name=='ZSCORE': x=(a-grouped.transform('mean'))/grouped.transform('std').replace(0,np.nan)
            else: x=a/a.abs().groupby(level='datetime').transform('sum').replace(0,np.nan)
            return clean(x)
        if name=='DELAY': x=ts(a,lambda s:s.shift(b))
        elif name=='EMA': x=ts(a,lambda s:ema(s,b))
        elif name.startswith('TS_'):
            method={'TS_SUM':'sum','TS_MEAN':'mean','TS_STD':'std','TS_MIN':'min','TS_MAX':'max'}[name]
            x=ts(a,lambda s:getattr(s.rolling(b,min_periods=b),method)())
        elif name=='ABS': x=a.abs()
        elif name=='SIGN': x=np.sign(a)
        elif name=='LOG': x=np.log(a.where(a>0))
        elif name=='SQRT': x=np.sqrt(a.where(a>=0))
        elif name=='POW':
            with np.errstate(all='ignore'): x=a**b
        elif name=='MIN': x=np.minimum(a,b)
        else: x=np.maximum(a,b)
        return clean(x)
    return clean(run(tree.body))

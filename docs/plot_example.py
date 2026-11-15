"""Recreate the README chart from aggregate daily results (requires matplotlib)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
ROOT=Path(__file__).resolve().parent
data=pd.read_csv(ROOT/'example_daily.csv',parse_dates=['date']).set_index('date').sort_index()
fig,(ic,nav)=plt.subplots(2,1,figsize=(11,7.5),sharex=True,gridspec_kw={'height_ratios':[1,1.15]})
colors={'candidate':'#2359c4','alpha22':'#d36b38'}
for key,label in [('candidate','Round 29, factor 2'),('alpha22','Alpha #22')]:
    ic.plot(data.index,data[f'{key}_rank_ic'].rolling(63,min_periods=63).mean(),label=label,color=colors[key],linewidth=1.8)
    nav.plot(data.index,data[f'{key}_nav'],label=label,color=colors[key],linewidth=1.8)
ic.axhline(0,color='#9ca3af',lw=0.9,linestyle='--')
nav.axhline(1,color='#9ca3af',lw=0.9,linestyle='--')
ic.set_title('63-trading-day rolling mean of daily Rank IC',loc='left',fontsize=12)
ic.set_ylabel('Rank IC')
nav.set_title('Highest-scoring quintile: net portfolio value',loc='left',fontsize=12)
nav.set_ylabel('Net value (initial = 1)')
for ax in (ic,nav):
    ax.grid(alpha=.23);ax.legend(loc='upper left',frameon=False);ax.spines[['top','right']].set_visible(False)
nav.xaxis.set_major_locator(mdates.MonthLocator(interval=4));nav.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.suptitle('CSI 300 · 2024–2026 example',fontsize=15,y=.98)
fig.text(.5,.02,'2026 through Sep 30 · portfolio: 5-day rebalance, 10 bp per side',ha='center',fontsize=9,color='#4b5563')
fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(ROOT/'example_results.png',dpi=170,bbox_inches='tight');plt.close(fig)

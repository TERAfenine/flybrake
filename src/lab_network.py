"""Real directed BANC graph views. IDs stay strings across the JS boundary."""
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

def neuron_connections(graph,index):
    if not 0<=index<len(graph['ids']):
        raise ValueError('뉴런 인덱스 범위를 벗어났습니다.')
    post,pre = np.where(graph['mask']>0)
    keep = (post==index)|(pre==index)
    post,pre = post[keep],pre[keep]
    counts = graph.get('counts')
    return pd.DataFrame({'방향':np.where(post==index,'들어옴 → 선택 뉴런','나감: 선택 뉴런 →'),
                         '출발 BANC ID':[str(graph['ids'][i]) for i in pre],
                         '도착 BANC ID':[str(graph['ids'][i]) for i in post],
                         'pre_index':pre,'post_index':post,
                         '시냅스 수':counts[post,pre] if counts is not None else np.full(len(pre),np.nan)})

def node_from_event(event,ids):
    """Ignore edge clicks or stale/invalid IDs; never round a 64-bit BANC ID."""
    points = event.get('selection',{}).get('points',[]) if isinstance(event,dict) else []
    if len(points) != 1:
        return None
    custom = points[0].get('customdata')
    if not isinstance(custom,(list,tuple)) or len(custom)<2:
        return None
    try:
        index = int(custom[0])
        if str(index)!=str(custom[0]) or not 0<=index<len(ids) or str(ids[index])!=custom[1]:
            return None
    except (ValueError,TypeError,OverflowError):
        return None
    return index

def network_layout(graph):
    # Display coordinates only. This seed does not choose a biological circuit.
    directed = nx.from_numpy_array(graph['mask'].T,create_using=nx.DiGraph)
    pos = nx.spring_layout(directed,seed=42,iterations=60)
    return np.asarray([pos[i] for i in range(len(graph['ids']))])

def visible_connections(graph,index,mode,nt_filter='전체'):
    """A display-only induced subgraph: never modifies graph arrays."""
    post,pre=np.where(graph['mask']>0)
    allowed=np.flatnonzero(graph['signs']<0) if nt_filter=='GABA 예측만' else np.arange(len(graph['ids']))
    keep=np.isin(post,allowed)&np.isin(pre,allowed)
    if mode=='선택 뉴런 중심':
        keep &= (post==index)|(pre==index)
    post,pre=post[keep],pre[keep]
    nodes=sorted(set(post)|set(pre)|({index} if index in allowed else set())) if mode=='선택 뉴런 중심' else allowed.tolist()
    return nodes,post,pre

def network_figure(graph,index,mode='전체 96개 뉴런',disabled=None,positions=None,nt_filter='전체',zoom=1.,revision=0):
    ids,mask,signs = graph['ids'],graph['mask'],graph['signs']
    pos = network_layout(graph) if positions is None else positions
    nodes,post,pre = visible_connections(graph,index,mode,nt_filter)
    fig=go.Figure()
    # Straight directed lines with arrowheads at 78% of each pre->post vector.
    x=[];y=[]
    for s,d in zip(pre,post):
        x.extend([pos[s,0],pos[d,0],None]);y.extend([pos[s,1],pos[d,1],None])
    fig.add_trace(go.Scatter(x=x,y=y,mode='lines',line=dict(width=.7,color='rgba(115,130,150,.22)'),
                             hoverinfo='skip',showlegend=False,name='실제 연결'))
    delta=pos[post]-pos[pre]
    arrows=pos[pre]+.78*delta
    angles=np.degrees(np.arctan2(delta[:,0],delta[:,1]))
    fig.add_trace(go.Scatter(x=arrows[:,0],y=arrows[:,1],mode='markers',
                             marker=dict(symbol='triangle-up',size=6,angle=angles,
                                color=['#176b86' if d==index else '#b95724' if s==index else 'rgba(95,110,130,.35)' for s,d in zip(pre,post)]),
                             hoverinfo='skip',showlegend=False,name='방향 pre → post'))
    neighbors=set(np.flatnonzero(mask[index]))|set(np.flatnonzero(mask[:,index]))
    for label,sign,color in [('GABA 예측',-1,'#667bd4'),('ACh 예측',1,'#d88945')]:
        subset=[i for i in nodes if signs[i]==sign]
        custom=[[str(i),str(ids[i]),label,int(mask[i].sum()),int(mask[:,i].sum())] for i in subset]
        fig.add_trace(go.Scatter(x=pos[subset,0],y=pos[subset,1],mode='markers',name=label,
            customdata=custom,marker=dict(size=[24 if i==index else 11 for i in subset],color=color,
                opacity=[1. if i==index else .3 if (mode=='GABA 예측 강조' and sign==1) or (mode!='선택 뉴런 중심' and i not in neighbors) else 1. for i in subset],
                line=dict(width=[3 if i==index else 1.5 if i in neighbors else 0 for i in subset],
                          color=['#162b43' if i==index else '#0f766e' for i in subset])),
            hovertemplate='인덱스 %{customdata[0]}<br>BANC %{customdata[1]}<br>%{customdata[2]}'
                          '<br>입력 %{customdata[3]} · 출력 %{customdata[4]}<extra></extra>'))
    for incoming,color,label in [(True,'#176b86','들어오는 연결'),(False,'#b95724','나가는 연결')]:
        keep=post==index if incoming else pre==index
        xx=[];yy=[]
        for s,d in zip(pre[keep],post[keep]):
            xx.extend([pos[s,0],pos[d,0],None]);yy.extend([pos[s,1],pos[d,1],None])
        fig.add_trace(go.Scatter(x=xx,y=yy,mode='lines',line=dict(width=1.3,color=color),
            hoverinfo='skip',name=label))
    if disabled is not None and disabled in nodes:
        fig.add_trace(go.Scatter(x=[pos[disabled,0]],y=[pos[disabled,1]],mode='markers',
            name='활성값 0 고정',customdata=[[str(disabled),str(ids[disabled])]],
            marker=dict(size=22,symbol='x',color='#c73342'),hovertemplate='계산상 비활성화<extra></extra>'))
    fig.update_layout(height=460,margin=dict(l=10,r=10,t=50,b=65),template='plotly_white',
        title=dict(text='BANC · 실제 방향성 연결',font=dict(size=16)),
        clickmode='event+select',dragmode='pan',showlegend=True,
        xaxis=dict(visible=False),yaxis=dict(visible=False,scaleanchor='x',scaleratio=1),
        uirevision=f'{mode}-{nt_filter}-{index}-{zoom}-{revision}',legend=dict(orientation='h',y=-.05,font=dict(size=10)),
        font=dict(color='#173343'),paper_bgcolor='#f7f9fb',plot_bgcolor='#f7f9fb')
    if nodes:
        low=pos[nodes].min(axis=0);high=pos[nodes].max(axis=0)
        center=pos[index] if zoom>1 and index in nodes else (low+high)/2
        span=max(float((high-low).max()),.2)*.65/max(float(zoom),.25)
        fig.update_xaxes(range=[center[0]-span,center[0]+span])
        fig.update_yaxes(range=[center[1]-span,center[1]+span])
    if index in nodes:
        fig.add_annotation(x=pos[index,0],y=pos[index,1],text=f'선택 {index}',showarrow=False,
                           yshift=25,bgcolor='#102d3b',font=dict(color='white',size=12),borderpad=4)
    return fig

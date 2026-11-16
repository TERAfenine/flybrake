"""Plots use exactly the same read-only view as the Lab cards and table."""
import plotly.graph_objects as go

COLORS = ['#176b86', '#c46a23', '#7562a7']

def comparison_figure(view, metrics, average=False):
    fig = go.Figure()
    for index, (model, group) in enumerate(view.groupby('model', sort=False)):
        data = group.set_index('metric').loc[metrics]
        fig.add_trace(go.Bar(
            name=model, x=data.label.tolist(), y=data.value.tolist(),
            marker_color=COLORS[index % len(COLORS)],
            error_y=dict(type='data', array=data['std'].tolist(), visible=average),
            customdata=data[['n_valid']].to_numpy(),
            hovertemplate='%{x}: %{y:.3%}<br>유효 시드 수: %{customdata[0]}<extra>%{fullData.name}</extra>'))
    fig.update_layout(barmode='group', height=330, margin=dict(l=16,r=16,t=24,b=16),
                      legend=dict(orientation='h',y=1.15), template='plotly_white',
                      yaxis=dict(title='표본 비율 (%)',tickformat='.2%' if any(m in metrics for m in ['false_alarm_per_empty','unsafe_false_empty_per_occupied']) else '.0%', rangemode='tozero'))
    return fig

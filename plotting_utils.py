import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils
import json

def plot_metrics(
    df: pd.DataFrame, 
    patient: str, 
    hemisphere: int, 
    changes_df: pd.DataFrame
) -> go.Figure:
    """
    Generate a plot with multiple subplots to visualize various metrics including LFP amplitude, linear AR model,
    and R² values over time, before and after DBS.

    Parameters:
        df (pd.Dataframe): Dataframe containing processed percept data.
        patient (str): Patient identifier.
        hemisphere (int): Hemisphere index (0 or 1).
        changes_df (pd.Dataframe): Dataframe containing stimulation parameters and changes.

    Returns:
        go.Figure: A Plotly figure with the generated subplots.
    """
    # Color and style settings
    c_preDBS = 'rgba(255, 215, 0, 0.5)'
    c_responder = 'rgba(0, 0, 255, 1)'
    c_nonresponder = 'rgba(255, 185, 0, 1)'
    c_disinhibited = '#ff0000'
    c_dots = 'rgba(128, 128, 128, 0.5)'
    c_linAR = 'rgba(51, 160, 44, 1)'
    c_OG = 'rgba(128, 128, 128, 0.7)'
    sz = 5

    # Load param settings
    with open('param.json', 'r') as f:
        param_dict = json.load(f)

    with open('ocd_patient_info.json', 'r') as f:
        patient_dict = json.load(f)[patient]
    
    model = param_dict['model']
    hemisphere = 'left' if hemisphere == 0 else 'right'

    pt_df = df.query('pt_id == @patient')
    days = pt_df.groupby('days_since_dbs').head(1)['days_since_dbs']
    t = pt_df['CT_timestamp']
    OG = pt_df[f'lfp_{hemisphere}_filled_{model}']
    linAR = pt_df[f'lfp_{hemisphere}_preds_{model}']
    res = pt_df[f'lfp_{hemisphere}_residuals_{model}']
    linAR_R2 = pt_df.groupby('days_since_dbs').head(1)[f'lfp_{hemisphere}_day_r2_{model}']
    state_labels = pt_df.groupby('days_since_dbs').head(1)['state_label']

    # Identify responder and non-responder indices
    pre_DBS_idx = np.where(days < 0)[0]

    # Responder indices
    if 3 in pt_df['state_label']:
        responder_idx = np.where(state_labels == 3)[0]
        unknown_idx = np.where(state_labels == 4)[0]
    # Non-responder indices
    else:
        non_responder_idx = np.where(state_labels == 2)[0]

    # Hypomanic indices
    if 1 in pt_df['state_label']:
        hypomanic_idx = np.where(state_labels == 1)[0]

    # Identify discontinuities in the days array
    start_index = np.where(np.diff(days) > 7)[0] + 1
    start_index = np.concatenate(([0], start_index, [len(days)]))

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=4,
        row_heights=[0.5, 0.5],
        column_widths=[0.3, 0.35, 0.35, 0.1],
        specs=[[{"colspan": 4}, None, None, None],
                [{"colspan": 3}, None, None, {"colspan": 1}]],
        subplot_titles=("Full Time-Domain Plot",
                        "Linear AR R² Over Time", "Linear AR R² Violin Plot"))

    # Set plot aesthetics
    grid_color = '#a0a0a0'
    title_font_color = '#2e2e2e'
    axis_title_font_color = '#2e2e2e'
    axis_line_color = '#2e2e2e'
    plot_bgcolor = 'rgba(240, 240, 240, 1)'
    paper_bgcolor = 'rgba(240, 240, 240, 1)'

    fig.update_layout(
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        annotations=[dict(
            text='',
            xref='paper',
            yref='paper',
            x=0,
            y=1,
            showarrow=False,
            font=dict(
                size=20,
                color=title_font_color,
                family="Helvetica"
            )
        )]
    )

    # Plot Full Time-Domain Plot
    for i in range(len(start_index) - 1):
        segment_days = np.ravel(days[start_index[i]+1:start_index[i+1]])
        segment_OG = pt_df.query('days_since_dbs in @segment_days')[f'lfp_{hemisphere}_z_scored_{model}']
        segment_linAR = pt_df.query('days_since_dbs in @segment_days')[f'lfp_{hemisphere}_preds_{model}']
        segment_times = pt_df.query('days_since_dbs in @segment_days')['CT_timestamp']

        mask = ~np.isnan(segment_times) & ~np.isnan(segment_OG)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            segments = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
            
            for segment in segments:
                if len(segment) > 0:
                    fig.add_trace(go.Scatter(
                        x=segment_times.values[segment],
                        y=segment_OG.values[segment],
                        mode='lines',
                        line=dict(color=c_OG, width=1),
                        showlegend=False
                    ), row=1, col=1)

        linAR_mask = ~np.isnan(segment_times) & ~np.isnan(segment_linAR)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            segments = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
            
            for segment in segments:
                if len(segment) > 0:
                    fig.add_trace(go.Scatter(
                        x=segment_times.values[segment],
                        y=segment_linAR.values[segment],
                        mode='lines',
                        line=dict(color=c_linAR, width=1),
                        showlegend=False
                    ), row=1, col=1)
    fig.add_vline(x=patient_dict['dbs_date'], row=1, col=1, line_dash='dash', line_color='hotpink', line_width=5)

    #fig.add_trace(go.Scatter(x=linAR_t, y=pt_df.dropna(subset=[f'lfp_{hemisphere}_preds_{model}'])[f'lfp_{hemisphere}_preds_{model}'], mode='lines', name="Linear AR", line=dict(color=c_linAR, width=1.5), showlegend=False), row=1, col=1)                
    fig.update_yaxes(title_text="9 Hz LFP (mV)", row=1, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)

     # Linear AR R² Over Time
    for i in range(len(start_index) - 1):
        segment_days = days.values[start_index[i]+1:start_index[i+1]]
        segment_linAR_R2 = linAR_R2.values[start_index[i]+1:start_index[i+1]]
        
        # Plot the dots
        fig.add_trace(go.Scatter(
            x=segment_days,
            y=segment_linAR_R2,
            mode='markers',
            marker=dict(color=c_dots, size=sz),
            showlegend=False
        ), row=2, col=1)
        
    color_dict = {0: c_preDBS, 1: c_disinhibited, 2: c_nonresponder, 3: c_responder, 4: c_dots}

    for color in color_dict.keys():
        if color not in pt_df['state_label'].values:
            continue    
        state_df = pt_df.query('state_label == @color')
        state_days = state_df.groupby('days_since_dbs').head(1)['days_since_dbs']
        state_r2 = state_df.groupby('days_since_dbs').head(1)[f'lfp_{hemisphere}_day_r2_{model}']

        fig.add_trace(go.Scatter(
            x=state_days,
            y=state_r2.rolling(window=5, min_periods=1).mean(),
            mode='lines',
            line=dict(color=color_dict[color]),
            showlegend=False
        ), row=2, col=1)

    fig.add_vline(x=0, row=2, col=1, line_dash='dash', line_color='hotpink', line_width=5)

    fig.update_yaxes(title_text="Linear AR R²", range=(-0.5, 1), row=2, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    
    # Linear AR R² Violin Plot
    VIOLIN_WIDTH = 7.0
    fig.add_trace(go.Violin(
            y=pt_df.query('state_label == 0').groupby('days_since_dbs').head(1)[f'lfp_{hemisphere}_day_r2_{model}'],
            name='', 
            side='negative', 
            line_color=c_preDBS, 
            fillcolor=c_preDBS,
            showlegend=False,
            width=VIOLIN_WIDTH,
            meanline_visible=True, 
            meanline=dict(color='black', width=2)
        ), row=2, col=4)

    if patient_dict['response_status'] == 1:
        fig.add_trace(go.Violin(
            y=pt_df.query("days_since_dbs >= @patient_dict['response_date']").groupby('days_since_dbs').head(1)[f'lfp_{hemisphere}_day_r2_{model}'],  
            side='positive', 
            line_color=c_responder, 
            fillcolor=c_responder,
            showlegend=False,
            width=VIOLIN_WIDTH,
            meanline_visible=True, 
            meanline=dict(color='white', width=2)
        ), row=2, col=4)
    else:
        fig.add_trace(go.Violin(
            y=pt_df.query('days_since_dbs > 0').groupby('days_since_dbs').head(1)[f'lfp_{hemisphere}_day_r2_{model}'], 
            side='positive', 
            line_color=c_nonresponder, 
            fillcolor=c_nonresponder,
            showlegend=False,
            width=VIOLIN_WIDTH,
            meanline_visible=True, 
            meanline=dict(color='black', width=2)
        ), row=2, col=4)

    fig.update_yaxes(range=(-0.5, 1), row=2, col=4, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    # Set overall layout aesthetics
    fig.update_layout(
        height=650,
        width=900,
        showlegend=True,
        legend=dict(x=0.85, y=0.85, bgcolor='rgba(255, 255, 255, 0.7)', font=dict(color=title_font_color), itemsizing='constant', itemwidth=30),
        margin=dict(l=50, r=50, b=50, t=50),
        font=dict(color=title_font_color, family="Helvetica"),
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor=grid_color),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor=grid_color),
    )

    annotations = [
        dict(
            text="Full Time-Domain Plot",
            x=0.5,
            xref="paper",
            y=1,
            yref="paper",
            showarrow=False,
            font=dict(size=14, color=title_font_color, family="Helvetica")
        ),
        dict(
            text="Linear AR R² Over Time",
            x=0.35,
            xref="paper",
            y=0.35,  # Adjusted Y position to move the annotation upwards
            yref="paper",
            showarrow=False,
            font=dict(size=14, color=title_font_color, family="Helvetica")
        )
    ]

    fig.update_layout(annotations=annotations)

    return fig

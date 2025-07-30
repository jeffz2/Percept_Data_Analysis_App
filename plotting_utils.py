import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils
import json
import scipy.stats as stats
import burdened_state_regression as reg

def plot_metrics(
    df: pd.DataFrame, 
    patient: str, 
    hemisphere: int, 
    changes_df: pd.DataFrame,
    show_reg: bool = False,
    show_changes: bool = False
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
    with open('data/param.json', 'r') as f:
        param_dict = json.load(f)

    with open('data/patient_info.json', 'r') as f:
        patients_dict = json.load(f)

    patient_dict = patients_dict[patient]
    model = param_dict['model']
    delta = param_dict['delta']
    hemisphere = 'left' if hemisphere == 0 else 'right'

    pt_df = df.query('pt_id == @patient and lead_location == "VC/VS"')
    days = pt_df.drop_duplicates(subset=['days_since_dbs'])['days_since_dbs']
    t = pt_df['CT_timestamp']
    OG = pt_df[f'lfp_{hemisphere}_filled_{model}']
    linAR = pt_df[f'lfp_{hemisphere}_preds_{model}']
    res = pt_df[f'lfp_{hemisphere}_residuals_{model}']
    linAR_R2 = pt_df.drop_duplicates(subset=['days_since_dbs'])[f'lfp_{hemisphere}_day_r2_{model}']
    state_labels = pt_df.drop_duplicates(subset=['days_since_dbs'])['state_label']

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

    if show_changes:
        fig.add_vline(x=1)

    #fig.add_trace(go.Scatter(x=linAR_t, y=pt_df.dropna(subset=[f'lfp_{hemisphere}_preds_{model}'])[f'lfp_{hemisphere}_preds_{model}'], mode='lines', name="Linear AR", line=dict(color=c_linAR, width=1.5), showlegend=False), row=1, col=1)                
    fig.update_yaxes(title_text="LFP (z-scored)", row=1, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(title_text="Date (CT)", tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)

     # Linear AR R² Over Time
    if delta:
        linAR_R2 = reg.delta_model(linAR_R2, [], patient)
    for i in range(len(start_index) - 1):
        segment_days = days.values[start_index[i]+1:start_index[i+1]]
        segment_linAR_R2 = linAR_R2.values[start_index[i]+1:start_index[i+1]]
        
        # Plot the dots
        fig.add_trace(go.Scatter(
            x=segment_days,
            y=pd.Series(segment_linAR_R2).rolling(window=5, min_periods=1).mean(),
            mode='lines',
            line=dict(color=c_dots),
            showlegend=False
        ), row=2, col=1)
        
    color_dict = {0: c_preDBS, 1: c_disinhibited, 2: c_nonresponder, 3: c_responder, 4: c_dots}

    for color in color_dict.keys():
        if color not in pt_df['state_label'].values:
            continue    
        state_df = pt_df.query('state_label == @color')
        state_days = state_df.drop_duplicates(subset=['days_since_dbs'])['days_since_dbs']
        state_r2 = state_df.drop_duplicates(subset=['days_since_dbs'])[f'lfp_{hemisphere}_day_r2_{model}']

        fig.add_trace(go.Scatter(
            x=state_days,
            y=state_r2,
            mode='markers',
            line=dict(color=color_dict[color]),
            showlegend=False
        ), row=2, col=1)

    fig.add_vline(x=0, row=2, col=1, line_dash='dash', line_color='hotpink', line_width=5)

    if show_reg:
        _, threshold, prediction, cm = reg.run_regression(pt_df, patients_dict.keys(), delta, [f'lfp_{hemisphere}_day_r2_{model}'], patient)

        tpr = cm[1][1] / (cm[1][1] + cm[1][0]) if cm.shape != (1, 1) else 1 if patient_dict['response_status'] == 1 else 0
        tnr = cm[0][0] / (cm[0][0] + cm[0][1]) if cm.shape != (1, 1) else 1 if patient_dict['response_status'] == 0 else 0

        fig.add_hline(y=threshold, row=2, col=1, line_dash='dash', line_color='black', line_width=2)
        fig.add_annotation(x=days[-1], y=threshold, row=2, col=1, text=f'Regression Decision Boundary: {np.round(threshold, 3)}', showarrow=False)
        fig.add_hrect(-0.5, threshold, row=2, col=1, fillcolor=c_responder, opacity=0.5, line_width=0)
        fig.add_annotation(x=days[-1], y=-0.5, row=2, col=1, text=f'TPR: {tpr}', showarrow=False)
        fig.add_hrect(threshold, 1, row=2, col=1, fillcolor=c_nonresponder, opacity=0.5, line_width=0)
        fig.add_annotation(x=days[-1], y=1, row=2, col=1, text=f'TNR: {tnr}', showarrow=False)

    fig.update_yaxes(title_text="Linear AR R²", range=(-0.5, 1), row=2, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(title_text='Days Since DBS Activation', tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    
    # Linear AR R² Violin Plot
    VIOLIN_WIDTH = 7.0
    violin_df = pt_df.drop_duplicates(subset=['days_since_dbs']).dropna(subset=[f'lfp_{hemisphere}_day_r2_{model}'])
    fig.add_trace(go.Violin(
            y=violin_df.query('state_label == 0')[f'lfp_{hemisphere}_day_r2_{model}'],
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
            y=violin_df.query("days_since_dbs >= @patient_dict['response_date']")[f'lfp_{hemisphere}_day_r2_{model}'],  
            side='positive', 
            line_color=c_responder, 
            fillcolor=c_responder,
            showlegend=False,
            width=VIOLIN_WIDTH,
            meanline_visible=True, 
            meanline=dict(color='white', width=2)
        ), row=2, col=4)

        t_val, p_val = stats.ttest_ind(violin_df.query('state_label == 0')[f'lfp_{hemisphere}_day_r2_{model}'], violin_df.query("days_since_dbs >= @patient_dict['response_date']")[f'lfp_{hemisphere}_day_r2_{model}'], equal_var=False)
    else:
        fig.add_trace(go.Violin(
            y=violin_df.query('days_since_dbs > 0')[f'lfp_{hemisphere}_day_r2_{model}'], 
            side='positive', 
            line_color=c_nonresponder, 
            fillcolor=c_nonresponder,
            showlegend=False,
            width=VIOLIN_WIDTH,
            meanline_visible=True, 
            meanline=dict(color='black', width=2)
        ), row=2, col=4)

        t_val, p_val = stats.ttest_ind(violin_df.query('state_label == 0')[f'lfp_{hemisphere}_day_r2_{model}'], violin_df.query("days_since_dbs > 0")[f'lfp_{hemisphere}_day_r2_{model}'], equal_var=False)

    fig.update_yaxes(range=(-0.5, 1), row=2, col=4, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(title_text=f"t = {np.round(t_val, 3)}\np = {np.round(p_val, 3)}",row=2, col=4,tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
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

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name='Raw LFP (z-scored)',
        marker=dict(color=c_OG, symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name='AR(1) predicted LFP (z-scored)',
        marker=dict(color=c_linAR, symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name='Pre-DBS',
        marker=dict(color=c_preDBS, symbol='circle')
    ))
    if patient_dict['response_status'] == 1:
        fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name='Response',
        marker=dict(color=c_responder, symbol='circle')
        ))
    else:
        fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name='Non-response',
        marker=dict(color=c_nonresponder, symbol='circle')
        ))
    if 1 in pt_df['state_label'].values:
        fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name='Disinhibited',
        marker=dict(color=c_disinhibited, symbol='circle')
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name='DBS On',
        marker=dict(color='hotpink', symbol='square')
        ))
    
    fig.update_layout(annotations=annotations, legend=dict(x=1, y=0.5, xanchor="right", yanchor="middle"))

    return fig

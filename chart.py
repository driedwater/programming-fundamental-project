import plotly.graph_objs as go
import plotly.io as pio
# Optimized function to create a gauge chart
def sentiment_gauge(score):
    emoji = "😊" if score >= 0 else "😡"
    
    fig = go.Figure(go.Indicator(
        mode="gauge",
        gauge={
            'axis': {
                'range': [-1, 1],
                'dtick': 1
            }
        },
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Analysis"}
    ))

    # Add emoji at the center
    fig.add_annotation(
        x=0.5, y=0.5,
        text=emoji,
        font=dict(size=60),
        showarrow=False
    )

    # Disable unnecessary interactions
    fig.update_layout(
        hovermode=False,
        dragmode=False
    )

    # Reusable config
    config = {
        'displaylogo': False,
        'responsive': False,
        'modeBarButtonsToRemove': ['sendDataToCloud']
    }

    return pio.to_html(fig, full_html=False, config=config)
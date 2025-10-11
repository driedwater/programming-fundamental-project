import plotly.graph_objs as go
import plotly.io as pio
    
"""
    Generates an HTML representation of a gauge chart for sentiment analysis.

    This function creates a Plotly gauge chart visualizing a sentiment score.
    The chart displays the score on an axis ranging from -1 (negative) to 1 (positive)
    and includes a corresponding emoji at the center. The output is a raw HTML string
    of the Plotly figure, suitable for embedding.

    :param score: The sentiment score to display, expected to be between -1 and 1.
    :type score: float
    :return: A string containing the HTML representation of the Plotly gauge chart.
    :rtype: str
"""
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
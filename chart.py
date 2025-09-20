import plotly.graph_objs as go
import plotly.io as pio
# Create a gauge chart
def pos_figure(pos_sentiment):
    fig = go.Figure(go.Indicator(
        mode = "gauge",
        gauge = {
            'axis': {
                'range': [-1, 1],
                'dtick': 1}},
        value = pos_sentiment,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment analysis"}))

    fig.add_annotation(
        x=0.5, y=0.5,  # Center of the plot
        text="😊",
        font=dict(size=60),  # Adjust size as needed
        showarrow=False
    )

    fig.update_layout(hovermode=False,dragmode=False)

    config = {
    'displaylogo': False,
    'responsive': False,
    'modeBarButtonsToRemove': ['sendDataToCloud']
}

    return pio.to_html(fig, full_html=False, config=config)

def neg_figure(neg_sentiment):
    fig = go.Figure(go.Indicator(
        mode = "gauge",
        gauge = {
            'axis': {
                'range': [-1, 1],
                'dtick': 1}},
        value = neg_sentiment,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment analysis"}))

    fig.add_annotation(
        x=0.5, y=0.5,  # Center of the plot
        text="😡",
        font=dict(size=60),  # Adjust size as needed
        showarrow=False
    )

    fig.update_layout(hovermode=False,dragmode=False)

    config = {
    'displaylogo': False,
    'responsive': False,
    'modeBarButtonsToRemove': ['sendDataToCloud']
}

    return pio.to_html(fig, full_html=False, config=config)

def pos_extract_figure(pos_extract_sentiment):
    fig = go.Figure(go.Indicator(
        mode = "gauge",
        gauge = {
            'axis': {
                'range': [-1, 1],
                'dtick': 1}},
        value = pos_extract_sentiment,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment analysis"}))

    fig.add_annotation(
        x=0.5, y=0.5,  # Center of the plot
        text="😊",
        font=dict(size=60),  # Adjust size as needed
        showarrow=False
    )

    fig.update_layout(hovermode=False,dragmode=False)

    config = {
    'displaylogo': False,
    'responsive': False,
    'modeBarButtonsToRemove': ['sendDataToCloud']
}

    return pio.to_html(fig, full_html=False, config=config)

def neg_extract_figure(neg_extract_sentiment):
    fig = go.Figure(go.Indicator(
        mode = "gauge",
        gauge = {
            'axis': {
                'range': [-1, 1],
                'dtick': 1}},
        value = neg_extract_sentiment,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment analysis"}))

    fig.add_annotation(
        x=0.5, y=0.5,  # Center of the plot
        text="😡",
        font=dict(size=60),  # Adjust size as needed
        showarrow=False
    )

    fig.update_layout(hovermode=False,dragmode=False)

    config = {
    'displaylogo': False,
    'responsive': False,
    'modeBarButtonsToRemove': ['sendDataToCloud']
}

    return pio.to_html(fig, full_html=False, config=config)
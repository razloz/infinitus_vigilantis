HOME_HEAD = """
<!DOCTYPE html>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<html style="color:#fff;background-color:#000;">
    <head>
        <style>
            iframe {
                border: 0px;
                padding: 0px;
            }
            button {
                height: 35px;
                width: 175px;
                padding: 5px;
                color: #EEE;
                background-color: #444;
                border-color: #222;
            }
        </style>
    </head>
"""

HOME_BODY = """
    <body>
        <script>
            function changeView(features, metrics, chart, forecast) {
                document.getElementById("viewport_features").src=features;
                document.getElementById("viewport_metrics").src=metrics;
                document.getElementById("viewport_chart").src=chart;
                document.getElementById("viewport_forecast").src=forecast;
            }
        </script>
        <div style="display:grid;gap:5px;">
            <div style="grid-column:1/1;grid-row:1/9;height:850px;width:100%;overflow:scroll;">
"""

HOME_FOOT = """
            </div>
            <div style="grid-column:2/20;grid-row:1/6;">
                <iframe width="30%" height="700px" src="" id="viewport_features"></iframe>
                <img width="60%" height="700px" src="" id="viewport_chart"></img>
            <div style="grid-column:2/20;grid-row:6/9;">
                <iframe width="30%" height="auto" src="" id="viewport_metrics"></iframe>
                <img width="60%" height="auto" src="" id="viewport_forecast"></img>
            </div>
        </div>
    </body>
</html>
"""

COMPASS_BUTTON = """
                <button onClick="changeView({0}, {1}, {2}, {3})"><b>{4}</b></button><br>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html style="color:#fff;background-color:#111;">
    <body>
        <div style="width:100%;height:100%;overflow:scroll;display:grid;">
            {0}
        </div>
    </body>
</html>
"""

def build_features(features, data):
    label_string = """<div style="padding:5px;grid-column:1/1;"><h3>"""
    value_string = """<div style="padding:5px;grid-column:2/10;"><h3>"""
    for index, label in enumerate(features):
        label_string += f'{str(label).upper()}:<br>'
        value_string += f'{round(data[index], 4)}<br>'
    label_string += """</h3></div>"""
    value_string += """</h3></div>"""
    return HTML_TEMPLATE.format(label_string + value_string)


def build_metrics(metrics):
    label_string = """<div style="padding:5px;grid-column:1/1;">"""
    value_string = """<div style="padding:5px;grid-column:2/10;">"""
    for key, value in metrics.items():
        label_string += f'{key}:<br>'.upper()
        value_string += f'{value}<br>'
    label_string += """</div>"""
    value_string += """</div>"""
    return HTML_TEMPLATE.format(label_string + value_string)


def generate_compass(symbols, features, candelabrum, metrics):
    files = dict()
    for index, symbol in enumerate(symbols):
        data = candelabrum[-1][index].tolist()
        files[f'{symbol}_features.html'] = build_features(features, data)
        files[f'{symbol}_metrics.html'] = build_metrics(metrics[index])
    return files


def make_compass(symbols):
    COMPASS = list()
    for symbol in symbols:
        COMPASS.append(
            COMPASS_BUTTON.format(
                """'{}_features.html'""".format(symbol),
                """'{}_metrics.html'""".format(symbol),
                """'./charts/{}_market.png'""".format(symbol),
                """'./charts/{}_forecast.png'""".format(symbol),
                """{}""".format(symbol),
            )
        )
    return ''.join(COMPASS)


def build(symbols, features, candelabrum, metrics):
    cabinet = dict()
    HTML_DOC = [HOME_HEAD, HOME_BODY, make_compass(symbols), HOME_FOOT]
    cabinet['home.html'] = ''.join(HTML_DOC)
    compass_files = generate_compass(symbols, features, candelabrum, metrics)
    cabinet.update(compass_files)
    return cabinet

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'
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
            }
            function quickJump() {
                let symbol = document.getElementById("symbol_select").value.toUpperCase();
                let features = symbol + "_features.html";
                let metrics = symbol + "_metrics.html";
                let chart = "./charts/" + symbol + "_market.png";
                changeView(features, metrics, chart);
            }
        </script>
        <div style="display:grid;gap:5px;">
"""

HOME_FOOT = """
            <div style="grid-column:2/2;grid-row:1/20;height:auto;width:100%;overflow:scroll;">
                <div style="display:grid;gap:0px;">
                    <div style="grid-column:1/1;grid-row:1/1;">
                        <b><i>INDIVIDUAL METRICS</i></b>
                        <iframe width="100%" height="auto" src="{1}" id="viewport_metrics"></iframe>
                    </div>
                    <div style="grid-column:1/1;grid-row:2/7;">
                        <b><i>SYMBOL FEATURES</i></b>
                        <iframe width="100%" height="550px" src="{0}" id="viewport_features"></iframe>
                    </div>
                    <div style="grid-column:1/1;grid-row:8/8;">
                        <b><i>TOTAL VALIDATION RESULTS</i></b>
                        <iframe width="100%" height="auto" src="validation_results.html" id="viewport_validation"></iframe>
                    </div>
                </div>
            </div>
            <div style="grid-column:3/3;grid-row:1/20;height:100%;width:100%;overflow:scroll;">
                <div style="display:grid;gap:0px;">
                    <div style="grid-column:1/1;grid-row:1/1;">
                        <img width="100%" height="900px" src="{2}" id="viewport_chart"></img>
                    </div>
                </div>
            </div>
        </div>
    </body>
</html>
"""

COMPASS_SELECT_HEAD = """
            <div style="grid-column:1/1;grid-row:1/20;height:auto;width:100%;overflow:scroll;">
                <div style="display:grid;gap:0px;">
                    <div style="grid-column:1/1;grid-row:1/2;height:auto;width:100%;overflow:scroll;">
                        <b>Quick Jump:</b><br>
                        <select style="width:125px;height:35px;" id="symbol_select">
                            <option value="" selected></option>
"""
COMPASS_OPTION = """
                            <option value="{0}">{1}</option>
"""
COMPASS_SELECT_TAIL = """
                        </select>
                        <button style="width:50px;" onClick="quickJump()"><b>Go</b></button><br><br>
                    </div>
"""

COMPASS_PICKS_HEAD = """
                    <div style="grid-column:1/1;grid-row:2/9;height:auto;width:100%;overflow:scroll;">
                        <b>Top Picks:</b><br>
"""
COMPASS_BUTTON = """
                        <button onClick="changeView({0}, {1}, {2})"><b>{3}</b></button><br>
"""
COMPASS_PICKS_TAIL = """
                    </div>
                </div>
            </div>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html style="color:#fff;background-color:#000;">
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
        value_string += f'{round(float(data[-1, index]), 4)}<br>'
    label_string += """</h3></div>"""
    value_string += """</h3></div>"""
    return HTML_TEMPLATE.format(label_string + value_string)


def build_metrics(metrics):
    label_string = """<div style="padding:5px;grid-column:1/1;"><h3>"""
    value_string = """<div style="padding:5px;grid-column:2/10;"><h3>"""
    for key, value in metrics.items():
        if key in ['forecast', 'target']:
            continue
        label_string += f'{key}:<br>'.upper()
        if key == 'accuracy':
            value_string += f'{value}%<br>'
        else:
            value_string += f'{value}<br>'
    label_string += """</h3></div>"""
    value_string += """</h3></div>"""
    return HTML_TEMPLATE.format(label_string + value_string)


def generate_compass(symbols, features, candelabrum, metrics):
    files = dict()
    for index, (symbol, candles) in enumerate(candelabrum.items()):
        if symbol in ('QQQ', 'SPY'):
            continue
        files[f'{symbol}_features.html'] = build_features(features, candles)
        files[f'{symbol}_metrics.html'] = build_metrics(metrics[index])
    files['validation_results.html'] = build_metrics(metrics['validation.metrics'])
    return files


def make_compass(symbols, picks):
    COMPASS = list()
    COMPASS.append(COMPASS_SELECT_HEAD)
    for symbol in symbols:
        COMPASS.append(COMPASS_OPTION.format(symbol, symbol))
    COMPASS.append(COMPASS_SELECT_TAIL)
    COMPASS.append(COMPASS_PICKS_HEAD)
    for symbol in picks:
        COMPASS.append(
            COMPASS_BUTTON.format(
                """'{}_features.html'""".format(symbol),
                """'{}_metrics.html'""".format(symbol),
                """'./charts/{}_market.png'""".format(symbol),
                """{}""".format(symbol),
            )
        )
    COMPASS.append(COMPASS_PICKS_TAIL)
    return ''.join(COMPASS)


def build(symbols, features, candelabrum, metrics, picks):
    cabinet = dict()
    COMPASS = make_compass(symbols, picks)
    default_symbol = picks[0]
    FOOTER = HOME_FOOT.format(
        f'{default_symbol}_features.html',
        f'{default_symbol}_metrics.html',
        f'./charts/{default_symbol}_market.png',
    )
    cabinet['home.html'] = ''.join([HOME_HEAD, HOME_BODY, COMPASS, FOOTER])
    compass_files = generate_compass(symbols, features, candelabrum, metrics)
    cabinet.update(compass_files)
    return cabinet

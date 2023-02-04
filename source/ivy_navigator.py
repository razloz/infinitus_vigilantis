RESOURCE_PATH = './resources'
HTML_HEAD = """
<!DOCTYPE html>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<html style="background-image: url('background.png');">
    <head>
        <style>
            ul {
                position: fixed;
                top: 1px;
                left: 1px;
                right: 1px;
                width: 100%;
                list-style-type: none;
                margin: 1px;
                padding: 1px;
                overflow: hidden;
                background-color: #333;
            }

            li {
                float: left;
            }

            li a {
                display: block;
                color: white;
                text-align: center;
                padding: 14px 16px;
                text-decoration: none;
            }

            li a:hover {
                background-color: #111;
            }

            .active {
                background-color: #04AA6D;
            }
        </style>
        <div>
            <ul>
                <li><a href="top_25.html">Top 25</a></li>
                <li><a href="trading.html">Trading</a></li>
                <li><a href="compass.html">Compass</a></li>
                <li style="float:right"><a href="info.html">Info</a></li>
                <li style="float:right"><a href="disclaimer.html">Disclaimer</a></li>
            </ul>
        </div>
    </head>
    <body>
        <div>
"""
HTML_TAIL = """
        </div>
    </body>
</html>
"""


def make_html(file_name, html):
    html_doc = str(HTML_HEAD)
    html_doc += html
    html_doc += HTML_TAIL


def make_home():
    file_name = RESOURCE_PATH + '/home.html'
    html = """"""
    make_html(file_name, html)


def make_top_25():
    file_name = RESOURCE_PATH + '/top_25.html'
    html = """"""
    make_html(file_name, html)


def make_trading():
    file_name = RESOURCE_PATH + '/trading.html'
    html = """"""
    make_html(file_name, html)


def make_compass():
    file_name = RESOURCE_PATH + '/compass.html'
    html = """"""
    make_html(file_name, html)


def make_metrics():
    file_name = RESOURCE_PATH + '/metrics.html'
    html = """"""
    make_html(file_name, html)


def make_info():
    file_name = RESOURCE_PATH + '/info.html'
    html = """"""
    make_html(file_name, html)


def make_disclaimer():
    file_name = RESOURCE_PATH + '/disclaimer.html'
    html = """"""
    make_html(file_name, html)

def make_all(inscribed_candles):
    for inscription in inscribed_candles:
        pass

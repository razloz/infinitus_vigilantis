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
            div.gallery {
                margin: auto 0;
                border: 1px solid #ccc;
                float: left;
                width: 30%;
                height: auto;
                }
            div.gallery:hover {
                border: 1px solid #777;
                }
            div.gallery img {
                width: 100%;
                height: auto;
                }
            div.desc {
                padding: 15px;
                text-align: center;
                }
            .metrics {
                color: rgb(55, 95, 55);
                background-color: rgb(13, 13, 13);
                text-align: center;
                padding-top: 45px;
                margin: 0 auto;
                max-width: 100%;
                height: auto;
                }
            .cartography {
                width: 100%;
                height: auto;
                }
        </style>
    </head>
    <body>
        <div>
            <ul>
                <li><a href="candelabrum.html">Candelabrum</a></li>
                <li><a href="forecast.html">Forecast</a></li>
                <li style="float:right"><a href="info.html">Info</a></li>
                <li style="float:right"><a href="disclaimer.html">Disclaimer</a></li>
            </ul>
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
    with open(file_name, 'w+') as f:
        f.write(html_doc)


def make_candelabrum(metrics):
    file_name = RESOURCE_PATH + '/candelabrum.html'
    html = """      <div class="metrics"><h2>\n"""
    for key, value in metrics.items():
        html += f"""          <p>{key}: {value}</p>\n"""
    html += """            <img class="cartography" src="candelabrum.png"></img>
            <img class="cartography" src="signal_gradient.png"></img>
            </h2></p></div></p>"""
    make_html(file_name, html)


def make_forecast(forecast):
    file_name = RESOURCE_PATH + '/forecast.html'
    html = """"""
    for day, probs in enumerate(forecast):
        img_name = f'forecast_{day}.png'
        symbol = probs[0]
        html += """
            <div class="gallery">
                <a target="_blank" href="{0}">
                    <img src="{0}" alt="{1}" width="1920" height="1080">
                </a>
                <div class="desc">{1}</div>
            </div>
            """.format(img_name, symbol)
    make_html(file_name, html)


def make_info(info):
    file_name = RESOURCE_PATH + '/info.html'
    html = f"""<div class="metrics"><p><b>{info}</b></p></div>"""
    make_html(file_name, html)


def make_disclaimer(disclaimer):
    file_name = RESOURCE_PATH + '/disclaimer.html'
    html = f"""<div class="metrics"><p><b>{disclaimer}</b></p></div>"""
    make_html(file_name, html)


def make_all(metrics, forecast, info, disclaimer):
    for day, probs in enumerate(forecast):
        metrics[f'day_{day}_prob'] = f'{probs[0]} ({probs[1]})'
    make_candelabrum(metrics)
    make_forecast(forecast)
    make_info(info)
    make_disclaimer(disclaimer)
    return True

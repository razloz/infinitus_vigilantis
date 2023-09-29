from toga import Box
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from toga import WebView
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
    <meta charset="utf-8">
    <head>
        <style>
            div {
                width: 100%;
                height: 100%;
                color: #000;
                background-color: #FFF;
            }
        </style>
    </head>
'''
HTML_ADDENDUM = '''
    <body>
        <div>
            {}
        </div>
    </body>
</html>
'''

class LogView(WebView):
    """
    HTML-based log viewer widget.
    """
    def __init__(self, root_app):
        """
        Create the WebView.
        """
        super().__init__(id='log_viewer')
        self.root_app = root_app
        self.log_path = str(root_app.log_path)
        self.build_view()

    def build_view(self):
        """
        Construct HTML document for asynchronous viewing of the log file.
        """
        html_doc = str(HTML_TEMPLATE)
        with open(self.log_path, 'r') as log_file:
            log_data = log_file.read()
        html_doc += HTML_ADDENDUM.format(log_data)
        self.set_content(None, html_doc)

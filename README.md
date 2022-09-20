# infinitus_vigilantis
Infinitus Vigilantis for Python

Requirements:
  - Alpaca Markets API keys

Dependencies:
  - pytorch
  - matplotlib
  - numpy
  - pandas
  - pillow
  - requests
  - PyGObject (venv instructions @ https://pygobject.readthedocs.io/en/latest/devguide/dev_environ.html)

Usage:
  - with IVY as an absolute path to infinitus vigilantis folder
  - $IVY/updater.py --build
  - $IVY/updater.py --research
  - $IVY/cartographer.py --all --size 160 --start_date 2020-01-01

Systemd Daemon:
  - Shell script, timer, and service located in daemon folder.
  - Don't forget to set key, secret key, and path in the .service file


Disclaimer:
    The information provided by Infinitus Vigilantis for Python (the
"Software") and accompanying material is for informational purposes
only. It should not be considered legal or financial advice. You should
consult with an attorney or other professional to determine what may be
best for your individual needs. Daniel Ward (the "Author") does not
make any guarantee or other promise as to any results that may be
obtained from using the Software. No one should make any investment
decision without first consulting their own financial advisor and
conducting their own research and due diligence. To the maximum extent
permitted by law, the Author disclaims any and all liability in the
event any information, commentary, analysis, opinions, advice and/or
recommendations prove to be inaccurate, incomplete or unreliable, or
result in any investment or other losses. Content contained on or made
available through the Software is not intended to and does not
constitute legal advice or investment advice and no attorney-client
relationship is formed. Your use of the information from the Software
or materials linked and/or obtained from the Software is at your own
risk. Past performance is not a guarantee of future return, nor is it
necessarily indicative of future performance. Keep in mind investing
involves risk. The value of your investment will fluctuate over time
and you may gain or lose money.

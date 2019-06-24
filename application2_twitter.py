# Application 2:      Twitter

# download the files twurl.py, hidden.py, oauth.py, and twitter1.py
# from www.py4e.com/code and put them all in a folder on your computer.

# To make use of these programs, need to have a Twitter account, and authorize
# your Python code as an application, set up a key, secret, token and
# token secret. You will edit the file hidden.py and put these four strings
# into the appropriate variables in the file:

# Keep this file separate
# https://apps.twitter.com/
# Create new App and get the four strings
def oauth():
    return {"consumer_key": "h7Lu...Ng",
            "consumer_secret": "dNKenAC3New...mmn7Q",
            "token_key": "10185562-eibxCp9n2...P4GEQQOSGI",
            "token_secret": "H0ycCFemmC4wyf1...qoIpBo"}

# The Twitter web service are accessed using a URL like this:
# https://api.twitter.com/1.1/statuses/user_timeline.json

#But once all of the security information has been added, the URL will look like

#https://api.twitter.com/1.1/statuses/user_timeline.json?count=2
#&oauth_version=1.0&oauth_token=101...SGI&screen_name=drchuck
#&oauth_nonce=09239679&oauth_timestamp=1380395644
#&oauth_signature=rLK...BoD&oauth_consumer_key=h7Lu...GNg
#&oauth_signature_method=HMAC-SHA1

# You can read the OAuth specification if you want to know more about the
# meaning of the various parameters that are added to meet the security
# requirements of OAuth.

# For the programs we run with Twitter, we hide all the complexity in the files
# oauth.py and twurl.py. We simply set the secrets in hideen.py and then send
# the desired URL to the twurl.augment() function and the library code adds all
# the necessary parameters to the URL for us.

# This program retrieves the timeline for a particular Twitter user and returns
# it to us in JSON format in a string. We simply print the first 250 characters
# of the string:

import urllib.request, urllib.parse, urllib.error
import twurl
import ssl

# https://apps.twitter.com/
# Create App and get the four strings, put them in hidden.py

TWITTER_URL = 'https://api.twitter.com/1.1/statuses/user_timeline.json'

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

while True:
    print('')
    acct = input('Enter Twitter Account:')
    if (len(acct) < 1): break
    url = twurl.augment(TWITTER_URL,
                        {'screen_name': acct, 'count': '2'})
    print('Retrieving', url)
    connection = urllib.request.urlopen(url, context=ctx)
    data = connection.read().decode()
    print(data[:250])
    headers = dict(connection.getheaders())
    # print headers
    print('Remaining', headers['x-rate-limit-remaining'])
# Code: http://wwwpy4e.com/code3/twitter1.py

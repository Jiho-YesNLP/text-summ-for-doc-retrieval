""" UTS (UMLS Terminology Services) API client
REST documentation: https://documentation.uts.nlm.nih.gov/rest/home.html
"""
import os
import json
import requests
from pathlib import Path
from lxml.html import fromstring


class UtsClient:
    """All the UTS REST API requests are handled through this client"""
    apikey_file = Path(__file__).resolve().parent / 'uts-api-key.txt'

    def __init__(self):
        if not os.path.exists(self.apikey_file):
            raise RuntimeError("API key file not exists [{}]"
                               .format(self.apikey_file))
        self.apikey = open(self.apikey_file).read().rstrip()
        self.service = "http://umlsks.nlm.nih.gov"
        self.headers = {
            "Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain",
            "User-Agent": "python"
        }
        self.tgt = None
        self.base_uri = "https://uts-ws.nlm.nih.gov"
        self.version = "current"

    def gettgt(self):
        """Retrieve a ticket granting ticket"""
        auth_uri = "https://utslogin.nlm.nih.gov"
        params = {"apikey": self.apikey}
        auth_endpoint = "/cas/v1/api-key"
        r = requests.post(auth_uri + auth_endpoint, data=params,
                          headers=self.headers)
        response = fromstring(r.text)
        # extract the entire URL needed from the HTML form (action attribute)
        # returned - looks similar to
        # https://utslogin.nlm.nih.gov/cas/v1/tickets/TGT-36471-aYqNLN2rFIJPXKzxwdTNC5ZT7z3B3cTAKfSc5ndHQcUxeaDOLN-cas
        # we make a POST call to this URL in the getst method
        self.tgt = response.xpath('//form/@action')[0]

    def getst(self):
        """Request a single-use service ticket"""
        if self.tgt is None:
            self.gettgt()
        params = {"service": self.service}
        r = requests.post(self.tgt, data=params, headers=self.headers)
        return r.text

    def query_get(self, endpoint, query):
        r = requests.get(self.base_uri+endpoint, params=query)
        if r.status_code == 404:
            return
        return json.loads(r.text)

    def get_concept_mesh_atoms(self, cui, tok):
        endpoint = "/rest/content/{}/CUI/{}/atoms".format(self.version, cui)
        params = {"ticket": self.getst(), "sabs": "MSH"}
        rst = self.query_get(endpoint, params)
        # If MSH term can't be found by CUI, use the defining word to search
        if rst is None:
            endpoint = "/rest/search/{}".format(self.version)
            params = {"ticket": self.getst(),
                      "string": tok, "searchType": "exact"}
            rst = self.query_get(endpoint, params)
            if rst is not None:
                search_max = 5
                for rec in rst['result']['results']:
                    # search MSH until found
                    endpoint = ("/rest/content/{}/CUI/{}/atoms"
                                "".format(self.version, rec['ui']))
                    params = {"ticket": self.getst(), "sabs": "MSH"}
                    rst = self.query_get(endpoint, params)
                    if rst is not None:
                        return rst
                    else:
                        search_max -= 1
                    if search_max == 0:
                        return None
        return rst

    def get_mesh_by_term_search(self, phrase):
        endpoint = "/rest/search/current"
        params = {
            'string': phrase,
            'returnIdType': 'sourceUi',
            'ticket': self.getst(),
            'sabs': 'MSH',
            'language': 'ENG',
            'pageSize': 10
        }
        rst = self.query_get(endpoint, params)
        if rst is not None:
            for cpt in rst['result']['results']:
                if cpt['ui'].startswith('D'):  # descriptor
                    return cpt
            return rst['result']['results'][0]
        return None

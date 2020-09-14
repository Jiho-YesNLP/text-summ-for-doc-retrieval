"""index_docs_solr

Solr core for Pubmed must be created with the name 'pubmed20'
Index documents into a Solr core"""

import gzip
import os
from lxml import etree as et
import pysolr

# client instance
solr = pysolr.Solr('http://localhost:8983/solr/pubmed20', always_commit=True)


def parse_xml(xml):
    docs = []
    articles = et.parse(gzip.open(xml))
    for a in articles.iterfind('PubmedArticle'):
        doc = {
            "id":
                a.findtext('.//PMID'),
            "ArticleTitle":
                a.findtext('.//ArticleTitle'),
            "AbstractText":
                ' '.join([p.text for p in a.findall('.//Abstract/AbstractText')
                          if p is not None and p.text is not None]),
            "JournalTitle":
                a.findtext('.//Journal/Title'),
            "JournalISOAbbreviation":
                a.findtext('.//Journal/ISOAbbreviation'),
            "Keyword":
                [k.text for k in a.findall('.//KeywordList/Keyword')],
            "MeshDescriptorUI":
                [m.get('UI') for m in
                 a.findall('.//MeshHeadingList/MeshHeading/DescriptorName')],
            "MeshDescriptorName":
                [m.text for m in
                 a.findall('.//MeshHeadingList/MeshHeading/DescriptorName')],
        }
        docs.append(doc)
    return docs


dir_docs = 'data/pubmed'
for n in range(1, 1016):
    pm_f = f'pubmed20n{n:04}.xml.gz'
    docs = parse_xml(os.path.join(dir_docs, pm_f))
    solr.add(docs)
    print(f'#{n}/1016: POSTing {len(docs)} docs from {pm_f}')

#!/usr/bin/env python
import re

import Levenshtein
import pymongo


def code_tokenize(line):
    tokens = re.findall(r"[^ \t\n\r\f\v_\W]+|[^a-zA-Z0-9_ ]", line)
    return tokens


class SearchEngine(object):
    def __init__(self, data_name, text_content):
        self.data_name = data_name
        self.text_content = text_content
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["recitation"]
        if data_name not in db.list_collection_names():
            col = db.create_collection(data_name)
            col.create_index([("code", "text")])
            self.load_text(text_content)
        self.col = db.get_collection(data_name)

    def load_text(self, text_content):
        col = self.col
        for idx, line in enumerate(text_content.splitlines()):
            line = line.strip()
            if len(line) <= 1:
                continue
            tokens = code_tokenize(line)
            joined_tokens = " ".join(tokens)
            col.insert_one({"line": idx, "code": joined_tokens})

    def search(self, query_line):
        tokens = code_tokenize(query_line)
        query = " ".join(tokens)
        r = self.col.find({"$text": {"$search": query}}).sort([("score", {"$meta": 'textScore'})]).limit(10)
        answers = []
        for i, item in enumerate(r):
            item_code = item["code"]
            item_dist = Levenshtein.distance(query_line, item_code)
            # print(f"item {i} (dist={item_dist}): {item_code}")
            answers.append((item_code, item_dist))
        return answers


if __name__ == "__main__":
    engine = SearchEngine("linux", None)
    query = "schedstat_add(cfs_rq1, exec_clock2, delta_exec3);"
    print(f"searching for {query}")
    print(engine.search(query))


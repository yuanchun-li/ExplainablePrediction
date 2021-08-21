#!/usr/bin/env python


class SearchEngine(object):
    def search(self, query):
        pass


class TextSearchEngine(SearchEngine):
    def __init__(self, text_content):
        self.text = text_content
        self.lines = text_content.splitlines()

    def search(self, query):
        answers = []
        return answers

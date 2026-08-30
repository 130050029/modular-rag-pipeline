"""Query processing and rewriting for Phase B query intelligence."""

from typing import Protocol

import requests
import config

from rag.query.rewriting import OllamaQueryRewriter, QueryRewriter
from rag.query.expansion import OllamaQueryExpander, QueryExpander


class QueryProcessor(Protocol):
    def process(self, query: str) -> list[str]:
        """Return one or more retrieval queries derived from the user query."""
        ...

class DefaultQueryProcessor:
    """Configurable query-processing pipeline.

    Currently supports optional single-query rewriting.
    Future Phase B strategies such as expansion and decomposition
    will be added here rather than creating separate processing paths.
    """

    def __init__(
        self,
        rewriter: QueryRewriter | None = None,
        expander: QueryExpander | None = None,
    ):
        self.rewriter = rewriter
        self.expander = expander

    def process(self, query: str) -> list[str]:
        if self.rewriter is not None:
            query = self.rewriter.rewrite(query)

        if self.expander is not None:
            return self.expander.expand(query)

        return [query]



def get_query_processor() -> QueryProcessor:
    rewriter = (
        OllamaQueryRewriter()
        if config.QUERY_REWRITE_ENABLED
        else None
    )

    expander = (
        OllamaQueryExpander()
        if config.QUERY_EXPANSION_ENABLED
        else None
    )

    return DefaultQueryProcessor(
        rewriter=rewriter,
        expander=expander,
    )
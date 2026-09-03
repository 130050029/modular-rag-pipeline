"""Query processing orchestration for Phase B query intelligence."""

from typing import Protocol
import config

from rag.query.decomposition import (
    OllamaQueryDecomposer,
    QueryDecomposer,
)
from rag.query.expansion import (
    OllamaQueryExpander,
    QueryExpander,
)
from rag.query.rewriting import (
    OllamaQueryRewriter,
    QueryRewriter,
)
from rag.query.routing import (
    DefaultQueryComplexityRouter,
    QueryComplexityRouter,
)


class QueryProcessor(Protocol):
    def process(self, query: str) -> list[str]:
        """Return one or more retrieval queries derived from the user query."""
        ...


class DefaultQueryProcessor:
    """Configurable query-processing pipeline.

    Query processing follows one of two paths selected by the
    complexity router:

        simple:
            rewrite -> expand

        complex:
            decompose -> expand

    Rewrite and decomposition are alternative first-stage strategies.
    Expansion is a second-stage strategy that may operate on the output
    of either strategy.

    The complexity router is mandatory so that every query passes through
    the same routing decision.
    """

    def __init__(
        self,
        complexity_router: QueryComplexityRouter,
        rewriter: QueryRewriter | None = None,
        expander: QueryExpander | None = None,
        decomposer: QueryDecomposer | None = None,
    ):
        self.complexity_router = complexity_router
        self.rewriter = rewriter
        self.expander = expander
        self.decomposer = decomposer

    def process(self, query: str) -> list[str]:
        """Route and process the query through the configured strategy chain."""

        is_complex = self.complexity_router.is_complex(query)

        # ---------------------------------------------------------------
        # First stage: rewrite OR decompose
        # ---------------------------------------------------------------
        if is_complex:
            if self.decomposer is not None:
                queries = self.decomposer.decompose(query)
            else:
                queries = [query]
        else:
            if self.rewriter is not None:
                queries = [self.rewriter.rewrite(query)]
            else:
                queries = [query]

        # ---------------------------------------------------------------
        # Second stage: expansion
        # ---------------------------------------------------------------
        if self.expander is not None:
            expanded_queries = []

            for processed_query in queries:
                expanded_queries.extend(
                    self.expander.expand(processed_query)
                )

            queries = expanded_queries

        return queries


def get_query_processor() -> QueryProcessor:
    """Build the configured query-processing pipeline.

    The complexity router always participates in processing.

    Simple queries:
        rewrite -> expansion

    Complex queries:
        decomposition -> expansion

    Whether rewriting, decomposition, or expansion is enabled is controlled
    by the corresponding configuration flags.
    """

    router = DefaultQueryComplexityRouter()

    rewriter = (
        OllamaQueryRewriter()
        if config.QUERY_REWRITE_ENABLED
        else None
    )

    decomposer = (
        OllamaQueryDecomposer()
        if config.QUERY_DECOMPOSITION_ENABLED
        else None
    )

    expander = (
        OllamaQueryExpander()
        if config.QUERY_EXPANSION_ENABLED
        else None
    )

    return DefaultQueryProcessor(
        complexity_router=router,
        rewriter=rewriter,
        decomposer=decomposer,
        expander=expander,
    )
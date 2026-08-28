"""Pure-logic tests for factory_floor/rag.py helpers.

`contextualize_question` and `rerank_documents` take an `llm=` — a fake is passed, so
nothing here reaches the network.
"""

from langchain_core.messages import AIMessage, HumanMessage

from factory_floor.rag import (
    build_chat_history,
    contextualize_question,
    format_context,
    rerank_documents,
    source_list,
)


class TestFormatContext:
    def test_numbers_sources_from_one_and_bumps_page_to_one_based(self, make_doc):
        docs = [
            make_doc(content="ground fault text", source_file="ListManual.pdf", page=907),
            make_doc(content="fan fault text", source_file="ListManual.pdf", page=915),
        ]
        out = format_context(docs)
        assert "[SOURCE 1] ListManual.pdf, page 908" in out
        assert "[SOURCE 2] ListManual.pdf, page 916" in out
        assert "ground fault text" in out

    def test_missing_metadata_falls_back(self):
        from langchain_core.documents import Document

        out = format_context([Document(page_content="x", metadata={})])
        assert "[SOURCE 1] unknown, page 1" in out


class TestSourceList:
    def test_bulleted_one_based_pages(self, make_doc):
        out = source_list([make_doc(source_file="M.pdf", page=41)])
        assert out == "- [SOURCE 1] M.pdf — page 42"


class TestBuildChatHistory:
    def test_replays_question_and_answer_as_messages(self):
        turns = [{"question": "why is F30021 tripping?", "answer": "earth fault suspected"}]
        msgs = build_chat_history(turns)
        assert [type(m) for m in msgs] == [HumanMessage, AIMessage]
        assert msgs[0].content == "why is F30021 tripping?"
        assert msgs[1].content == "earth fault suspected"

    def test_folds_vision_context_into_the_human_turn(self):
        turns = [
            {
                "question": "is this cable damaged?",
                "answer": "looks abraded",
                "vision_context": "Vision analysis: predicted condition = structural_damage",
            }
        ]
        human = build_chat_history(turns)[0]
        assert "is this cable damaged?" in human.content
        assert "structural_damage" in human.content

    def test_photo_only_turn_has_no_empty_human_message(self):
        turns = [{"question": "", "answer": "a", "vision_context": "Vision analysis: good"}]
        human = build_chat_history(turns)[0]
        assert "Vision analysis: good" in human.content

    def test_respects_max_turns(self):
        turns = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)]
        msgs = build_chat_history(turns, max_turns=2)
        assert len(msgs) == 4
        assert msgs[0].content == "q8"


class TestContextualizeQuestion:
    def test_empty_history_returns_question_verbatim_without_calling_llm(self):
        # No llm passed and _no_network is active — if this tried to build one it would raise.
        assert contextualize_question("what about F30021?", []) == "what about F30021?"

    def test_with_history_uses_the_llm_and_strips(self, make_fake_llm):
        llm = make_fake_llm("  standalone rewrite  ")
        out = contextualize_question(
            "and then?", [HumanMessage(content="q"), AIMessage(content="a")], llm=llm
        )
        assert out == "standalone rewrite"


class TestRerankDocuments:
    def test_short_pool_is_returned_unchanged_without_calling_llm(self, make_doc):
        docs = [make_doc(content=f"d{i}") for i in range(3)]
        assert rerank_documents("q", docs, top_n=5) == docs

    def test_reorders_by_the_llm_ranking(self, make_fake_llm, make_doc):
        docs = [make_doc(content=f"d{i}") for i in range(6)]
        llm = make_fake_llm("2,0,1,3,4,5")
        ranked = rerank_documents("q", docs, llm=llm, top_n=3)
        assert [d.page_content for d in ranked] == ["d2", "d0", "d1"]

    def test_fails_open_to_original_order_on_garbage_output(self, make_fake_llm, make_doc):
        docs = [make_doc(content=f"d{i}") for i in range(6)]
        llm = make_fake_llm("the model refused to answer")
        ranked = rerank_documents("q", docs, llm=llm, top_n=3)
        assert [d.page_content for d in ranked] == ["d0", "d1", "d2"]

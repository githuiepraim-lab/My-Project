import json

import pytest
from memory import memory_manager as M


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MEMORY_PATH", tmp_path / "long_term.json")
    monkeypatch.setattr(M, "PENDING_PATH", tmp_path / "pending.json")
    monkeypatch.setattr(M, "_memory_changed", lambda: None)
    return M


def seed(mem):
    mem.update_memory({
        "identity": {"name": {"value": "Ephraim"}, "city": {"value": "Machakos"}},
        "relationships": {"sister_name": {"value": "Wanjiru, studies nursing"},
                          "best_friend": {"value": "Kamau who runs a barbershop"}},
        "projects": {"drip_closet": {"value": "Kenyan streetwear brand, new hoodie drop in May"},
                     "arcade": {"value": "NextGen Gaming Arcade near the university"}},
        "preferences": {"answer_style": {"value": "short answers, no preamble"}},
    })


def test_ranked_recall_finds_by_stem_and_prefix(mem):
    seed(mem)
    assert mem.search_entries("my sisters")[0]["key"] == "sister_name"
    assert mem.search_entries("hoodies")[0]["key"] == "drip_closet"
    assert mem.search_entries("arca")[0]["key"] == "arcade"         # prefix (4+ chars)
    assert mem.search_entries("barber")[0]["key"] == "best_friend"  # value match


def test_key_match_outranks_value_mention(mem):
    mem.update_memory({"notes": {"cat": {"value": "likes sleeping"},
                                 "diary": {"value": "walked past a cat today, also a cat later"}}})
    assert mem.search_entries("cat")[0]["key"] == "cat"


def test_unrelated_query_returns_nothing(mem):
    seed(mem)
    assert mem.search_entries("quantum chromodynamics") == []
    assert mem.search_memory("quantum chromodynamics").startswith("Nothing stored")


def test_search_memory_output_format_unchanged(mem):
    seed(mem)
    out = mem.search_memory("sister")
    assert out.startswith("Stored facts matching 'sister':") and "relationships/sister name:" in out


def test_correction_keeps_history_and_replaces_value(mem):
    mem.update_memory({"identity": {"city": {"value": "Nairobi"}}})
    mem.update_memory({"identity": {"city": {"value": "Machakos"}}})
    assert mem.load_memory()["identity"]["city"]["value"] == "Machakos"
    assert mem.history_of("city", "identity")[0]["value"] == "Nairobi"
    assert "Nairobi" not in mem.format_memory_for_prompt(mem.load_memory())   # old fact never shown


def test_duplicate_under_new_key_is_not_a_new_entry(mem):
    mem.update_memory({"preferences": {"answer_style": {"value": "short answers, no preamble"}}})
    mem.update_memory({"preferences": {"style_of_answers": {"value": "Short answers, no preamble!"}}})
    assert list(mem.load_memory()["preferences"]) == ["answer_style"]


def test_edit_and_delete(mem):
    seed(mem)
    assert mem.edit_entry("identity", "city", value="Kitui").startswith("Updated")
    assert mem.load_memory()["identity"]["city"]["value"] == "Kitui"
    assert mem.history_of("city", "identity")[0]["value"] == "Machakos"
    mem.edit_entry("identity", "city", new_key="home_town")
    assert "home_town" in mem.load_memory()["identity"]
    assert mem.delete_entry("home_town", "identity").startswith("Forgotten")
    assert mem.edit_entry("identity", "nope", value="x").startswith("Not found")


def test_learning_is_pending_until_approved(mem):
    added = mem.observe("I prefer answers in bullet points when I ask for comparisons.")
    assert added and mem.load_memory()["preferences"] == {}           # NOT stored yet
    assert mem.list_pending()
    pid = mem.list_pending()[0]["id"]
    assert "Remembered" in mem.approve_pending(pid)
    assert any("bullet points" in v["value"] for v in mem.load_memory()["preferences"].values())
    assert mem.list_pending() == []


def test_rejected_learning_is_never_stored(mem):
    mem.observe("From now on call every meeting a sync, please")
    pid = mem.list_pending()[0]["id"]
    mem.reject_pending(pid)
    assert mem.list_pending() == [] and mem.load_memory()["preferences"] == {}


def test_observe_skips_known_and_duplicate_and_chatter(mem):
    mem.update_memory({"preferences": {"bullets": {"value": "I prefer answers in bullet points"}}})
    assert mem.observe("I prefer answers in bullet points") == []
    assert mem.observe("what's the weather like?") == []
    mem.observe("I never eat breakfast before ten")
    assert mem.observe("I never eat breakfast before ten") == []


def test_relevant_context_is_small_and_on_topic(mem):
    seed(mem)
    ctx = mem.relevant_context("how is my sister doing?", 300)
    assert "Wanjiru" in ctx and "hoodie" not in ctx and len(ctx) <= 340


def test_old_plain_string_entries_still_work(mem):
    (mem.MEMORY_PATH).write_text(json.dumps({"notes": {"legacy": "old plain string"}}))
    assert mem.search_entries("legacy")[0]["value"] == "old plain string"


def test_index_never_serves_a_stale_answer(mem):
    mem.update_memory({"notes": {"colour": {"value": "favourite colour is blue"}}})
    assert mem.search_entries("favourite colour")[0]["value"].endswith("blue")
    mem.update_memory({"notes": {"colour": {"value": "favourite colour is teal"}}})   # same length
    assert mem.search_entries("favourite colour")[0]["value"].endswith("teal")
    mem.delete_entry("colour", "notes")
    assert mem.search_entries("favourite colour") == []

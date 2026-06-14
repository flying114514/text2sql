"""Tests for the agent's response parsing (no LLM/network required).

We can't unit-test the model itself, but we *can* test our parsing layer — the
part most likely to break when a provider returns slightly off-spec output.
"""

from __future__ import annotations

import pytest

from text2sql.agent import parse_generation


def test_parses_clean_json():
    gen = parse_generation('{"reasoning": "r", "sql": "SELECT 1"}')
    assert gen.sql == "SELECT 1"
    assert gen.reasoning == "r"


def test_strips_code_fences_around_json():
    raw = '```json\n{"reasoning": "r", "sql": "SELECT 1"}\n```'
    gen = parse_generation(raw)
    assert gen.sql == "SELECT 1"


def test_extracts_json_when_surrounded_by_prose():
    raw = 'Sure! Here you go: {"reasoning": "r", "sql": "SELECT 1"} hope it helps'
    gen = parse_generation(raw)
    assert gen.sql == "SELECT 1"


def test_strips_trailing_semicolon_and_fences_in_sql():
    raw = '{"reasoning": "r", "sql": "```sql\\nSELECT 1;\\n```"}'
    gen = parse_generation(raw)
    assert gen.sql == "SELECT 1"


def test_raises_on_non_json():
    with pytest.raises(ValueError):
        parse_generation("this is not json at all")

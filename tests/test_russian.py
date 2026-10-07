"""Tests for Russian language support: phonetics, visemes, numbers, guard, and Whisper."""
import numpy as np
import pytest

from lipflow.cleanup import Cleaner, basic_cleanup, numbers_to_digits
from lipflow.confidence import Hypothesis, Quality, assess
from lipflow.guard import check
from lipflow.russian import RUSSIAN_PRACTICE, numbers_to_digits_ru, visemes
from lipflow.text import contains, has_cyrillic, tokens
from lipflow.whisper import WhisperASR


def test_has_cyrillic_and_tokens():
    assert has_cyrillic("Привет")
    assert has_cyrillic("hello мир")
    assert not has_cyrillic("hello world")

    t = tokens("Привет, как твои дела-то?")
    assert "привет" in t
    assert "как" in t
    assert "дела-то" in t


def test_contains():
    assert contains("Привет, Анна!", "Анна")
    assert contains("Добрый день, Иван Петрович", "Иван")
    assert not contains("Добрый день, Иван Петрович", "Сергей")


def test_russian_visemes():
    assert visemes("мама") == "MAMA" or visemes("мама") == "MA"
    assert "M" in visemes("папа")
    assert "F" in visemes("фото")
    assert "SH" in visemes("хорошо")


def test_russian_numbers_to_digits():
    assert numbers_to_digits_ru("тысяча девятьсот сорок третий год") == "1943 год"
    assert numbers_to_digits_ru("двадцать пять рублей") == "25 рублей"
    assert numbers_to_digits_ru("одиннадцать человек") == "11 человек"
    assert numbers_to_digits_ru("триста шестьдесят пять дней") == "365 дней"
    # Unified dispatcher in cleanup
    assert numbers_to_digits("двадцать пять") == "25"


def test_russian_practice_sentences():
    assert len(RUSSIAN_PRACTICE) == 60
    for s in RUSSIAN_PRACTICE:
        assert isinstance(s, str)
        assert len(s.split()) >= 3


def test_russian_basic_cleanup():
    assert basic_cleanup("привет как дела") == "Привет как дела."
    assert basic_cleanup("где ты находишься") == "Где ты находишься?"
    assert basic_cleanup("кто это сделал") == "Кто это сделал?"
    assert basic_cleanup("двадцать первое мая") == "21 мая."


def test_russian_guard_checks():
    # Negation insertion should be caught
    warnings = check("Я пойду на встречу", "Я не пойду на встречу")
    assert any("Отрицание" in w or "Negation" in w for w in warnings)

    # Number change should be caught
    warnings = check("Нам нужно пять штук", "Нам нужно десять штук")
    assert any("Числа" in w or "Numbers" in w for w in warnings)

    # Date/amount change should be caught
    warnings = check("Встреча в понедельник", "Встреча во вторник")
    assert any("Дата" in w or "Date" in w for w in warnings)

    # Legitimate casing/number normalization should pass
    clean_warnings = check("двадцать пять рублей", "25 рублей.")
    assert len(clean_warnings) == 0


def test_russian_confidence_assessment():
    q_good = Quality(face_ratio=0.9, brightness=120.0, contrast=25.0, mouth_pixels=80.0)
    hyps = [Hypothesis("Привет мир", -1.0, 2), Hypothesis("Привет май", -5.0, 2)]

    # In review policy, it should ask user to choose
    a_review = assess(hyps, "Привет мир", q_good, policy="review", language="ru")
    assert a_review.action == "review"

    # In auto policy with good margin and agreement, it should auto-approve
    a_auto = assess(hyps, "Привет мир", q_good, policy="auto", min_margin=0.5, language="ru")
    assert a_auto.action == "auto"

    # Bad face ratio should request retry
    q_bad = Quality(face_ratio=0.4, brightness=120.0, contrast=25.0, mouth_pixels=80.0)
    a_bad = assess(hyps, "Привет мир", q_bad, language="ru")
    assert a_bad.action == "retry"
    assert "камер" in a_bad.reason


def test_whisper_asr_empty():
    asr = WhisperASR(language="ru")
    # Silence or invalid audio should return empty list
    assert asr.hypotheses(None) == []
    assert asr.hypotheses(np.zeros(100, dtype=np.float32)) == []
    assert asr.hypotheses(np.zeros(16000, dtype=np.float32)) == []


def test_linux_mic_recording():
    import time
    from lipflow.mic import Mic, segment
    m = Mic()
    m.start()
    assert m.error is None
    time.sleep(0.3)
    chunks = m.stop()
    assert len(chunks) > 0
    wave = segment(chunks, chunks[0][0], 8)
    assert wave is not None

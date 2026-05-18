from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
INDEX = WEB_DIR / "index.html"
MODEL_CARD = WEB_DIR / "model-card.html"
STYLES = WEB_DIR / "styles.css"
SCRIPT = WEB_DIR / "script.js"


def test_homepage_assets_exist() -> None:
    assert INDEX.is_file()
    assert MODEL_CARD.is_file()
    assert STYLES.is_file()
    assert SCRIPT.is_file()


def test_homepage_exposes_release_synced_version_badge() -> None:
    html = INDEX.read_text(encoding="utf-8")

    assert "DevOps Incident Triage Model" in html
    assert "v0.3.0" in html
    assert "Release synced" in html
    assert "GitHub Release" in html
    assert 'data-release-version' in html
    assert 'data-release-status' in html


def test_homepage_links_to_web_model_card() -> None:
    html = INDEX.read_text(encoding="utf-8")

    assert 'href="./model-card.html"' in html
    assert 'href="../docs/model_card.md"' not in html


def test_brand_name_links_back_to_home() -> None:
    for page in (INDEX, MODEL_CARD):
        html = page.read_text(encoding="utf-8")

        assert 'class="brand-home"' in html
        assert 'href="./index.html"' in html


def test_language_dropdown_is_available_before_theme_toggle() -> None:
    html = INDEX.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")

    assert html.index("language-select") < html.index("data-theme-toggle")
    assert 'data-language-select' in html
    assert 'option value="ko"' in html
    assert 'option value="en"' in html
    assert "KR 한국어" in html
    assert "EN English" in html
    assert "LANGUAGE_STORAGE_KEY" in script
    assert "function applyLanguage(language)" in script
    assert "data-i18n" in script


def test_korean_translation_copy_is_natural() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "알림이 장애로 번지기 전에 인시던트를 먼저 라우팅하세요" in script
    assert "로그, 알림, 배포 실패 내용을 분석해 담당 팀과 다음 조치로 연결합니다." in script
    assert "릴리즈 동기화됨" in script
    assert "사람의 검토를 거쳐야 하는 1차 분류 보조 도구" in script
    assert "자율적인 운영 조치나 복구 명령을 직접 실행하는 용도" in script


def test_model_card_page_contains_trust_sections() -> None:
    html = MODEL_CARD.read_text(encoding="utf-8")

    assert "Model Card" in html
    assert "DevOps Incident Triage Model" in html
    assert "first-pass triage support" in html
    assert "not autonomous decision-making" in html
    assert "Intended use" in html
    assert "Not intended for" in html
    assert "Data disclosure" in html
    assert "How to use" in html
    assert "Human review policy" in html
    assert "Limitations" in html
    assert "GitHub Release" in html
    assert "data-release-version" in html


def test_homepage_fetches_latest_github_release() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "api.github.com/repos/dongkoony/DevOps-Incident-Triage-Model/releases/latest" in script
    assert "data-release-version" in script
    assert "data-release-status" in script


def test_theme_toggle_is_safe_for_file_url_storage_limits() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "function safeGetStoredTheme()" in script
    assert "function safeSetStoredTheme(theme)" in script
    assert "localStorage.getItem" in script
    assert "localStorage.setItem" in script
    assert "try {" in script
    assert "aria-pressed" in script
    assert 'document.documentElement.dataset.theme = theme' in script


def test_theme_transition_animation_contract() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    styles = STYLES.read_text(encoding="utf-8")

    assert "THEME_TRANSITION_MS" in script
    assert "function startThemeTransition()" in script
    assert "theme-transitioning" in script
    assert "setTimeout" in script
    assert ".theme-transitioning" in styles
    assert "prefers-reduced-motion: no-preference" in styles

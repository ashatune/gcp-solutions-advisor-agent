"""Automated evaluation harness verifying accuracy, grounding, model routing, PII redaction, and tone."""

import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.coordinator import GCPSolutionsCoordinator
from agent.telemetry.logger import get_logger

logger = get_logger("eval.runner")


async def evaluate_scenario(
    coordinator: GCPSolutionsCoordinator,
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single golden benchmark scenario and score accuracy, grounding, tone, and PII scrubbing."""
    scenario_id = scenario["scenario_id"]
    response = await coordinator.handle_inquiry(
        session_id=f"eval-session-{scenario_id}",
        customer_query=scenario["customer_query"],
    )
    await coordinator.session_store.flush_background_tasks()

    text = response.response_text
    lower_text = text.lower()

    # 1. Accuracy Check: Expected technical keywords present
    missing_keywords = [
        kw for kw in scenario.get("expected_keywords", []) if kw.lower() not in lower_text
    ]
    accuracy_passed = len(missing_keywords) == 0

    # 2. Grounding Check: Guardrail passed & expected official URL cited
    expected_url = scenario.get("expected_citation_prefix", "")
    grounding_passed = bool(
        response.guardrail_result
        and response.guardrail_result.passed
        and any(url.startswith(expected_url) for url in response.citations)
    )

    # 3. Strategic Model Routing Check
    routing_passed = response.routed_model == scenario.get("expected_model")

    # 4. Tone & Conciseness Check: Executive headings present & word count within non-verbose bound
    word_count = len(text.split())
    max_words = int(scenario.get("max_word_count", 350))
    has_headings = "### Executive Summary" in text and "### Grounded Citations" in text
    tone_passed = has_headings and (word_count <= max_words)

    # 5. PII Safety Check: Ensure any PII in input query was scrubbed from output
    leaked_pii = [
        forbidden
        for forbidden in scenario.get("forbidden_strings", [])
        if forbidden in text
    ]
    pii_passed = len(leaked_pii) == 0

    overall_passed = all(
        [accuracy_passed, grounding_passed, routing_passed, tone_passed, pii_passed]
    )

    return {
        "scenario_id": scenario_id,
        "title": scenario["title"],
        "overall_passed": overall_passed,
        "routed_model": response.routed_model,
        "expected_model": scenario.get("expected_model"),
        "grounding_score": (
            response.guardrail_result.grounding_score if response.guardrail_result else 0.0
        ),
        "word_count": word_count,
        "checks": {
            "accuracy_passed": accuracy_passed,
            "grounding_passed": grounding_passed,
            "routing_passed": routing_passed,
            "tone_passed": tone_passed,
            "pii_passed": pii_passed,
        },
        "missing_keywords": missing_keywords,
        "leaked_pii": leaked_pii,
    }


async def run_golden_evaluation(dataset_path: Path | None = None) -> Dict[str, Any]:
    """Execute all scenarios in `eval/golden_dataset.json` and return an aggregate report."""
    resolved_path = dataset_path or (Path(__file__).resolve().parent / "golden_dataset.json")
    scenarios: List[Dict[str, Any]] = json.loads(resolved_path.read_text(encoding="utf-8"))

    coordinator = GCPSolutionsCoordinator()
    results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        res = await evaluate_scenario(coordinator, scenario)
        results.append(res)

    passed_count = sum(1 for r in results if r["overall_passed"])
    summary = {
        "total_scenarios": len(results),
        "passed_scenarios": passed_count,
        "pass_rate": round(passed_count / max(1, len(results)), 2),
        "results": results,
    }
    logger.info(
        f"Completed Golden Dataset Evaluation: {passed_count}/{len(results)} scenarios passed.",
        metadata={"pass_rate": summary["pass_rate"]},
    )
    return summary


def main() -> int:
    """CLI entrypoint for running automated regression evaluations in CI/CD."""
    report = asyncio.run(run_golden_evaluation())
    print("\n=== GCP Solutions Advisor Agent: Golden Evaluation Report ===")
    for item in report["results"]:
        status_label = "PASS" if item["overall_passed"] else "FAIL"
        print(
            f"[{status_label}] {item['scenario_id']} ({item['title']}) | "
            f"Model: {item['routed_model']} | Grounding Score: {item['grounding_score']} | "
            f"Words: {item['word_count']}"
        )
    print(
        f"=== Summary: {report['passed_scenarios']}/{report['total_scenarios']} Passed "
        f"({report['pass_rate'] * 100:.0f}%) ===\n"
    )
    return 0 if report["passed_scenarios"] == report["total_scenarios"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
